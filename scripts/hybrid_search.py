"""
Dense(ChromaDB) + BM25 Hybrid Search with RRF.

검색 흐름:
1. Dense 검색 — ChromaDB에서 채널별 Top-N 청크
2. BM25 검색 — 채널별 pickle 인덱스로 동일 쿼리 검색
3. RRF(k=60) 결합 — 청크 단위 순위 통합
4. video_id 단위 집계 — RRF 점수 합산 + 메타 청크 가중치 +0.15
5. Top-K 영상 반환

BM25 인덱스(data/bm25/bm25_{channel_id}.pkl)가 없는 채널은
Dense 전용으로 자동 폴백.
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from reranker import rerank as _rerank  # noqa: E402
from vector_store import (  # noqa: E402
    DEFAULT_MODEL,
    META_CHUNK_BOOST,
    _iter_jsonl,
    collection_name,
    get_client,
    get_embedder,
)

BM25_DIR = ROOT / "data" / "bm25"
RRF_K = 60
TOP_N = 30          # Dense / BM25 각각에서 가져올 청크 수
RERANK_CANDIDATES = 20  # video_id 집계 후 Re-ranking에 넘길 후보 영상 수


# ---------------------------------------------------------------------------
# 토큰화
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """한국어+영숫자 단순 토큰화 (형태소 분석 없이 공백/구두점 분리)."""
    tokens = re.split(r"[^가-힣a-z0-9]+", (text or "").lower())
    return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 인덱스 빌드 / 로드
# ---------------------------------------------------------------------------

def bm25_path(channel_id: str) -> Path:
    return BM25_DIR / f"bm25_{channel_id}.pkl"


def build_bm25_index(*, channel_id: str, chunks_jsonl: Path) -> Path:
    """채널 chunks jsonl로 BM25Okapi 인덱스를 빌드하고 pickle로 저장."""
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    ids, tokens, docs, metas = [], [], [], []
    for row in _iter_jsonl(chunks_jsonl):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        chunk_type = row.get("chunk_type") or (
            "metadata" if row.get("chunk_kind") == "meta" else "transcript"
        )
        ids.append(str(row.get("chunk_id") or ""))
        tokens.append(_tokenize(text))
        docs.append(text)
        metas.append(
            {
                "video_id": str(row.get("video_id") or ""),
                "title": str(row.get("title") or ""),
                "chunk_type": chunk_type,
                "chunk_index": int(row.get("chunk_index") or 0),
                "published_at": str(row.get("published_at") or ""),
                "upload_date": int(row.get("upload_date") or 0),
                "video_url": str(row.get("video_url") or ""),
                "view_count": int(row.get("view_count") or 0),
                "duration_sec": int(row.get("duration_sec") or 0),
                "episode_hint": str(row.get("episode_hint") or ""),
                "description": str(row.get("description") or "")[:1500],
            }
        )

    bm25 = BM25Okapi(tokens, k1=1.5, b=0.75)
    out = bm25_path(channel_id)
    with out.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "docs": docs, "metas": metas}, f)

    print(f"BM25 인덱스 저장: {out} ({len(ids)}개 청크)")
    return out


def _load_bm25(channel_id: str) -> Optional[dict]:
    p = bm25_path(channel_id)
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------

def hybrid_search(
    *,
    channel_ids: list[str],
    query: str,
    n_results: int,
    model_name: str = DEFAULT_MODEL,
    use_rerank: bool = True,
) -> list[dict]:
    """
    Dense + BM25 → RRF → video_id 집계 → (Re-ranking) → Top-K 반환.
    채널별로 BM25 인덱스가 없으면 Dense 전용 폴백.
    use_rerank=False 시 집계 점수 순서 그대로 반환 (평가 비교용).
    """
    q = (query or "").strip()
    if not q:
        return []

    client = get_client()
    embedder = get_embedder(model_name)
    q_emb = embedder.encode([q], normalize_embeddings=True).tolist()[0]
    q_tokens = _tokenize(q)

    # 청크 단위 RRF 점수 누적
    chunk_scores: dict[str, float] = {}
    chunk_payloads: dict[str, dict] = {}

    for ch_id in channel_ids:
        # --- Dense ---
        try:
            col = client.get_collection(collection_name(ch_id))
            res = col.query(
                query_embeddings=[q_emb],
                n_results=TOP_N,
                include=["metadatas", "documents", "distances"],
            )
            dense = zip(
                (res.get("ids") or [[]])[0],
                (res.get("documents") or [[]])[0],
                (res.get("metadatas") or [[]])[0],
                (res.get("distances") or [[]])[0],
            )
            for rank, (cid, doc, meta, _dist) in enumerate(dense):
                meta = meta or {}
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + _rrf_score(rank)
                if cid not in chunk_payloads:
                    chunk_payloads[cid] = {"chunk_id": cid, "text": doc, **meta}
        except Exception:
            pass

        # --- BM25 (인덱스 없으면 건너뜀) ---
        bm25_data = _load_bm25(ch_id)
        if bm25_data and q_tokens:
            scores_arr = bm25_data["bm25"].get_scores(q_tokens)
            top_indices = sorted(range(len(scores_arr)), key=lambda i: -scores_arr[i])[:TOP_N]
            rank = 0
            for idx in top_indices:
                if scores_arr[idx] <= 0:
                    break
                cid = bm25_data["ids"][idx]
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + _rrf_score(rank)
                if cid not in chunk_payloads:
                    meta = {**bm25_data["metas"][idx], "channel_id": ch_id}
                    chunk_payloads[cid] = {"chunk_id": cid, "text": bm25_data["docs"][idx], **meta}
                rank += 1

    # --- video_id 단위 집계 (기존 로직과 동일 구조) ---
    per_video: dict[str, dict] = {}

    for cid, rrf in chunk_scores.items():
        payload = chunk_payloads[cid]
        kind = payload.get("chunk_type") or payload.get("chunk_kind") or "transcript"
        score = rrf + (META_CHUNK_BOOST if kind in ("metadata", "meta") else 0.0)

        vid = str(payload.get("video_id") or cid.split("_")[0])
        entry = per_video.get(vid)
        if entry is None:
            per_video[vid] = {
                "best_score": score,
                "match_count": 1,
                "best": payload,
                "snippets": [payload.get("text", "")[:280]],
            }
        else:
            entry["match_count"] += 1
            if score > entry["best_score"]:
                entry["best_score"] = score
                entry["best"] = payload
            if len(entry["snippets"]) < 3:
                entry["snippets"].append(payload.get("text", "")[:280])

    scored: list[tuple[float, dict]] = []
    for info in per_video.values():
        bonus = min(0.10, 0.02 * (info["match_count"] - 1))
        final = info["best_score"] + bonus
        scored.append((final, {**info["best"], "_score": final, "_match_count": info["match_count"], "_snippets": info["snippets"]}))

    scored.sort(key=lambda x: -x[0])
    candidates = [d for _, d in scored[:RERANK_CANDIDATES]]

    if use_rerank:
        return _rerank(query, candidates, n_results)
    return candidates[:n_results]
