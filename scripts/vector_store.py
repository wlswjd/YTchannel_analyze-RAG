from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402


DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"

# 메타 청크(제목+설명+태그)는 자막보다 정보 신뢰도가 높으므로 점수 가산
META_CHUNK_BOOST = 0.15


_PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


@contextmanager
def _without_proxies():
    """
    Some environments set proxy vars that break HuggingFace downloads (e.g. 403).
    Temporarily remove them just for model loading / downloads.
    """
    old = {k: os.environ.get(k) for k in _PROXY_ENV_KEYS if k in os.environ}
    try:
        for k in _PROXY_ENV_KEYS:
            os.environ.pop(k, None)
        yield
    finally:
        for k in _PROXY_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(old)


@contextmanager
def _hf_cache_in_workspace():
    """
    Ensure HF cache is writable even in sandboxed environments.
    """
    cache_root = (ROOT / ".hf_cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
    ]
    old = {k: os.environ.get(k) for k in keys if k in os.environ}
    try:
        os.environ["HF_HOME"] = str(cache_root)
        os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_root / "sentence-transformers")
        yield
    finally:
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(old)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def chroma_dir() -> Path:
    d = ROOT / "chroma_db"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(chroma_dir()))


# 모델/클라이언트 싱글턴 (검색마다 SentenceTransformer 재로드되는 비용 제거)
_embedder_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    with _without_proxies(), _hf_cache_in_workspace():
        model = SentenceTransformer(model_name)
    _embedder_cache[model_name] = model
    return model


def collection_name(channel_id: str) -> str:
    return f"yt_chunks_{channel_id}"


def upsert_chunks_jsonl(
    *,
    channel_id: str,
    chunks_jsonl: Path,
    channel_label: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
) -> dict:
    """
    Build/Update a Chroma collection from chunks jsonl.
    Rebuilds if collection already exists (simple + deterministic).
    """
    chunks_jsonl = chunks_jsonl.expanduser().resolve()
    if not chunks_jsonl.exists():
        raise FileNotFoundError(f"chunks jsonl not found: {chunks_jsonl}")

    client = get_client()
    name = collection_name(channel_id)

    try:
        client.delete_collection(name)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "model": model_name, "channel": channel_id},
    )

    embedder = get_embedder(model_name)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    total = 0
    meta_count = 0
    transcript_count = 0
    for row in _iter_jsonl(chunks_jsonl):
        text = (row.get("text") or "").strip()
        if not text:
            continue

        chunk_kind = row.get("chunk_kind") or ("meta" if row.get("chunk_index", 0) == -1 else "transcript")

        ids.append(str(row.get("chunk_id") or f"{row.get('video_id','')}_{row.get('chunk_index',0)}"))
        docs.append(text)
        episode_hint = row.get("episode_hint")
        if episode_hint is None:
            episode_hint = ""
        metas.append(
            {
                "channel_id": str(channel_id),
                "channel_label": str(channel_label),
                "video_id": str(row.get("video_id") or ""),
                "title": str(row.get("title") or ""),
                "description": str(row.get("description") or "")[:1500],
                "episode_hint": str(episode_hint),
                "published_at": str(row.get("published_at") or ""),
                "video_url": str(row.get("video_url") or ""),
                "view_count": int(row.get("view_count") or 0),
                "duration_sec": int(row.get("duration_sec") or 0),
                "chunk_kind": chunk_kind,
                "chunk_index": int(row.get("chunk_index") or 0),
            }
        )

        if chunk_kind == "meta":
            meta_count += 1
        else:
            transcript_count += 1

        if len(ids) >= batch_size:
            embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            total += len(ids)
            ids, docs, metas = [], [], []

    if ids:
        embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        total += len(ids)

    return {
        "channel_id": channel_id,
        "collection": name,
        "chunks_indexed": total,
        "meta_chunks": meta_count,
        "transcript_chunks": transcript_count,
        "db_dir": str(chroma_dir()),
    }


def semantic_search(
    *,
    channel_ids: list[str],
    query: str,
    n_results: int,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """
    의미 검색 후 영상(video_id) 단위로 점수 집계.
    같은 영상의 여러 청크가 매치되면 가장 높은 점수의 청크를 대표로 사용하고,
    매치 개수에 비례한 보너스를 더해 영상이 더 위로 올라오게 함.
    메타 청크(제목+설명+태그)는 신뢰도 가중치 추가.
    """
    q = (query or "").strip()
    if not q:
        return []

    client = get_client()
    embedder = get_embedder(model_name)
    q_emb = embedder.encode([q], normalize_embeddings=True).tolist()[0]

    per_video: dict[str, dict] = {}

    for ch_id in channel_ids:
        name = collection_name(ch_id)
        try:
            col = client.get_collection(name)
        except Exception:
            continue

        # 영상 단위 집계를 위해 후보를 넉넉히 가져옴
        res = col.query(
            query_embeddings=[q_emb],
            n_results=max(20, n_results * 6),
            include=["metadatas", "documents", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            score = 1.0 - float(dist)
            kind = meta.get("chunk_kind") or "transcript"
            if kind == "meta":
                score += META_CHUNK_BOOST

            vid = str(meta.get("video_id") or _id.split("_")[0])
            entry = per_video.get(vid)
            payload = {
                "chunk_id": _id,
                "text": doc,
                **meta,
            }

            if entry is None:
                per_video[vid] = {
                    "best_score": score,
                    "match_count": 1,
                    "best": payload,
                    "snippets": [doc[:280]],
                }
            else:
                entry["match_count"] += 1
                if score > entry["best_score"]:
                    entry["best_score"] = score
                    entry["best"] = payload
                if len(entry["snippets"]) < 3:
                    entry["snippets"].append(doc[:280])

    # 영상 단위 최종 점수: best + 매치 개수 보너스(소수)
    scored: list[tuple[float, dict]] = []
    for vid, info in per_video.items():
        bonus = min(0.10, 0.02 * (info["match_count"] - 1))
        final = info["best_score"] + bonus
        out_doc = {
            **info["best"],
            "_score": final,
            "_match_count": info["match_count"],
            "_snippets": info["snippets"],
        }
        scored.append((final, out_doc))

    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:n_results]]
