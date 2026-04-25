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


DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


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


def get_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    with _without_proxies(), _hf_cache_in_workspace():
        return SentenceTransformer(model_name)


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

    # simplest: recreate collection for deterministic results
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
    for row in _iter_jsonl(chunks_jsonl):
        text = (row.get("text") or "").strip()
        if not text:
            continue

        ids.append(str(row.get("chunk_id") or f"{row.get('video_id','')}_{row.get('chunk_index',0)}"))
        docs.append(text)
        episode_hint = row.get("episode_hint")
        if episode_hint is None:
            episode_hint = ""
        metas.append(
            {
                # Chroma metadata values must be primitive (str/int/float/bool).
                "channel_id": str(channel_id),
                "channel_label": str(channel_label),
                "video_id": str(row.get("video_id") or ""),
                "title": str(row.get("title") or ""),
                "episode_hint": str(episode_hint),
                "published_at": str(row.get("published_at") or ""),
                "video_url": str(row.get("video_url") or ""),
                "chunk_index": int(row.get("chunk_index") or 0),
            }
        )

        if len(ids) >= batch_size:
            embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            total += len(ids)
            ids, docs, metas = [], [], []

    if ids:
        embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        total += len(ids)

    return {"channel_id": channel_id, "collection": name, "chunks_indexed": total, "db_dir": str(chroma_dir())}


def semantic_search(
    *,
    channel_ids: list[str],
    query: str,
    n_results: int,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []

    client = get_client()
    embedder = get_embedder(model_name)
    q_emb = embedder.encode([q], normalize_embeddings=True).tolist()[0]

    merged: list[tuple[float, dict]] = []
    for ch_id in channel_ids:
        name = collection_name(ch_id)
        try:
            col = client.get_collection(name)
        except Exception:
            continue

        res = col.query(
            query_embeddings=[q_emb],
            n_results=max(3, n_results * 2),
            include=["metadatas", "documents", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance -> similarity
            score = 1.0 - float(dist)
            merged.append(
                (
                    score,
                    {
                        "chunk_id": _id,
                        "text": doc,
                        **(meta or {}),
                    },
                )
            )

    merged.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[dict] = []
    for score, doc in merged:
        cid = str(doc.get("chunk_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append({**doc, "_score": score})
        if len(out) >= n_results:
            break
    return out

