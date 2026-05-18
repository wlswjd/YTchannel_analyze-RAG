"""Cross-encoder 기반 Re-ranking 모듈 (영상 단위).

입력: hybrid_search가 반환한 dict 리스트 (video_id 단위로 집계된 영상 후보)
출력: Cross-encoder 점수로 재정렬된 상위 top_k개 dict (원본 구조 그대로)
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

_RERANKER_MODEL = "Dongjin-kr/ko-reranker"
_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_RERANKER_MODEL)
    return _model


def _build_rerank_text(candidate: dict) -> str:
    """영상 dict에서 Cross-encoder 입력용 텍스트 구성."""
    parts: list[str] = []

    title = (candidate.get("title") or "").strip()
    if title:
        parts.append(title)

    description = (candidate.get("description") or "").strip()
    if description:
        parts.append(description[:300])

    snippets = candidate.get("_snippets") or []
    for s in snippets[:3]:
        s = (s or "").strip()
        if s:
            parts.append(s[:200])

    if not parts:
        parts.append((candidate.get("text") or "")[:400])

    return " ".join(parts)


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Cross-encoder로 candidates를 재정렬해 상위 top_k개 반환.

    candidates가 비어있거나 모델 로드 실패 시 원본 순서 그대로 반환.
    """
    if not candidates:
        return []

    try:
        model = get_reranker()
        texts = [_build_rerank_text(c) for c in candidates]
        pairs = [(query, t) for t in texts]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        return [d for _, d in ranked[:top_k]]
    except Exception:
        return candidates[:top_k]
