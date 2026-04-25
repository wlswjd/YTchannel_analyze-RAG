"""
Gemini API 기반 LLM 기능.
- detect_intent: 질문이 에피소드 검색인지 채널 분석인지 판단
- generate_episode_answer: 검색 결과 기반 에피소드 추천 답변
- generate_analytics_summary: 채널 통계 기반 분석 요약
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


# ── 클라이언트 ────────────────────────────────────────────────

def llm_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _get_model():
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)


def _call(prompt: str, max_output_tokens: int = 600) -> str | None:
    try:
        import google.generativeai as genai
        model = _get_model()
        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_output_tokens),
        )
        return (resp.text or "").strip() or None
    except Exception:
        return None


# ── Intent 감지 ───────────────────────────────────────────────

def detect_intent(query: str, available_channel_labels: list[str]) -> dict:
    """
    질문 의도 분석.

    Returns:
        {
            "intent": "episode_search" | "analytics",
            "channel_ids": ["ddeunddeun", ...] | None,
            "date_from": "YYYY-MM" | None,
            "date_to":   "YYYY-MM" | None,
        }
    """
    default: dict = {
        "intent": "episode_search",
        "channel_ids": None,
        "date_from": None,
        "date_to": None,
    }

    if not llm_available():
        return default

    channels_str = ", ".join(available_channel_labels)
    prompt = f"""사용자 질문을 분석해서 JSON만 반환해. 설명 없이 JSON 딱 하나만.

사용 가능 채널: {channels_str}
채널 이름 → id:
  뜬뜬 → ddeunddeun
  쑥쑥 → ssookssook
  15야 → channel15ya

반환 형식 (현재 연도 기준: 2026년):
{{
  "intent": "episode_search" 또는 "analytics",
  "channel_ids": ["id1"] 또는 null,
  "date_from": "YYYY-MM" 또는 null,
  "date_to": "YYYY-MM" 또는 null
}}

intent 판단:
- analytics: 분석/통계/트렌드/몇 편/업로드 추이/기간 조회 등
- episode_search: 특정 에피소드·장면·대사 찾기

질문: {query}"""

    text = _call(prompt, max_output_tokens=256)
    if text:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return {**default, **json.loads(m.group())}
            except Exception:
                pass

    return default


# ── 에피소드 검색 답변 ─────────────────────────────────────────

def generate_episode_answer(query: str, candidates: list[dict]) -> str:
    if not candidates:
        return (
            f"「{query}」와 관련된 자막을 찾지 못했어요.\n\n"
            "혹시 기억나는 **출연자 이름**, **장소**, **음식이나 소품**, **대사 키워드** 중 "
            "하나만 더 알려주시면 다시 찾아볼게요."
        )

    if not llm_available():
        return _fallback_episode(query, candidates)

    lines = []
    for i, c in enumerate(candidates[:6], 1):
        title = (c.get("title") or "")[:120]
        url = c.get("video_url") or ""
        excerpt = (c.get("text") or "")[:350].replace("\n", " ").strip()
        lines.append(f"{i}. 제목: {title}\n   링크: {url}\n   자막: {excerpt}")
    cand_text = "\n\n".join(lines)

    prompt = f"""너는 유튜브 채널 자막 데이터 기반으로 특정 영상을 찾아주는 도우미야.

규칙:
- 한국어로, 친근하고 간결하게
- 가장 가능성 높은 영상 1개를 추천하고 링크 포함
- 왜 그 영상인지 자막 근거 1~2문장
- 확신이 낮으면 솔직하게 말하고 상위 후보 2~3개 제시
- 후보에 없는 영상은 절대 지어내지 마

질문: {query}

검색 후보:
{cand_text}"""

    result = _call(prompt, max_output_tokens=600)
    return result if result else _fallback_episode(query, candidates)


def _fallback_episode(query: str, candidates: list[dict]) -> str:
    if not candidates:
        return "관련 영상을 찾지 못했어요. 다른 키워드로 다시 시도해보세요."
    top = candidates[0]
    title = top.get("title", "")
    url = top.get("video_url", "")
    excerpt = (top.get("text") or "")[:300]
    return (
        f"이 영상일 가능성이 있어요.\n\n"
        f"- [{title}]({url})\n\n"
        f"> {excerpt}\n\n"
        "출연자 이름, 장소, 음식, 대사 키워드를 더 알려주시면 더 정확히 찾아볼게요."
    )


# ── 채널 분석 요약 ─────────────────────────────────────────────

def generate_analytics_summary(query: str, stats: dict) -> str:
    if not llm_available():
        return _fallback_analytics(stats)

    prompt = f"""유튜브 채널 분석 데이터를 바탕으로 한국어 인사이트 분석을 마크다운으로 작성해줘.

요청: {query}

데이터:
- 채널: {stats.get('channel_label')}
- 기간: {stats.get('date_from')} ~ {stats.get('date_to')}
- 총 영상: {stats.get('total_videos')}개
- 총 조회수: {stats.get('total_views', 0):,}회
- 평균 조회수: {int(stats.get('avg_views', 0)):,}회
- 월평균 업로드: {stats.get('avg_monthly_uploads', 0):.1f}편
- 가장 활발한 달: {stats.get('best_month')}
- 최고 조회수 영상: {stats.get('top_video_title')} ({stats.get('top_video_views', 0):,}회)
- 조회수 추이: {stats.get('trend_description')}

핵심 인사이트 3~5개를 간결하게. 각 항목 1~2줄."""

    result = _call(prompt, max_output_tokens=600)
    return result if result else _fallback_analytics(stats)


def _fallback_analytics(stats: dict) -> str:
    return (
        f"**{stats.get('channel_label', '')} 분석**\n\n"
        f"- 총 영상: {stats.get('total_videos', 0)}개\n"
        f"- 총 조회수: {stats.get('total_views', 0):,}회\n"
        f"- 평균 조회수: {int(stats.get('avg_views', 0)):,}회\n"
        f"- 월평균 업로드: {stats.get('avg_monthly_uploads', 0):.1f}편\n"
        f"- 조회수 추이: {stats.get('trend_description', '')}\n"
    )
