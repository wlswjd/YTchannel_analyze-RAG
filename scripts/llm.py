"""
Gemini API 기반 LLM 기능.
LLM 호출 실패 시 룰 기반 fallback으로 최소 동작 보장.
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


def _call(prompt: str, max_tokens: int = 600) -> str | None:
    """Gemini 호출. 실패 시 None."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        text = resp.text
        return text.strip() if text else None
    except Exception:
        return None


# ── 룰 기반 Intent / 날짜 / 채널 추출 ─────────────────────────

_ANALYTICS_KEYWORDS = ["분석", "통계", "트렌드", "업로드", "조회수 추이", "몇 편", "기간", "현황"]

_CHANNEL_MAP: dict[str, str] = {
    "뜬뜬": "ddeunddeun",
    "핑계고": "ddeunddeun",
    "쑥쑥": "ssookssook",
    "쑥덕": "ssookssook",
    "15야": "channel15ya",
    "십오야": "channel15ya",
}


def _rule_intent(query: str) -> str:
    return "analytics" if any(kw in query for kw in _ANALYTICS_KEYWORDS) else "episode_search"


def _rule_dates(query: str) -> dict:
    """
    날짜 추출. 두 가지 패턴 지원:
    - '25년 7월'  / '2025년 7월'  → 연월 함께
    - '7월부터 9월'              → 앞 연월에서 연도 상속
    """
    result: dict = {"date_from": None, "date_to": None}

    # 1차: 연도+월 패턴
    full_pattern = r"(\d{2,4})년\s*(\d{1,2})월"
    full_matches = re.findall(full_pattern, query)
    dates = []
    last_year: int | None = None
    for y, m in full_matches:
        year = int(y) + (2000 if len(y) == 2 else 0)
        last_year = year
        dates.append(f"{year:04d}-{int(m):02d}")

    # 2차: 연도 없는 단독 월 (앞 연도 상속)
    if last_year and len(dates) < 2:
        # "7월부터 9월" 중 아직 못 잡은 단독 월
        solo = re.findall(r"(?<!\d)(\d{1,2})월", query)
        for m_str in solo:
            candidate = f"{last_year:04d}-{int(m_str):02d}"
            if candidate not in dates:
                dates.append(candidate)

    if len(dates) >= 2:
        result["date_from"], result["date_to"] = dates[0], dates[1]
    elif len(dates) == 1:
        result["date_from"] = dates[0]
    return result


def _rule_channels(query: str) -> list[str] | None:
    found = [cid for name, cid in _CHANNEL_MAP.items() if name in query]
    return list(dict.fromkeys(found)) or None  # 중복 제거, 없으면 None


# ── Intent 감지 ───────────────────────────────────────────────

def detect_intent(query: str, available_channel_labels: list[str]) -> dict:
    """
    룰 기반으로 먼저 감지 → LLM으로 보정.

    Returns:
        {
            "intent": "episode_search" | "analytics",
            "channel_ids": ["ddeunddeun", ...] | None,
            "date_from": "YYYY-MM" | None,
            "date_to":   "YYYY-MM" | None,
        }
    """
    # 룰 기반 (항상 실행 — LLM 없어도 기본 동작)
    result: dict = {
        "intent": _rule_intent(query),
        "channel_ids": _rule_channels(query),
        "date_from": None,
        "date_to": None,
    }
    result.update(_rule_dates(query))

    if not llm_available():
        return result

    # LLM 보정 (실패해도 룰 기반 결과 유지)
    channels_str = ", ".join(available_channel_labels)
    prompt = f"""사용자 질문을 분석해서 JSON만 반환해. 설명 없이 JSON 딱 하나만.

채널 이름 → id: 뜬뜬→ddeunddeun, 쑥쑥→ssookssook, 15야→channel15ya
현재 연도: 2026년

{{
  "intent": "episode_search" 또는 "analytics",
  "channel_ids": ["id"] 또는 null,
  "date_from": "YYYY-MM" 또는 null,
  "date_to": "YYYY-MM" 또는 null
}}

analytics: 분석/통계/트렌드/몇 편/기간 조회
episode_search: 특정 에피소드·장면 찾기

질문: {query}"""

    text = _call(prompt, max_tokens=200)
    if text:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                llm_result = json.loads(m.group())
                # LLM 결과로 룰 기반 보정 (값이 있을 때만 덮어쓰기)
                for k in ("intent", "channel_ids", "date_from", "date_to"):
                    if llm_result.get(k) is not None:
                        result[k] = llm_result[k]
            except Exception:
                pass

    return result


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
        excerpt = (c.get("text") or "")[:300].replace("\n", " ").strip()
        lines.append(f"{i}. 제목: {title}\n   링크: {url}\n   자막: {excerpt}")
    cand_text = "\n\n".join(lines)

    prompt = f"""너는 유튜브 채널 자막 기반으로 특정 영상을 찾아주는 도우미야.

규칙:
- 한국어로, 친근하고 간결하게
- 가장 가능성 높은 영상 1개 추천 + 링크 포함
- 왜 그 영상인지 자막 근거 1~2문장
- 확신 낮으면 솔직히 말하고 상위 후보 2~3개 제시
- 후보에 없는 영상은 절대 지어내지 마

질문: {query}

검색 후보:
{cand_text}"""

    result = _call(prompt)
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
        f"**[{title}]({url})**\n\n"
        f"> {excerpt}\n\n"
        "출연자 이름, 장소, 음식, 대사 키워드를 더 알려주시면 더 정확히 찾아볼게요."
    )


# ── 채널 분석 요약 ─────────────────────────────────────────────

def generate_analytics_summary(query: str, stats: dict) -> str:
    if not llm_available():
        return _fallback_analytics(stats)

    prompt = f"""유튜브 채널 분석 데이터 기반으로 한국어 인사이트를 마크다운으로 작성해줘.

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

핵심 인사이트 3~5개, 각 항목 1~2줄."""

    result = _call(prompt)
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
