from __future__ import annotations

import os
from typing import Any


def _format_candidates(candidates: list[dict], max_items: int = 6) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates[:max_items], 1):
        title = (c.get("title") or "")[:140]
        url = c.get("video_url") or ""
        ch = c.get("_channel_label") or c.get("channel_label") or ""
        ep = c.get("episode_hint") or "—"
        excerpt = (c.get("text") or "")[:420].replace("\n", " ").strip()
        lines.append(f"{i}. [{title}]({url}) | 채널:{ch} | EP?:{ep} | 발췌:{excerpt}")
    return "\n".join(lines)


def llm_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def generate_llm_answer(query: str, candidates: list[dict]) -> str | None:
    """
    Returns markdown answer or None if no LLM key.
    Uses Gemini if GEMINI_API_KEY is present.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except Exception:
        return None

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    cand_text = _format_candidates(candidates)
    prompt = f"""너는 유튜브 자막 기반으로 '어떤 회차인지' 찾아주는 도우미야.
사용자의 질문과 검색 후보(자막 발췌 + 제목 + 링크)를 보고 가장 그럴듯한 회차를 골라 자연스럽게 답해줘.

규칙:
- 한국어로, 친절하고 짧게.
- 1순위 후보를 1개 추천하고(링크 포함), 왜 그렇게 봤는지 근거를 1~2문장으로.
- 확신이 낮으면 그 사실을 말하고, 사용자가 추가로 답하면 더 좁힐 수 있도록 확인 질문 1~2개를 해.
- 후보가 애매하면 '상위 후보 2~3개'도 함께 제시.
- 절대 후보에 없는 영상을 지어내지 마.

질문: {query}

후보:
{cand_text}
"""

    try:
        resp = model.generate_content(prompt)
    except Exception:
        return None

    text = getattr(resp, "text", None)
    if not text:
        return None
    return text.strip()


def fallback_answer(query: str, candidates: list[dict]) -> str:
    """
    No-LLM fallback: suggest top candidate and ask clarifying question.
    """
    if not candidates:
        return (
            "찾으시는 회차를 딱 집기 어렵네요.\n\n"
            "- 기억나는 **인물/상황(장소, 먹은 음식, 게임/주제)**\n"
            "- 대사 중 **특정 단어 1~2개**\n\n"
            "이 중 하나만 더 알려주시면 다시 찾아볼게요."
        )

    top = candidates[0]
    title = top.get("title", "")
    url = top.get("video_url", "")
    ch = top.get("_channel_label") or top.get("channel_label") or ""
    excerpt = (top.get("text") or "")[:700]
    return (
        "아마 이 편일 가능성이 있어요(정확도는 아직 낮을 수 있어요).\n\n"
        f"- 추천 후보: [{title}]({url}) · *{ch}*\n"
        f"> {excerpt}\n\n"
        "혹시 **출연자(이름)**, **장소**, **먹은 메뉴**, **대사 키워드** 중 기억나는 게 하나만 더 있을까요? "
        "그걸로 다시 더 정확히 좁혀볼게요."
    )

