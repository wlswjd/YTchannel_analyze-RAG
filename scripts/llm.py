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


def _call(prompt: str, max_tokens: int = 2000) -> str | None:
    """Gemini 호출. 실패 시 None.

    주의: gemini-2.5-flash 계열은 reasoning 모델이라 `max_output_tokens` 안에
    thinking 토큰이 같이 들어감. thinking_budget=0 으로 비활성화하고
    실제 출력 한도를 충분히 줘야 답변이 중간에 잘리지 않는다.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

        # thinking 끄기 시도 (지원하지 않는 모델이면 fallback)
        try:
            cfg = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        except Exception:
            cfg = types.GenerateContentConfig(max_output_tokens=max_tokens)

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg,
        )

        text = resp.text
        # 응답이 토큰 한도로 잘렸는지 감지
        try:
            cand = (resp.candidates or [None])[0]
            finish_reason = getattr(cand, "finish_reason", None) if cand else None
            if finish_reason and str(finish_reason).endswith("MAX_TOKENS") and not text:
                # thinking이 다 먹어버린 경우 — 한 번 더 큰 한도로 재시도
                cfg2 = types.GenerateContentConfig(max_output_tokens=max_tokens * 3)
                resp = client.models.generate_content(
                    model=model_name, contents=prompt, config=cfg2
                )
                text = resp.text
        except Exception:
            pass

        return text.strip() if text else None
    except Exception:
        return None


# ── 룰 기반 Intent / 날짜 / 채널 추출 ─────────────────────────

_ANALYTICS_KEYWORDS = ["분석", "통계", "트렌드", "업로드", "조회수 추이", "몇 편", "기간", "현황"]

# 인물·콘텐츠·채널이 "무엇/누구"인지 묻는 정보 질문 신호
_CONCEPT_QA_KEYWORDS = [
    "뭐야", "뭐임", "뭐지", "뭐예요", "뭐인가요", "뭐인가",
    "누구야", "누구임", "누구지", "누구예요", "누구인가",
    "어떤 채널", "어떤 콘텐츠", "어떤 프로그램", "어떤 방송",
    "설명해", "알려줘",
]

# concept_qa 키워드가 잡혀도 회차/에피소드 검색 의도가 분명하면 episode_search로 폴백
_EPISODE_OVERRIDE_KEYWORDS = ["회차", "에피소드", "ep.", "ep ", "몇 회"]

_CHANNEL_MAP: dict[str, str] = {
    "뜬뜬": "ddeunddeun",
    "핑계고": "ddeunddeun",
    "쑥쑥": "ssookssook",
    "쑥덕": "ssookssook",
    "채널십오야": "channel15ya",
    "십오야": "channel15ya",
    "15야": "channel15ya",
}


def _rule_intent(query: str) -> str:
    q = query.lower()
    if any(kw in query for kw in _ANALYTICS_KEYWORDS):
        return "analytics"
    if any(kw in query for kw in _CONCEPT_QA_KEYWORDS):
        # "회차 알려줘" / "EP.78 알려줘" 등은 영상 찾기로 폴백
        if any(kw in q for kw in _EPISODE_OVERRIDE_KEYWORDS):
            return "episode_search"
        return "concept_qa"
    return "episode_search"


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
            "intent": "episode_search" | "concept_qa" | "analytics",
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
    prompt = f"""사용자 질문을 분석해서 JSON만 반환해. 설명 없이 JSON 딱 하나만.

채널 이름 → id: 뜬뜬→ddeunddeun, 쑥쑥→ssookssook, 채널십오야(=십오야,15야)→channel15ya
현재 연도: 2026년

{{
  "intent": "episode_search" | "concept_qa" | "analytics",
  "channel_ids": ["id"] 또는 null,
  "date_from": "YYYY-MM" 또는 null,
  "date_to": "YYYY-MM" 또는 null
}}

intent 분류 기준:
- analytics: 분석/통계/트렌드/몇 편/기간/조회수 추이 등 수치 기반 질문
  예) "뜬뜬 25년 1월부터 분석해줘", "월별 업로드 수 알려줘"
- concept_qa: 인물·콘텐츠·채널이 "무엇/누구"인지 묻는 정보·설명 질문
  (특정 영상 한 편을 찾는 게 아니라, 검색된 영상들의 정보를 종합해 답해야 함)
  예) "풍향고가 뭐야?", "유재석이 누구야?", "유재석이 진행하는 채널은?"
- episode_search: 특정 에피소드·장면·회차를 찾아달라는 질문
  예) "이서진이랑 떡국 끓여먹던 편이 뭐였지?", "박보검 나온 핑계고 회차"

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

    # intent 화이트리스트 강제
    if result.get("intent") not in {"episode_search", "concept_qa", "analytics"}:
        result["intent"] = "episode_search"

    return result


# ── 에피소드 검색 답변 ─────────────────────────────────────────

def generate_episode_answer(query: str, candidates: list[dict]) -> str:
    if not candidates:
        return (
            f"「{query}」와 관련된 영상을 찾지 못했어요.\n\n"
            "혹시 기억나는 **출연자 이름**, **장소**, **음식이나 소품**, **대사 키워드** 중 "
            "하나만 더 알려주시면 다시 찾아볼게요."
        )

    if not llm_available():
        return _fallback_episode(query, candidates)

    seen_ids: set[str] = set()
    unique_candidates: list[dict] = []
    for c in candidates:
        vid = c.get("video_id") or c.get("chunk_id", "")
        if vid not in seen_ids:
            seen_ids.add(vid)
            unique_candidates.append(c)

    top = unique_candidates[:6]
    n = len(top)

    lines = []
    for i, c in enumerate(top, 1):
        title = (c.get("title") or "")[:120]
        url = c.get("video_url") or ""
        pub = (c.get("published_at") or "")[:10]
        desc = (c.get("description") or "")[:400].replace("\n", " ").strip()
        snippets = c.get("_snippets") or []
        if snippets:
            excerpt = " / ".join(s.replace("\n", " ").strip() for s in snippets[:2])[:500]
        else:
            excerpt = (c.get("text") or "")[:300].replace("\n", " ").strip()
        score = c.get("_score")
        score_str = f" (관련도 {score:.2f})" if isinstance(score, (int, float)) else ""
        meta_line = f"   업로드: {pub}" if pub else ""
        lines.append(
            f"{i}. 제목: {title}{score_str}\n"
            f"   링크: {url}\n"
            f"{meta_line}\n"
            f"   영상 설명: {desc if desc else '(없음)'}\n"
            f"   자막 발췌: {excerpt if excerpt else '(자막 매치 없음)'}"
        )
    cand_text = "\n\n".join(lines)

    if n == 1:
        answer_guide = (
            "후보가 1개뿐이야. 아래 형식으로 답해줘:\n"
            "1) 첫 줄: '찾으시는 영상은 **[제목](링크)** 인 것 같아요.'\n"
            "2) 다음 단락: 이 영상이 어떤 내용인지 영상 설명·자막 근거로 2~3문장 자연스럽게 소개\n"
            "3) 마지막 줄: 업로드 날짜 한 줄 (있으면)"
        )
    else:
        answer_guide = (
            f"후보가 {n}개야. 점수가 가장 높은 1번이 정답일 가능성이 가장 높아.\n"
            "아래 형식으로 답해줘:\n"
            "1) 첫 줄: '찾으시는 영상은 **[1번 제목](1번 링크)** 인 것 같아요.'\n"
            "2) 다음 단락: 이 영상이 어떤 내용인지 영상 설명·자막 근거로 2~3문장 자연스럽게 소개\n"
            "3) '혹시 이 영상이 아니라면, 아래 후보도 확인해보세요:' 라고 적고\n"
            "   2번부터 4번까지를 마크다운 리스트로 (번호 없이) '- [제목](링크) — 1줄 요약' 형태로 제시\n"
            "4) 단, 1번의 점수가 명백히 낮거나 질문과 동떨어져 보이면 솔직히 '정확히 일치하는 영상은 못 찾았는데, 비슷한 후보로는…'이라고 밝혀도 돼"
        )

    prompt = f"""너는 유튜브 채널 영상을 찾아주는 도우미야.
사용자 질문에 가장 잘 맞는 영상을 추천하는 게 임무야.

판단 자료:
- 영상 제목 (가장 신뢰도 높음)
- 영상 설명(description) — 출연자/회차/줄거리/해시태그가 들어있는 핵심 정보
- 자막 발췌 — 실제 대사. 사람 이름은 잘못 들리거나 다르게 적힐 수 있음
- 관련도 점수 — 0~1.2 사이, 높을수록 의미상 가까움

원칙:
- 한국어, 친근하고 간결하게
- 링크는 반드시 [제목](링크) 마크다운으로
- 후보에 없는 영상은 절대 지어내지 마
- 출연자 이름은 자막의 음차 표기보다 영상 설명·태그를 우선시해

{answer_guide}

질문: {query}

검색 후보:
{cand_text}"""

    result = _call(prompt, max_tokens=2000)
    if result and len(result.strip()) >= 20:
        return result
    return _fallback_episode(query, unique_candidates)


def _fallback_episode(query: str, candidates: list[dict]) -> str:
    if not candidates:
        return "관련 영상을 찾지 못했어요. 다른 키워드로 다시 시도해보세요."

    seen_ids: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        vid = c.get("video_id") or c.get("chunk_id", "")
        if vid not in seen_ids:
            seen_ids.add(vid)
            unique.append(c)

    top = unique[0]
    title = top.get("title", "")
    url = top.get("video_url", "")
    desc = (top.get("description") or "")[:200]

    head = f"찾으시는 영상은 **[{title}]({url})** 인 것 같아요."
    body = f"\n\n> {desc}" if desc else ""

    if len(unique) == 1:
        return head + body

    others = []
    for c in unique[1:5]:
        t = c.get("title", "")
        u = c.get("video_url", "")
        pub = (c.get("published_at") or "")[:10]
        date_str = f" ({pub})" if pub else ""
        others.append(f"- [{t}]({u}){date_str}")

    tail = "\n\n혹시 이 영상이 아니라면, 아래 후보도 확인해보세요:\n" + "\n".join(others)
    return head + body + tail


# ── 개념/인물 설명 답변 (concept_qa) ───────────────────────────

def generate_concept_answer(query: str, candidates: list[dict]) -> str:
    """
    "풍향고가 뭐야?", "유재석이 누구?" 같은 개념·인물 질문에 대해
    검색된 영상들의 정보(제목/설명/태그)를 종합해 풀어서 답변.

    episode_search와 달리 영상 한 편을 찾는 게 아니라,
    후보들의 메타데이터를 종합해 자연스러운 설명을 만든 뒤
    본문 맨 아래에 '관련 영상' 1~2개를 마크다운 링크로 첨부.
    """
    if not candidates:
        return (
            f"「{query}」 관련 정보를 찾지 못했어요.\n\n"
            "조금 더 구체적인 키워드(인물 이름, 콘텐츠 이름, 채널 등)로 다시 물어봐 주세요."
        )

    if not llm_available():
        return _fallback_concept(query, candidates)

    seen_ids: set[str] = set()
    unique_candidates: list[dict] = []
    for c in candidates:
        vid = c.get("video_id") or c.get("chunk_id", "")
        if vid not in seen_ids:
            seen_ids.add(vid)
            unique_candidates.append(c)

    # 개념 질문은 종합이 핵심 — 후보를 episode_search보다 많이 보여줌
    top = unique_candidates[:8]

    lines = []
    for i, c in enumerate(top, 1):
        title = (c.get("title") or "")[:140]
        url = c.get("video_url") or ""
        pub = (c.get("published_at") or "")[:10]
        desc = (c.get("description") or "")[:600].replace("\n", " ").strip()
        tags_raw = c.get("tags") or []
        if isinstance(tags_raw, list):
            tags_text = ", ".join(str(t) for t in tags_raw[:20])
        else:
            tags_text = str(tags_raw)[:300]
        snippets = c.get("_snippets") or []
        if snippets:
            excerpt = " / ".join(s.replace("\n", " ").strip() for s in snippets[:2])[:400]
        else:
            excerpt = (c.get("text") or "")[:250].replace("\n", " ").strip()
        channel = c.get("_channel_label") or c.get("channel_label", "")
        score = c.get("_score")
        score_str = f" (관련도 {score:.2f})" if isinstance(score, (int, float)) else ""

        lines.append(
            f"{i}. 제목: {title}{score_str}\n"
            f"   채널: {channel if channel else '(미상)'}\n"
            f"   링크: {url}\n"
            f"   업로드: {pub if pub else '(미상)'}\n"
            f"   영상 설명: {desc if desc else '(없음)'}\n"
            f"   태그: {tags_text if tags_text else '(없음)'}\n"
            f"   자막 발췌: {excerpt if excerpt else '(없음)'}"
        )
    cand_text = "\n\n".join(lines)

    prompt = f"""너는 유튜브 채널 콘텐츠 정보를 정리해 알려주는 도우미야.
사용자의 질문은 영상 한 편을 찾아달라는 게 아니라,
**인물·콘텐츠·채널이 "무엇/누구"인지** 설명해달라는 질문이야.

판단 자료(중요도 순):
- 영상 제목 — 콘텐츠 명칭과 회차가 가장 정확하게 드러남
- 영상 설명(description) — 출연자/줄거리/콘셉트/해시태그 등 핵심 정보
- 태그 — 콘텐츠 키워드 (출연자·시리즈명 보완)
- 자막 발췌 — 실제 대사. 사람 이름은 음차/오타 가능성 있음
- 관련도 점수 — 0~1.2, 높을수록 의미상 가까움

답변 작성 원칙:
1. **종합**해서 답해. 후보들의 제목·설명·태그를 묶어 콘텐츠의 정체/콘셉트/출연자/시즌을 풀어 설명.
   (한 영상만 콕 집어 "이 영상이에요" 식의 답은 금지)
2. 한국어로 2~5문장. 친근하고 자연스럽게.
3. 핵심 정보(콘텐츠 이름, 채널, 출연자, 콘셉트/포맷, 시즌·시리즈 구분)는 가능한 한 빠뜨리지 마.
4. 검색 결과에 **없는 내용은 절대 지어내지 마**. 정보가 부족하면 "정확히는 확인이 어렵다"고 솔직히 밝혀.
5. 자막 음차보다 영상 설명·태그를 우선시해 (예: 자막의 "보검이" → 설명의 "박보검").
6. 본문 끝에 빈 줄을 두고 다음 형식으로 가장 관련 높은 영상 1~2개를 첨부:
   ```
   **관련 영상**
   - [영상 제목](영상 링크)
   - [영상 제목](영상 링크)
   ```
   (후보에 없는 영상은 절대 만들지 말 것)

질문: {query}

검색 후보:
{cand_text}"""

    result = _call(prompt, max_tokens=2000)
    if result and len(result.strip()) >= 30:
        return result
    return _fallback_concept(query, unique_candidates)


def _fallback_concept(query: str, candidates: list[dict]) -> str:
    if not candidates:
        return "관련 정보를 찾지 못했어요. 다른 키워드로 다시 시도해 주세요."

    seen_ids: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        vid = c.get("video_id") or c.get("chunk_id", "")
        if vid not in seen_ids:
            seen_ids.add(vid)
            unique.append(c)

    head = f"「{query}」 관련 영상에서 찾은 정보를 정리했어요.\n\n"

    bullets: list[str] = []
    for c in unique[:3]:
        title = c.get("title") or ""
        desc = (c.get("description") or "")[:180].replace("\n", " ").strip()
        if title and desc:
            bullets.append(f"- **{title}** — {desc}…")
        elif title:
            bullets.append(f"- **{title}**")

    body = "\n".join(bullets) if bullets else "_(영상 설명을 가져오지 못했어요.)_"

    related_lines = []
    for c in unique[:2]:
        t = c.get("title", "")
        u = c.get("video_url", "")
        if t and u:
            related_lines.append(f"- [{t}]({u})")
    related = ""
    if related_lines:
        related = "\n\n**관련 영상**\n" + "\n".join(related_lines)

    return head + body + related


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
