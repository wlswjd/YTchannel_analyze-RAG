"""
유튜브 채널 봇.
  - 에피소드 검색: "이서진이랑 떡국 끓여먹던 편 뭐였지?" → AI가 관련 영상 찾아줌
  - 채널 분석:    "뜬뜬 채널 25년 1월부터 분석해줘"    → 그래프 + AI 분석 화면 분할

실행:
  streamlit run scripts/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channels import CHANNELS, chunks_path, raw_path  # noqa: E402
from llm import (  # noqa: E402
    detect_intent,
    generate_analytics_summary,
    generate_episode_answer,
    llm_available,
)
from vector_store import get_embedder, semantic_search  # noqa: E402


@st.cache_resource(show_spinner="검색 모델 로드 중...")
def _warm_embedder():
    """SentenceTransformer를 앱 부팅 시 한 번만 로드해 캐시."""
    return get_embedder()

# ── 데이터 로드 ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_raw(path_str: str) -> list[dict]:
    p = Path(path_str)
    return json.loads(p.read_text("utf-8")) if p.exists() else []


@st.cache_data(show_spinner=False)
def _load_chunks(path_str: str) -> list[dict]:
    p = Path(path_str)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


# ── 검색 ────────────────────────────────────────────────────────────────────

def _keyword_search(
    chunks: list[dict], query: str, limit: int, channel_label: str
) -> list[dict]:
    """쿼리 단어 단위 매칭. 제목·설명·자막 가중치를 다르게 줌.
    제목/설명은 인물·테마가 잘 드러나므로 자막보다 높은 가중치."""
    keywords = [w for w in query.split() if len(w) >= 2]
    if not keywords:
        return []
    per_video: dict[str, tuple[float, dict]] = {}
    for c in chunks:
        text = (c.get("text") or "").lower()
        title = (c.get("title") or "").lower()
        desc = (c.get("description") or "").lower()
        score = 0.0
        for kw in keywords:
            kw_l = kw.lower()
            score += title.count(kw_l) * 6
            score += desc.count(kw_l) * 4
            score += text.count(kw_l) * 1
        if score <= 0:
            continue
        vid = c.get("video_id") or c.get("chunk_id", "")
        existing = per_video.get(vid)
        payload = {**c, "_channel_label": channel_label, "_score": score / 10.0}
        if existing is None or score > existing[0]:
            per_video[vid] = (score, payload)
    hits = sorted(per_video.values(), key=lambda x: -x[0])
    return [h[1] for h in hits[:limit]]


def run_episode_search(query: str, enabled: list[dict], n: int = 8) -> list[dict]:
    """의미 검색 시도 → 실패 시 키워드 검색 fallback."""
    channel_ids = [ch["id"] for ch in enabled]

    # 1차: 의미 검색 (ChromaDB)
    try:
        results = semantic_search(channel_ids=channel_ids, query=query, n_results=n)
        if results:
            return results
    except Exception:
        pass

    # 2차: 키워드 검색 fallback
    all_hits: list[dict] = []
    for ch in enabled:
        chunks = _load_chunks(str(chunks_path(ch)))
        if chunks:
            all_hits.extend(_keyword_search(chunks, query, n * 2, ch["label"]))

    seen: set[str] = set()
    out: list[dict] = []
    for item in all_hits:
        cid = str(item.get("chunk_id") or item.get("video_id") or "")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(item)
        if len(out) >= n:
            break
    return out


# ── 채널 분석 통계 계산 ──────────────────────────────────────────────────────

def compute_analytics(
    videos: list[dict], date_from: str | None, date_to: str | None
) -> dict | None:
    filtered = []
    for v in videos:
        pub = (v.get("published_at") or "")[:7]
        if not pub:
            continue
        if date_from and pub < date_from:
            continue
        if date_to and pub > date_to:
            continue
        filtered.append(v)

    if not filtered:
        return None

    # 월별 집계
    monthly: dict[str, dict] = {}
    for v in filtered:
        month = (v.get("published_at") or "")[:7]
        if not month:
            continue
        if month not in monthly:
            monthly[month] = {"uploads": 0, "views": 0}
        monthly[month]["uploads"] += 1
        monthly[month]["views"] += int(v.get("view_count") or 0)

    monthly_stats = [
        {"month": k, "uploads": v["uploads"], "views": v["views"]}
        for k, v in sorted(monthly.items())
    ]

    total_views = sum(int(v.get("view_count") or 0) for v in filtered)

    # TOP 영상
    top_sorted = sorted(filtered, key=lambda x: int(x.get("view_count") or 0), reverse=True)
    top_videos = [
        {
            "title": v["title"][:45] + ("…" if len(v.get("title", "")) > 45 else ""),
            "url": v.get("video_url", ""),
            "views": int(v.get("view_count") or 0),
        }
        for v in top_sorted[:10]
    ]

    # 조회수 구간 분포 (도넛 차트 데이터)
    bucket_defs: list[tuple[str, int, int]] = [
        ("1000만 이상", 10_000_000, 10**12),
        ("100만 ~ 1000만", 1_000_000, 10_000_000),
        ("10만 ~ 100만", 100_000, 1_000_000),
        ("10만 미만", 0, 100_000),
    ]
    view_distribution: list[dict] = []
    for label, lo, hi in bucket_defs:
        bucket_videos = [
            v for v in filtered if lo <= int(v.get("view_count") or 0) < hi
        ]
        if not bucket_videos:
            continue
        view_distribution.append(
            {
                "bucket": label,
                "count": len(bucket_videos),
                "views": sum(int(v.get("view_count") or 0) for v in bucket_videos),
            }
        )

    # TOP10 점유율 (전체 조회수 중 상위 10편 비중)
    top10_views = sum(int(v.get("view_count") or 0) for v in top_sorted[:10])
    top10_share = (top10_views / total_views * 100) if total_views else 0.0

    # 추이 (전반기 vs 후반기 평균 조회수 비교)
    mid = len(monthly_stats) // 2
    if mid >= 2:
        first_avg = sum(m["views"] for m in monthly_stats[:mid]) / mid
        second_avg = sum(m["views"] for m in monthly_stats[mid:]) / (len(monthly_stats) - mid)
        if second_avg > first_avg * 1.15:
            trend = "상승 추세 📈"
        elif second_avg < first_avg * 0.85:
            trend = "하락 추세 📉"
        else:
            trend = "안정적 ➡️"
    else:
        trend = "데이터 부족"

    best_month = (
        max(monthly_stats, key=lambda x: x["uploads"])["month"] if monthly_stats else ""
    )
    actual_from = monthly_stats[0]["month"] if monthly_stats else (date_from or "")
    actual_to = monthly_stats[-1]["month"] if monthly_stats else (date_to or "")

    stats = {
        "total_videos": len(filtered),
        "total_views": total_views,
        "avg_views": total_views / len(filtered) if filtered else 0,
        "top_video_title": top_sorted[0]["title"] if top_sorted else "",
        "top_video_views": int(top_sorted[0].get("view_count") or 0) if top_sorted else 0,
        "avg_monthly_uploads": len(filtered) / len(monthly_stats) if monthly_stats else 0,
        "best_month": best_month,
        "trend_description": trend,
        "date_from": actual_from,
        "date_to": actual_to,
        "top10_share": top10_share,
    }

    return {
        "monthly_stats": monthly_stats,
        "top_videos": top_videos,
        "view_distribution": view_distribution,
        "stats": stats,
    }


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _fmt_duration(secs) -> str:
    secs = int(secs or 0)
    if not secs:
        return ""
    h, r = divmod(secs, 3600)
    m, _ = divmod(r, 60)
    return f"{h}시간 {m}분" if h else f"{m}분"


def _fmt_views(n) -> str:
    n = int(n or 0)
    if n >= 10000:
        return f"{n / 10000:.1f}만"
    return f"{n:,}"


# ── 렌더링 ────────────────────────────────────────────────────────────────────

_COLORS = ["#4f8ef7", "#f76f4f", "#4ff7a8", "#f7c94f"]


def _render_episode_candidates(candidates: list[dict]) -> None:
    """에피소드 검색 후보를 카드로 표시.
    1번(가장 점수 높은 영상)은 LLM 답변 본문에 이미 들어가 있으므로 여기선 제외하고,
    나머지를 접힘(expander) 안에 넣어 답변에 집중하도록 함."""
    seen: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        vid = c.get("video_id") or c.get("chunk_id", "")
        if vid in seen:
            continue
        seen.add(vid)
        unique.append(c)

    if len(unique) <= 1:
        return

    others = unique[1:5]
    with st.expander(f"다른 후보 영상 {len(others)}개 보기"):
        for c in others:
            title = c.get("title", "")
            url = c.get("video_url") or c.get("url", "")
            views = _fmt_views(c.get("view_count") or 0)
            pub = (c.get("published_at") or "")[:10]
            dur = _fmt_duration(c.get("duration_sec") or 0)
            channel = c.get("_channel_label") or c.get("channel_label", "")

            col_title, col_ch, col_views, col_date, col_dur = st.columns(
                [4, 1.2, 1.2, 1.5, 1]
            )
            col_title.markdown(f"[{title}]({url})" if url else title)
            col_ch.caption(channel)
            col_views.caption(f"👁 {views}")
            col_date.caption(pub)
            col_dur.caption(dur)


def _render_analytics(msg: dict) -> None:
    """분석 결과를 좌(차트) / 우(AI 요약) 분할 화면으로 렌더링."""
    monthly_stats: list[dict] = msg.get("monthly_stats", [])
    top_videos: list[dict] = msg.get("top_videos", [])
    view_distribution: list[dict] = msg.get("view_distribution", [])
    stats: dict = msg.get("stats", {})
    summary: str = msg.get("summary", "")
    channel_label: str = msg.get("channel_label", "")

    st.markdown(
        f"#### {channel_label} · {stats.get('date_from')} ~ {stats.get('date_to')}"
    )

    col_charts, col_summary = st.columns([1.3, 1], gap="large")

    # ── 왼쪽: 차트 ──
    with col_charts:
        if monthly_stats:
            df = pd.DataFrame(monthly_stats)

            fig_up = px.bar(
                df, x="month", y="uploads",
                title="월별 업로드 수",
                labels={"month": "", "uploads": "편"},
                color_discrete_sequence=[_COLORS[0]],
            )
            fig_up.update_layout(
                height=220, margin=dict(t=35, b=5, l=0, r=0), showlegend=False
            )
            fig_up.update_xaxes(tickangle=-45, tickfont_size=10)
            st.plotly_chart(fig_up, use_container_width=True)

            fig_vw = px.line(
                df, x="month", y="views",
                title="월별 조회수",
                labels={"month": "", "views": "조회수"},
                markers=True,
                color_discrete_sequence=[_COLORS[1]],
            )
            fig_vw.update_layout(
                height=220, margin=dict(t=35, b=5, l=0, r=0), showlegend=False
            )
            fig_vw.update_xaxes(tickangle=-45, tickfont_size=10)
            st.plotly_chart(fig_vw, use_container_width=True)

        if top_videos:
            df_top = pd.DataFrame(top_videos[:8])
            fig_top = px.bar(
                df_top, x="views", y="title",
                orientation="h",
                title="조회수 TOP 영상",
                labels={"views": "조회수", "title": ""},
                color_discrete_sequence=[_COLORS[2]],
            )
            fig_top.update_layout(height=300, margin=dict(t=35, b=5, l=0, r=10))
            fig_top.update_yaxes(tickfont_size=10, autorange="reversed")
            st.plotly_chart(fig_top, use_container_width=True)

        # 조회수 구간 분포 도넛 — 콘텐츠 흥행 분포를 한눈에
        if view_distribution:
            df_dist = pd.DataFrame(view_distribution)
            total_videos_in_dist = int(df_dist["count"].sum())
            fig_donut = px.pie(
                df_dist,
                values="count",
                names="bucket",
                title="조회수 구간별 영상 분포",
                hole=0.55,
                color="bucket",
                color_discrete_map={
                    "1000만 이상": "#f76f4f",
                    "100만 ~ 1000만": "#f7c94f",
                    "10만 ~ 100만": "#4f8ef7",
                    "10만 미만": "#7a8a9a",
                },
                category_orders={
                    "bucket": [
                        "1000만 이상",
                        "100만 ~ 1000만",
                        "10만 ~ 100만",
                        "10만 미만",
                    ]
                },
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="%{label}<br>%{value}편 (%{percent})<extra></extra>",
            )
            fig_donut.update_layout(
                height=320,
                margin=dict(t=40, b=10, l=0, r=0),
                legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
                annotations=[
                    dict(
                        text=f"<b>{total_videos_in_dist}</b><br><span style='font-size:11px'>편</span>",
                        x=0.5, y=0.5, showarrow=False, font=dict(size=18),
                    )
                ],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            top10_share = stats.get("top10_share", 0)
            if top10_share:
                st.caption(
                    f"💡 TOP 10 영상이 전체 조회수의 **{top10_share:.1f}%** 를 차지하고 있어요."
                )

    # ── 오른쪽: AI 요약 ──
    with col_summary:
        st.markdown("##### 핵심 지표")
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("총 영상", f"{stats.get('total_videos', 0)}편")
        r1c2.metric("총 조회수", f"{stats.get('total_views', 0):,}")
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("평균 조회수", f"{int(stats.get('avg_views', 0)):,}")
        r2c2.metric("월평균 업로드", f"{stats.get('avg_monthly_uploads', 0):.1f}편")

        trend = stats.get("trend_description", "")
        if trend:
            st.caption(f"조회수 추이: {trend}")

        st.divider()
        st.markdown("##### AI 분석")
        st.markdown(summary)

        if top_videos:
            st.divider()
            st.markdown("##### 최고 조회수 영상")
            top = top_videos[0]
            st.markdown(f"[{top['title']}]({top['url']})")
            st.caption(f"{top['views']:,}회")


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "analytics":
            _render_analytics(msg)
        elif msg.get("type") == "episode":
            st.markdown(msg.get("content", ""))
            if msg.get("candidates"):
                _render_episode_candidates(msg["candidates"])
        else:
            st.markdown(msg.get("content", ""))


# ── 쿼리 처리 ─────────────────────────────────────────────────────────────────

def _resolve_target_channels(intent_info: dict, sidebar_enabled: list[dict]) -> list[dict]:
    """intent에서 채널 명시 → 해당 채널, 없으면 사이드바 선택."""
    ids = intent_info.get("channel_ids")
    if ids:
        matched = [ch for ch in CHANNELS if ch["id"] in set(ids)]
        if matched:
            return matched
    return sidebar_enabled


def handle_query(query: str, enabled: list[dict]) -> dict:
    """쿼리를 처리해서 메시지 dict 반환."""
    if not enabled:
        return {
            "role": "assistant",
            "type": "text",
            "content": "왼쪽 사이드바에서 검색할 채널을 하나 이상 선택해 주세요.",
        }

    intent_info = detect_intent(query, [ch["label"] for ch in CHANNELS])
    intent = intent_info.get("intent", "episode_search")
    target = _resolve_target_channels(intent_info, enabled)

    # ── 채널 분석 ──
    if intent == "analytics":
        date_from = intent_info.get("date_from")
        date_to = intent_info.get("date_to")

        all_videos: list[dict] = []
        used_labels: list[str] = []
        for ch in target:
            vids = _load_raw(str(raw_path(ch)))
            if vids:
                all_videos.extend(vids)
                used_labels.append(ch["label"])

        if not all_videos:
            return {
                "role": "assistant",
                "type": "text",
                "content": "분석할 데이터가 없습니다. 먼저 데이터를 수집해 주세요.",
            }

        analytics = compute_analytics(all_videos, date_from, date_to)
        if analytics is None:
            period = f"{date_from} ~ {date_to}" if date_from else "전체 기간"
            return {
                "role": "assistant",
                "type": "text",
                "content": f"해당 기간({period})에 해당하는 영상이 없습니다.",
            }

        channel_label = " · ".join(used_labels)
        summary_stats = {**analytics["stats"], "channel_label": channel_label}
        summary = generate_analytics_summary(query, summary_stats)

        return {
            "role": "assistant",
            "type": "analytics",
            "channel_label": channel_label,
            "monthly_stats": analytics["monthly_stats"],
            "top_videos": analytics["top_videos"],
            "stats": analytics["stats"],
            "summary": summary,
        }

    # ── 에피소드 검색 ──
    results = run_episode_search(query, target, n=8)
    answer = generate_episode_answer(query, results)
    return {
        "role": "assistant",
        "type": "episode",
        "content": answer,
        "candidates": results,
    }


# ── 사이드바 ──────────────────────────────────────────────────────────────────

_EXAMPLE_QUERIES = [
    "이서진이랑 떡국 끓여먹던 편이 뭐였지?",
    "윤경호 나와서 썰푸는 편이 어떤 편이었지?",
    "뜬뜬 채널 25년 4월부터 26년 4월까지 분석해줘",
    "쑥쑥 채널 처음 만들어졌을때부터 지금까지 분석해줘",
]


@st.cache_data(show_spinner=False, ttl=300)
def _channel_overview(raw_file_str: str) -> dict:
    """채널별 영상 수·총 조회수·최근 업로드 일자를 캐시해서 반환."""
    p = Path(raw_file_str)
    if not p.exists():
        return {}
    try:
        videos = json.loads(p.read_text("utf-8"))
    except Exception:
        return {}
    if not videos:
        return {}
    total_views = sum(int(v.get("view_count") or 0) for v in videos)
    pub_dates = [(v.get("published_at") or "")[:10] for v in videos]
    pub_dates = [d for d in pub_dates if d]
    return {
        "videos": len(videos),
        "views": total_views,
        "latest": max(pub_dates) if pub_dates else "",
        "earliest": min(pub_dates) if pub_dates else "",
    }


def _sidebar() -> list[dict]:
    # ── 채널 선택 ──
    st.markdown("### 채널 선택")
    selected: list[dict] = []
    for ch in CHANNELS:
        has_raw = raw_path(ch).exists()
        has_chunks = chunks_path(ch).exists()
        key = f"ch_{ch['id']}"

        if key not in st.session_state:
            st.session_state[key] = has_raw and ch["id"] != "channel15ya"

        label = ch["label"]
        if has_raw and not has_chunks:
            help_txt = "키워드 검색만 가능 (벡터 인덱스 없음)"
        elif not has_raw:
            help_txt = "데이터 없음"
        else:
            help_txt = None

        st.checkbox(label, key=key, disabled=not has_raw, help=help_txt)
        if st.session_state.get(key):
            selected.append(ch)

    st.divider()

    # ── 데이터 현황 (선택된 채널만) ──
    st.markdown("### 데이터 현황")
    if not selected:
        st.caption("채널을 1개 이상 선택해 주세요.")
    else:
        for ch in selected:
            ov = _channel_overview(str(raw_path(ch)))
            if not ov:
                st.caption(f"**{ch['label']}** — 데이터 없음")
                continue
            views_short = (
                f"{ov['views'] / 10000:.0f}만회"
                if ov["views"] >= 10000
                else f"{ov['views']:,}회"
            )
            st.caption(
                f"**{ch['label']}** — {ov['videos']}편 · {views_short}\n\n"
                f"📅 {ov['earliest']} ~ {ov['latest']}"
            )

    st.divider()

    # ── 예시 질문 (클릭 시 채팅창에 자동 입력) ──
    st.markdown("### 예시 질문")
    for i, q in enumerate(_EXAMPLE_QUERIES):
        if st.button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state["queued_query"] = q
            st.rerun()

    st.divider()

    # ── 도구 ──
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("queued_query", None)
        st.rerun()

    if not llm_available():
        st.info("`.env` 에 `GEMINI_API_KEY` 를 추가하면 AI 답변이 활성화됩니다.")

    # ── 푸터 ──
    st.divider()
    st.caption(
        "ⓘ 유튜브 채널 봇 · RAG 기반 영상 검색/분석\n\n"
        "Powered by Gemini · ChromaDB · Streamlit"
    )

    return selected


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="유튜브 채널 봇",
        page_icon="▶️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _warm_embedder()

    with st.sidebar:
        enabled = _sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "type": "text",
                "content": (
                    "안녕하세요! 유튜브 채널 봇입니다.\n\n"
                    "**에피소드 검색** — 기억이 가물가물한 편을 찾아드려요\n"
                    "> 예: *이서진이랑 떡국 끓여먹던 편이 뭐였지?*\n\n"
                    "**채널 분석** — 기간별 통계와 AI 인사이트를 보여드려요\n"
                    "> 예: *뜬뜬 채널 25년 4월부터 26년 4월까지 분석해줘*\n\n"
                    "왼쪽에서 검색할 채널을 선택해 주세요."
                ),
            }
        ]

    st.markdown("## ▶️ 유튜브 채널 봇")

    # 이전 메시지 렌더링
    for msg in st.session_state.messages:
        _render_message(msg)

    # 사이드바에서 예시 질문이 클릭된 경우 자동으로 질의 처리
    queued = st.session_state.pop("queued_query", None)
    typed = st.chat_input("에피소드 검색 또는 채널 분석을 질문해보세요")
    prompt = queued or typed

    if prompt:
        st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("분석 중..."):
            response = handle_query(prompt, enabled)

        with st.chat_message("assistant"):
            if response.get("type") == "analytics":
                _render_analytics(response)
            elif response.get("type") == "episode":
                st.markdown(response.get("content", ""))
                if response.get("candidates"):
                    _render_episode_candidates(response["candidates"])
            else:
                st.markdown(response.get("content", ""))

        st.session_state.messages.append(response)


if __name__ == "__main__":
    main()
