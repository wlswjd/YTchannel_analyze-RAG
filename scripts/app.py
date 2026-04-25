"""
채널별 청크 키워드 검색 챗봇 UI.

실행 (프로젝트 루트):
  streamlit run scripts/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, data_processed_dir, resolve_raw_json  # noqa: E402
from vector_store import DEFAULT_MODEL, semantic_search  # noqa: E402
from channels import CHANNELS, chunks_path, raw_path  # noqa: E402
from llm import fallback_answer, generate_llm_answer, llm_available  # noqa: E402

@st.cache_data(show_spinner=False)
def _load_chunks_file(path_str: str) -> list[dict] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def _load_raw_file(path_str: str) -> list[dict]:
    path = Path(path_str)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def search_chunks(
    chunks: list[dict], q: str, limit: int, channel_label: str
) -> list[dict]:
    if not q.strip():
        return []
    q_lower = q.lower()
    hits: list[tuple[float, dict]] = []
    for c in chunks:
        text = (c.get("text") or "").lower()
        title = (c.get("title") or "").lower()
        if q_lower in text or q_lower in title:
            score = text.count(q_lower) * 2 + title.count(q_lower) * 3
            doc = {**c, "_channel_label": channel_label}
            hits.append((score, doc))
    hits.sort(key=lambda x: -x[0])
    return [h[1] for h in hits[:limit]]


def combine_hits(per_channel: list[list[dict]], query: str, total_limit: int) -> list[dict]:
    """채널별 히트를 점수 순으로 합치고 chunk_id 기준 중복 제거."""
    q = query.lower()
    if not q.strip():
        return []
    flat: list[tuple[float, dict]] = []
    for group in per_channel:
        for doc in group:
            text = (doc.get("text") or "").lower()
            title = (doc.get("title") or "").lower()
            score = text.count(q) * 2 + title.count(q) * 3
            flat.append((score, doc))
    flat.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[dict] = []
    for _, doc in flat:
        cid = str(doc.get("chunk_id") or doc.get("video_id") or "")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(doc)
        if len(out) >= total_limit:
            break
    return out


def analytics_df(videos: list[dict]) -> pd.DataFrame:
    rows = []
    for v in videos:
        pa = v.get("published_at") or ""
        if not pa:
            continue
        try:
            dt = pd.to_datetime(pa, utc=True)
        except Exception:
            continue
        rows.append(
            {
                "month": dt.to_period("M").to_timestamp(),
                "published_at": dt,
                "view_count": int(v.get("view_count") or 0),
                "title": v.get("title", ""),
                "video_url": v.get("video_url", ""),
            }
        )
    return pd.DataFrame(rows)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        /* 다크 톤 보조 (Streamlit 다크 테마와 함께 사용) */
        .block-container { padding-top: 1rem; }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_about() -> None:
    st.markdown("### 유튜브 대사 검색")
    st.caption("수집한 자막 청크에서 키워드를 찾아 영상 링크와 발췌를 보여 줍니다.")
    st.markdown(
        "채널을 켜고 질문하면, **켜진 채널 범위**에서만 검색합니다. "
        "청크는 `python scripts/chunk_data.py --input data/raw/…json` 으로 만듭니다."
    )


def sidebar_channels() -> list[dict]:
    st.subheader("검색 범위 (채널)")
    selected: list[dict] = []
    for ch in CHANNELS:
        cp = chunks_path(ch)
        has_chunks = cp.exists()
        key = f"use_channel_{ch['id']}"
        if key not in st.session_state:
            st.session_state[key] = has_chunks and ch["id"] == "ddeunddeun"

        help_txt = f"청크: `data/processed/{cp.name}`"
        if not has_chunks:
            help_txt += " (아직 없음 — 청크 생성 후 검색 가능)"

        st.toggle(ch["label"], key=key, help=help_txt)

        if st.session_state[key]:
            selected.append(ch)

    return selected


def sidebar_settings() -> int:
    st.subheader("설정")
    st.radio(
        "답변 스타일",
        options=["리스트(근거 보기)", "LLM처럼 요약 답변"],
        index=0,
        key="answer_style",
        help="LLM처럼 요약 답변은 (선택) Gemini API 키가 있을 때 더 자연스럽게 답해줍니다.",
    )
    st.radio(
        "검색 모드",
        options=["키워드(빠름)", "의미(질문형)"],
        index=0,
        key="search_mode",
        help="키워드는 현재처럼 문자열 포함 검색. 의미 검색은 임베딩 기반이라 질문형 문장에 더 강하지만 인덱스가 필요합니다.",
    )
    n = st.slider("검색 결과 개수", min_value=3, max_value=50, value=12, step=1)
    if st.session_state.get("search_mode") == "의미(질문형)":
        st.caption(
            "의미 검색은 `python scripts/build_vector_db.py` 로 인덱스를 만든 뒤 사용하세요. "
            f"(기본 모델: {DEFAULT_MODEL})"
        )
    if st.session_state.get("answer_style") == "LLM처럼 요약 답변" and not llm_available():
        st.warning("LLM 요약 답변을 쓰려면 `.env`에 `GEMINI_API_KEY=...` 를 추가하세요. (없으면 간단 템플릿으로 답합니다)")
    with st.expander("고급 · 경로", expanded=False):
        st.caption("비우면 자동 경로 (`data/raw`, `data/processed`)")
        for ch in CHANNELS:
            st.text_input(
                f"{ch['label']} 청크 .jsonl (선택)",
                value="",
                key=f"override_chunks_{ch['id']}",
                placeholder=str(chunks_path(ch)),
            )
    return n


def resolve_chunk_path(ch: dict) -> Path:
    o = st.session_state.get(f"override_chunks_{ch['id']}", "").strip()
    if o:
        return Path(o).expanduser().resolve()
    return chunks_path(ch)


def sidebar_stats(enabled: list[dict]) -> None:
    with st.expander("월별 업로드 · 조회수", expanded=False):
        all_v: list[dict] = []
        for ch in enabled:
            rp = raw_path(ch)
            all_v.extend(_load_raw_file(str(rp)))
        if not all_v:
            st.info("켜진 채널에 Raw JSON이 없거나 비어 있습니다.")
            return
        df = analytics_df(all_v)
        if df.empty:
            st.warning("published_at 이 있는 영상이 없습니다.")
            return
        monthly = (
            df.groupby(df["month"].dt.to_period("M"))
            .agg(uploads=("title", "count"), views=("view_count", "sum"))
            .reset_index()
        )
        monthly["month"] = monthly["month"].astype(str)
        st.line_chart(monthly.set_index("month")[["uploads", "views"]])
        st.dataframe(monthly, use_container_width=True, hide_index=True)


def format_reply(
    query: str, hits: list[dict], enabled_labels: list[str], title_only: bool
) -> str:
    if not enabled_labels:
        return "사이드바에서 **검색할 채널**을 하나 이상 켜 주세요. (청크 파일이 있는 채널만 켤 수 있습니다.)"

    if title_only:
        lines = [
            f"청크가 없어 **제목**만 검색했습니다. (범위: {', '.join(enabled_labels)})",
            "",
        ]
        for i, v in enumerate(hits[:20], 1):
            t = v.get("title", "")
            u = v.get("video_url", "")
            lines.append(f"{i}. [{t}]({u})")
        return "\n".join(lines)

    if not hits:
        return (
            f"「{query}」와 맞는 대사/제목이 없습니다. "
            f"(검색 채널: {', '.join(enabled_labels)})"
        )

    lines = [
        f"**「{query}」** 검색 · 채널: {', '.join(enabled_labels)}",
        "",
    ]
    for i, c in enumerate(hits, 1):
        title = (c.get("title") or "")[:120]
        url = c.get("video_url", "")
        ep = c.get("episode_hint") or "—"
        ch_lab = c.get("_channel_label", "")
        excerpt = (c.get("text") or "")[:900]
        lines.append(f"**{i}.** [{title}]({url}) · *{ch_lab}* · EP? {ep}")
        lines.append(f"> {excerpt}")
        lines.append("")
    return "\n".join(lines)


def _title_hits_for_channel(ch: dict, query: str) -> list[dict]:
    raw_p = raw_path(ch)
    videos = _load_raw_file(str(raw_p))
    q = query.lower()
    out: list[dict] = []
    for v in videos:
        if q in (v.get("title") or "").lower():
            out.append({**v, "_channel_label": ch["label"]})
    return out


def run_search(
    query: str, enabled: list[dict], limit: int
) -> tuple[str, bool]:
    """Returns (markdown reply, used_title_only)."""
    labels = [ch["label"] for ch in enabled]

    if not enabled:
        return format_reply(query, [], [], False), False

    # semantic search mode (embedding + chroma)
    if st.session_state.get("search_mode") == "의미(질문형)":
        channel_ids = [ch["id"] for ch in enabled]
        hits = semantic_search(channel_ids=channel_ids, query=query, n_results=limit)

        # match format_reply schema
        normalized = []
        for h in hits:
            normalized.append(
                {
                    "chunk_id": h.get("chunk_id", ""),
                    "video_id": h.get("video_id", ""),
                    "title": h.get("title", ""),
                    "episode_hint": h.get("episode_hint", ""),
                    "published_at": h.get("published_at", ""),
                    "video_url": h.get("video_url", ""),
                    "chunk_index": h.get("chunk_index", 0),
                    "text": h.get("text", ""),
                    "_channel_label": h.get("channel_label", ""),
                }
            )
        if st.session_state.get("answer_style") == "LLM처럼 요약 답변":
            llm_text = generate_llm_answer(query, normalized)
            if llm_text:
                return llm_text, False
            return fallback_answer(query, normalized), False

        if not normalized:
            return (
                "의미 검색 결과가 없습니다. 아직 인덱스가 없을 수 있어요.\n\n"
                "아래를 먼저 실행해 보세요.\n\n"
                "```bash\n"
                "pip install -r requirements.txt\n"
                "python scripts/build_vector_db.py\n"
                "```\n",
                False,
            )
        reply = format_reply(query, normalized, labels, False)
        return reply, False

    per_hits: list[list[dict]] = []
    title_only_rows: list[dict] = []
    no_chunk_channels: list[str] = []

    for ch in enabled:
        cp = resolve_chunk_path(ch)
        chunks = _load_chunks_file(str(cp))
        if chunks:
            per_hits.append(search_chunks(chunks, query, limit * 2, ch["label"]))
        else:
            no_chunk_channels.append(ch["label"])
            title_only_rows.extend(_title_hits_for_channel(ch, query))

    if per_hits:
        merged = combine_hits(per_hits, query, limit)
        if st.session_state.get("answer_style") == "LLM처럼 요약 답변":
            llm_text = generate_llm_answer(query, merged)
            if llm_text:
                return llm_text, False
            return fallback_answer(query, merged), False

        reply = format_reply(query, merged, labels, False)
        if no_chunk_channels and title_only_rows:
            lines = ["\n\n---\n### 청크 없음 채널 — 제목만 검색\n"]
            for i, v in enumerate(title_only_rows[:8], 1):
                t = v.get("title", "")
                u = v.get("video_url", "")
                lab = v.get("_channel_label", "")
                lines.append(f"{i}. [{t}]({u}) · *{lab}*")
            reply += "\n".join(lines)
        elif no_chunk_channels:
            reply += (
                "\n\n---\n**참고:** "
                + ", ".join(f"*{c}*" for c in no_chunk_channels)
                + " 채널은 아직 청크 파일이 없어 대사 검색에서 빠졌습니다. "
                "`python scripts/chunk_data.py` 로 jsonl을 만든 뒤 다시 시도해 보세요."
            )
        return reply, False

    if title_only_rows:
        title_only_rows = title_only_rows[:limit]
        return format_reply(query, title_only_rows, labels, True), True

    return format_reply(query, [], labels, False), False


def main() -> None:
    st.set_page_config(
        page_title="채널 대사 검색",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_style()

    with st.sidebar:
        sidebar_about()
        st.divider()
        enabled = sidebar_channels()
        st.divider()
        limit = sidebar_settings()
        st.divider()
        sidebar_stats(enabled)

    st.markdown("## 대사 검색 챗봇")
    st.caption("키워드·짧은 문장을 입력하면 수집 자막 청크에서 찾아 줍니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요. 왼쪽에서 **검색할 채널**을 켠 뒤, 찾고 싶은 단어나 장면을 입력해 보세요.",
            }
        ]

    if prompt := st.chat_input("대사·키워드로 질문…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("청크 검색 중…"):
            reply, _ = run_search(prompt, enabled, limit)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


if __name__ == "__main__":
    main()
