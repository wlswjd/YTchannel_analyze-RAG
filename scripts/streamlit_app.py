"""
로컬 수집 데이터로 키워드 검색 + 업로드/조회수 간단 분석.

실행 (프로젝트 루트에서):
  streamlit run scripts/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, resolve_raw_json  # noqa: E402

_default_raw = resolve_raw_json("ddeunddeun_raw_data.json")
_default_chunks = ROOT / "data" / "processed" / "ddeunddeun_raw_data_chunks.jsonl"


@st.cache_data
def load_chunks(path: Path):
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_data
def load_raw_videos(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def search_chunks(chunks: list[dict], q: str, limit: int) -> list[dict]:
    if not q.strip():
        return []
    q_lower = q.lower()
    hits = []
    for c in chunks:
        text = (c.get("text") or "").lower()
        title = (c.get("title") or "").lower()
        if q_lower in text or q_lower in title:
            score = text.count(q_lower) * 2 + title.count(q_lower) * 3
            hits.append((score, c))
    hits.sort(key=lambda x: -x[0])
    return [h[1] for h in hits[:limit]]


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


def main():
    st.set_page_config(page_title="채널 RAG (로컬)", layout="wide")
    st.title("유튜브 채널 로컬 검색 · 간단 분석")

    col_a, col_b = st.columns(2)
    with col_a:
        raw_path = Path(
            st.text_input("Raw JSON 경로", value=str(_default_raw))
        ).expanduser()
    with col_b:
        chunks_path = Path(
            st.text_input("청크 JSONL 경로", value=str(_default_chunks))
        ).expanduser()

    videos = load_raw_videos(raw_path)
    chunks = load_chunks(chunks_path)

    tab_search, tab_stats = st.tabs(["🔎 키워드 검색 (대사)", "📈 기간·성장(조회수)"])

    with tab_search:
        st.caption(
            "청크 생성: `python scripts/chunk_data.py --input data/raw/…json`"
        )
        limit = st.slider("결과 개수", 5, 50, 15)
        with st.form("search_form"):
            q = st.text_input(
                "질문/키워드",
                placeholder="예: 라면, EP.92, 런닝맨 …",
            )
            submitted = st.form_submit_button("검색", type="primary")

        if chunks:
            if submitted:
                found = search_chunks(chunks, q, limit)
                if not q.strip():
                    st.info("키워드를 입력하세요.")
                elif not found:
                    st.warning("일치하는 청크가 없습니다.")
                else:
                    for c in found:
                        ep = c.get("episode_hint") or "—"
                        with st.expander(f"{c.get('title', '')[:80]} · EP? {ep}"):
                            st.markdown(f"**링크:** [{c.get('video_url')}]({c.get('video_url')})")
                            st.text_area("발췌", c.get("text", "")[:4000], height=200, key=c["chunk_id"])
        else:
            st.error(f"청크 파일이 없습니다: `{chunks_path}`")
            if videos:
                st.info("Raw에서 제목만 빠르게 필터합니다 (대사 검색은 청크 필요).")
                with st.form("title_only"):
                    q2 = st.text_input("제목 키워드", key="q2")
                    go = st.form_submit_button("제목 검색")
                if go and q2.strip():
                    sub = [v for v in videos if q2.lower() in (v.get("title") or "").lower()]
                    for v in sub[:limit]:
                        st.markdown(f"- [{v.get('title')}]({v.get('video_url')})")

    with tab_stats:
        if not videos:
            st.warning("Raw JSON을 찾을 수 없습니다.")
            return
        df = analytics_df(videos)
        if df.empty:
            st.warning("published_at 이 있는 영상이 없습니다.")
            return
        monthly = (
            df.groupby(df["month"].dt.to_period("M"))
            .agg(uploads=("title", "count"), views=("view_count", "sum"))
            .reset_index()
        )
        monthly["month"] = monthly["month"].astype(str)
        st.subheader("월별 업로드 수 · 조회수 합")
        st.line_chart(monthly.set_index("month")[["uploads", "views"]])
        st.dataframe(monthly, use_container_width=True)


if __name__ == "__main__":
    main()
