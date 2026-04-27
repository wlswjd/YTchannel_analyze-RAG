"""
Raw JSON(영상별 transcript + description + tags)을 RAG용 텍스트 청크로 나눠 .jsonl로 저장합니다.

청크 종류:
- 'meta'  : 영상마다 1개. 제목 + description + tags + 출연자/회차 힌트.
            검색에서 가중치를 더 받음 (대사보다 신뢰도 높은 정보).
- 'transcript' : 자막 본문을 max_chars 단위로 슬라이딩.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import data_processed_dir  # noqa: E402


def _ep_from_title(title: str) -> str | None:
    m = re.search(r"EP\.?\s*(\d+)", title, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)\s*회", title)
    if m:
        return m.group(1)
    return None


def _clean_description(desc: str) -> str:
    """description에서 검색 노이즈(URL, 반복 문구 등)를 어느 정도 정리."""
    if not desc:
        return ""
    desc = re.sub(r"https?://\S+", " ", desc)
    desc = re.sub(r"#\S+", lambda m: m.group(0).replace("#", ""), desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    return desc


def _build_meta_text(row: dict) -> str:
    """검색 임베딩에 들어갈 영상 요약 텍스트.
    제목·요약(description)·태그·회차 정보를 한 덩어리로."""
    title = (row.get("title") or "").strip()
    desc = _clean_description(row.get("description") or "")
    tags = row.get("tags") or []
    tags_text = ", ".join(tags) if tags else ""
    ep = _ep_from_title(title)
    ep_str = f"회차: EP.{ep}" if ep else ""

    parts = [
        f"제목: {title}" if title else "",
        f"요약: {desc}" if desc else "",
        f"태그: {tags_text}" if tags_text else "",
        ep_str,
    ]
    return "\n".join(p for p in parts if p)


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = end - overlap
        if i < 0:
            i = end
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="수집 완료된 raw JSON (예: data/raw/ssookssook_raw_data.json)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="청크 jsonl 저장 폴더 (기본: data/processed)",
    )
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=200)
    args = p.parse_args()

    inp = args.input.resolve()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = data_processed_dir()
    else:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(inp.read_text(encoding="utf-8"))
    out_path = out_dir / f"{inp.stem}_chunks.jsonl"

    meta_written = 0
    transcript_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in data:
            vid = row.get("video_id", "")
            title = row.get("title", "")
            ep = _ep_from_title(title)
            base_meta = {
                "video_id": vid,
                "title": title,
                "episode_hint": ep,
                "published_at": row.get("published_at", ""),
                "video_url": row.get("video_url", ""),
                "view_count": row.get("view_count", 0),
                "like_count": row.get("like_count", 0),
                "duration_sec": row.get("duration_sec", 0),
                "description": (row.get("description") or "")[:2000],
            }

            meta_text = _build_meta_text(row)
            if meta_text:
                doc = {
                    **base_meta,
                    "chunk_id": f"{vid}_meta",
                    "chunk_index": -1,
                    "chunk_kind": "meta",
                    "text": meta_text,
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                meta_written += 1

            t = row.get("transcript") or ""
            if len(t) >= 30 and "자막이 제공되지 않" not in t and "자막 추출 중 에러" not in t:
                parts = chunk_text(t, args.max_chars, args.overlap)
                for idx, part in enumerate(parts):
                    doc = {
                        **base_meta,
                        "chunk_id": f"{vid}_{idx}",
                        "chunk_index": idx,
                        "chunk_kind": "transcript",
                        "text": part,
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    transcript_written += 1

    print(
        f"메타 청크 {meta_written}개 + 자막 청크 {transcript_written}개 "
        f"→ {out_path}"
    )


if __name__ == "__main__":
    main()
