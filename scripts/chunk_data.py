"""
Raw JSON(영상별 transcript)을 RAG용 텍스트 청크로 나눠 .jsonl로 저장합니다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import data_processed_dir, resolve_raw_json  # noqa: E402


def _ep_from_title(title: str) -> str | None:
    m = re.search(r"EP\.?\s*(\d+)", title, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)\s*회", title)
    if m:
        return m.group(1)
    return None


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
        default=None,
        help="수집 완료된 raw JSON (기본: data/raw/ddeunddeun_raw_data.json 또는 루트 레거시)",
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

    inp = args.input
    if inp is None:
        inp = resolve_raw_json("ddeunddeun_raw_data.json")
    inp = inp.resolve()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = data_processed_dir()
    else:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(inp.read_text(encoding="utf-8"))
    out_path = out_dir / f"{inp.stem}_chunks.jsonl"

    lines_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in data:
            vid = row.get("video_id", "")
            title = row.get("title", "")
            t = row.get("transcript") or ""
            if len(t) < 30:
                continue
            parts = chunk_text(t, args.max_chars, args.overlap)
            ep = _ep_from_title(title)
            for idx, part in enumerate(parts):
                doc = {
                    "chunk_id": f"{vid}_{idx}",
                    "video_id": vid,
                    "title": title,
                    "episode_hint": ep,
                    "published_at": row.get("published_at", ""),
                    "video_url": row.get("video_url", ""),
                    "view_count": row.get("view_count", 0),
                    "like_count": row.get("like_count", 0),
                    "duration_sec": row.get("duration_sec", 0),
                    "chunk_index": idx,
                    "text": part,
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                lines_written += 1

    print(f"청크 {lines_written}개 → {out_path}")


if __name__ == "__main__":
    main()
