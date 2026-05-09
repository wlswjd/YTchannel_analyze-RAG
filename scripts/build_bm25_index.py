"""
채널별 BM25 인덱스를 빌드하고 data/bm25/bm25_{channel_id}.pkl 로 저장합니다.

사용법:
  python scripts/build_bm25_index.py                        # 전체 채널
  python scripts/build_bm25_index.py --channels ddeunddeun  # 특정 채널만
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channels import CHANNELS, chunks_path  # noqa: E402
from hybrid_search import build_bm25_index  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build BM25 index from processed chunks jsonl.")
    p.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="채널 id들 (예: ddeunddeun ssookssook). 비우면 CHANNELS 전부.",
    )
    args = p.parse_args()

    targets = CHANNELS
    if args.channels:
        want = set(args.channels)
        targets = [c for c in CHANNELS if c["id"] in want]

    if not targets:
        raise SystemExit("대상 채널이 없습니다. --channels 를 확인하세요.")

    for ch in targets:
        cp = chunks_path(ch)
        if not cp.exists():
            print(f"[SKIP] {ch['label']} — chunks jsonl 없음: {cp}")
            continue
        out = build_bm25_index(channel_id=ch["id"], chunks_jsonl=cp)
        print(f"[DONE] {ch['label']} → {out}")


if __name__ == "__main__":
    main()
