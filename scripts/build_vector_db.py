from __future__ import annotations

import argparse
from pathlib import Path

from channels import CHANNELS, chunks_path
from vector_store import DEFAULT_MODEL, upsert_chunks_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Build Chroma vector DB from processed chunks jsonl.")
    p.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="채널 id들 (예: ddeunddeun ssookssook). 비우면 app.py의 CHANNELS 전부.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    targets = CHANNELS
    if args.channels:
        want = set(args.channels)
        targets = [c for c in CHANNELS if c["id"] in want]

    if not targets:
        raise SystemExit("대상이 되는 채널이 없습니다. --channels 를 확인하세요.")

    for ch in targets:
        cp: Path = chunks_path(ch)
        info = upsert_chunks_jsonl(
            channel_id=ch["id"],
            chunks_jsonl=cp,
            channel_label=ch["label"],
            model_name=args.model,
            batch_size=args.batch_size,
        )
        print(info)


if __name__ == "__main__":
    main()

