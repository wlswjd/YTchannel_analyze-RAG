"""
이미 수집된 raw JSON에 description/tags/channel_title/comment_count 같은
풍부한 메타데이터를 보강해 넣는 스크립트.

YouTube 모바일 앱에서 썸네일 아래에 보이는 짧은 요약은
사실 영상의 'description' 일부야. 검색 정확도를 끌어올리려면 이게 꼭 필요해.

사용:
    .venv/bin/python scripts/enrich_metadata.py --input data/raw/ddeunddeun_raw_data.json
    .venv/bin/python scripts/enrich_metadata.py --all   # 등록된 모든 채널
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channels import CHANNELS, raw_path  # noqa: E402
from paths import ROOT  # noqa: E402

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise SystemExit("`.env`에 YOUTUBE_API_KEY 가 없습니다.")

youtube = build("youtube", "v3", developerKey=API_KEY)


def fetch_snippets(video_ids: list[str]) -> dict[str, dict]:
    """video_id -> snippet+statistics dict 반환 (50개씩 배치)."""
    out: dict[str, dict] = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch),
        )
        resp = request.execute()
        for item in resp.get("items", []):
            vid = item["id"]
            snip = item.get("snippet", {})
            stats = item.get("statistics", {})
            out[vid] = {
                "description": snip.get("description", ""),
                "tags": snip.get("tags", []),
                "channel_title": snip.get("channelTitle", ""),
                "comment_count": int(stats.get("commentCount", 0)),
            }
        time.sleep(0.2)
    return out


def enrich_one(input_path: Path, force: bool = False) -> tuple[int, int]:
    """파일 하나 보강. (보강된 영상 수, 전체 수) 반환."""
    if not input_path.exists():
        print(f"  파일 없음: {input_path}")
        return 0, 0

    data = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"  총 {len(data)}개 영상 로드: {input_path.name}")

    targets = [
        v for v in data
        if force or not v.get("description") and not v.get("tags")
    ]
    if not targets:
        print(f"  이미 모든 영상에 description/tags가 있음 (--force 로 강제 갱신 가능)")
        return 0, len(data)

    print(f"  보강 대상: {len(targets)}개")

    video_ids = [v["video_id"] for v in targets]
    enriched = fetch_snippets(video_ids)

    updated = 0
    for v in data:
        meta = enriched.get(v["video_id"])
        if not meta:
            continue
        v["description"] = meta["description"]
        v["tags"] = meta["tags"]
        v["channel_title"] = meta["channel_title"]
        v["comment_count"] = meta["comment_count"]
        updated += 1

    input_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  완료: {updated}개 영상 메타 보강 → {input_path.name}")
    return updated, len(data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=None, help="raw JSON 경로")
    p.add_argument("--all", action="store_true", help="등록된 모든 채널 처리")
    p.add_argument("--force", action="store_true", help="이미 description 있어도 다시 가져오기")
    args = p.parse_args()

    if args.all:
        for ch in CHANNELS:
            path = raw_path(ch)
            print(f"[{ch['label']}] {path}")
            enrich_one(path, force=args.force)
            print()
        return

    if not args.input:
        raise SystemExit("--input 또는 --all 중 하나는 필요합니다.")

    enrich_one(args.input.resolve(), force=args.force)


if __name__ == "__main__":
    main()
