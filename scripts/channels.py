from __future__ import annotations

from pathlib import Path

from paths import data_processed_dir, resolve_raw_json


# 새 채널 추가 시 여기만 확장하면 됨 (raw 파일명 = chunk_data.py 출력 규칙과 동일)
CHANNELS: list[dict] = [
    {
        "id": "ddeunddeun",
        "label": "뜬뜬",
        "raw_file": "ddeunddeun_raw_data.json",
    },
    {
        "id": "ssookssook",
        "label": "쑥쑥",
        "raw_file": "ssookssook_raw_data.json",
    },
]


def raw_path(ch: dict) -> Path:
    return resolve_raw_json(ch["raw_file"])


def chunks_path(ch: dict) -> Path:
    stem = Path(ch["raw_file"]).stem
    return data_processed_dir() / f"{stem}_chunks.jsonl"

