"""프로젝트 루트 기준 경로 (scripts/ 기준 상위 폴더)."""
from pathlib import Path

# 프로젝트 루트 (…/YTchannel_analyze RAG)
ROOT = Path(__file__).resolve().parent.parent


def resolve_raw_json(filename: str) -> Path:
    """data/raw/<name> 우선, 없으면 레거시(루트) 동일 파일명."""
    preferred = ROOT / "data" / "raw" / filename
    if preferred.exists():
        return preferred
    legacy = ROOT / filename
    if legacy.exists():
        return legacy
    return preferred


def data_raw_dir() -> Path:
    d = ROOT / "data" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_processed_dir() -> Path:
    d = ROOT / "data" / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d
