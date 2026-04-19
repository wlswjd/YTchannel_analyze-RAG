"""이전 진입점 호환: `streamlit run scripts/app.py` 권장."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import main  # noqa: E402

if __name__ == "__main__":
    main()
