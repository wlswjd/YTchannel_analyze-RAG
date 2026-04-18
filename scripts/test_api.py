import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

load_dotenv(ROOT / ".env")
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

req = youtube.channels().list(part="id", forHandle="@ddeunddeun")
res = req.execute()
print(res)
