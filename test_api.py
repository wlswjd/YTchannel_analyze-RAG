import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

req = youtube.channels().list(part="id", forHandle="@ddeunddeun")
res = req.execute()
print(res)
