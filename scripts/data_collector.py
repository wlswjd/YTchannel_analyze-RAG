import json
import os
import sys
import time
from pathlib import Path

import isodate
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, data_raw_dir  # noqa: E402

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)


def get_uploads_playlist_id(channel_id):
    request = youtube.channels().list(part="contentDetails", id=channel_id)
    response = request.execute()
    if not response.get("items"):
        raise ValueError("채널을 찾을 수 없습니다.")
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_video_ids_from_playlist(playlist_id):
    video_ids = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        )
        response = request.execute()

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids


def get_video_details(video_ids):
    video_data = []
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i : i + 50]
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch_ids),
        )
        response = request.execute()

        for item in response.get("items", []):
            duration_iso = item["contentDetails"]["duration"]
            duration_sec = isodate.parse_duration(duration_iso).total_seconds()

            if duration_sec <= 60:
                continue

            video_id = item["id"]
            snippet = item["snippet"]
            statistics = item.get("statistics", {})

            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = thumbnails.get("maxres", thumbnails.get("high", thumbnails.get("default", {}))).get("url", "")

            video_data.append(
                {
                    "video_id": video_id,
                    "title": snippet.get("title", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "view_count": int(statistics.get("viewCount", 0)),
                    "like_count": int(statistics.get("likeCount", 0)),
                    "duration_sec": duration_sec,
                    "thumbnail_url": thumbnail_url,
                    "video_url": f"https://www.youtube.com/watch?v={video_id}",
                    "transcript": "",
                }
            )

    return video_data


def append_transcripts(video_data, save_path=None):
    yt_api = YouTubeTranscriptApi()
    for i, video in enumerate(video_data):
        try:
            transcript_list = yt_api.fetch(video["video_id"], languages=["ko"])
            full_text = " ".join([t.text for t in transcript_list])
            video["transcript"] = full_text
            print(f"[{i+1}/{len(video_data)}] 자막 추출 성공: {video['title']}")
        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"[{i+1}/{len(video_data)}] 자막 없음 (무시됨): {video['title']}")
            video["transcript"] = "자막이 제공되지 않는 영상입니다."
        except Exception as e:
            print(f"[{i+1}/{len(video_data)}] 자막 추출 에러 ({video['video_id']}) - IP 차단 등")
            video["transcript"] = "자막 추출 중 에러가 발생했습니다."
        
        # 중간 저장 (끊겨도 데이터가 날아가지 않도록)
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(video_data, f, ensure_ascii=False, indent=2)
                
        time.sleep(5)

    return video_data


if __name__ == "__main__":
    # TARGET_CHANNEL_ID = "UCDNvRZRgvkBTUkQzFoT_8rA"  # 뜬뜬
    TARGET_CHANNEL_ID = "UC3egV8AyVrnwUEOkGZB-s2Q"  # 쑥쑥 SsookSsook
    output_filename = "ssookssook_raw_data.json"

    print("1. 업로드 재생목록 ID 조회 중...")
    uploads_playlist_id = get_uploads_playlist_id(TARGET_CHANNEL_ID)

    print("2. 재생목록 내 영상 ID 수집 중...")
    all_video_ids = get_video_ids_from_playlist(uploads_playlist_id)
    print(f"총 {len(all_video_ids)}개의 영상 ID를 찾았습니다.")

    print("3. 메타데이터 수집 및 쇼츠 필터링 중...")
    filtered_videos = get_video_details(all_video_ids)
    print(f"쇼츠를 제외한 총 {len(filtered_videos)}개의 유효 영상을 확보했습니다.")

    out_dir = data_raw_dir()
    output_path = out_dir / output_filename

    # 1차 저장 (메타데이터만 먼저 저장해서, 자막 수집 중단돼도 날아가지 않게 함)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered_videos, f, ensure_ascii=False, indent=2)
    print(f"-> 메타데이터 1차 저장 완료: {output_path}")

    print("4. 자막 데이터 추출 및 병합 중 (시간이 소요됩니다)...")
    try:
        final_data = append_transcripts(filtered_videos, output_path)
    except KeyboardInterrupt:
        print("\n[!] 사용자가 강제 종료(Ctrl+C)했습니다. 지금까지 수집된 상태로 저장을 마칩니다.")
        sys.exit(0)

    print(f"\n데이터 수집 완료. '{output_path}'에 최종 저장되었습니다.")
