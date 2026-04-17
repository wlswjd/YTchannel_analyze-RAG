import os
import time
import json
import isodate
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# 환경 변수 및 API 클라이언트 초기화
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_uploads_playlist_id(channel_id):
    """채널의 업로드 재생목록 ID를 가져옵니다."""
    request = youtube.channels().list(part="contentDetails", id=channel_id)
    response = request.execute()
    if not response.get("items"):
        raise ValueError("채널을 찾을 수 없습니다.")
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

def get_video_ids_from_playlist(playlist_id):
    """재생목록의 모든 영상 ID를 추출합니다."""
    video_ids = []
    next_page_token = None
    
    while True:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
            
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
            
    return video_ids

def get_video_details(video_ids):
    """영상 ID 리스트를 바탕으로 상세 메타데이터를 수집하고 쇼츠를 제외합니다."""
    video_data = []
    # API는 한 번에 최대 50개의 ID만 조회 가능하므로 분할 처리
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch_ids)
        )
        response = request.execute()
        
        for item in response.get("items", []):
            duration_iso = item["contentDetails"]["duration"]
            duration_sec = isodate.parse_duration(duration_iso).total_seconds()
            
            # 쇼츠 제외 (60초 이하 영상 필터링)
            if duration_sec <= 60:
                continue
                
            video_id = item["id"]
            snippet = item["snippet"]
            statistics = item.get("statistics", {})
            
            # 고해상도 썸네일 추출 (없을 경우 기본값)
            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = thumbnails.get("maxres", thumbnails.get("high", thumbnails.get("default", {}))).get("url", "")
            
            video_data.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "published_at": snippet.get("publishedAt", ""),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
                "duration_sec": duration_sec,
                "thumbnail_url": thumbnail_url,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "transcript": "" # 자막은 다음 단계에서 병합
            })
            
    return video_data

def append_transcripts(video_data):
    """각 영상의 한글 자막을 추출하여 데이터에 병합합니다."""
    yt_api = YouTubeTranscriptApi() # Initialize the API client once
    for i, video in enumerate(video_data):
        try:
            transcript_list = yt_api.fetch(video["video_id"], languages=['ko'])
            # 타임스탬프를 제외하고 텍스트만 결합하여 하나의 문자열로 생성
            full_text = " ".join([t.text for t in transcript_list])
            video["transcript"] = full_text
            print(f"[{i+1}/{len(video_data)}] 자막 추출 성공: {video['title']}")
        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"[{i+1}/{len(video_data)}] 자막 없음 (무시됨): {video['title']}")
            video["transcript"] = "자막이 제공되지 않는 영상입니다."
        except Exception as e:
            import traceback
            print(f"[{i+1}/{len(video_data)}] 자막 추출 에러 ({video['video_id']}):")
            traceback.print_exc() # Print full traceback
            video["transcript"] = "자막 추출 중 에러가 발생했습니다."
        time.sleep(5) # Add a delay to avoid IP blocking
            
    return video_data

if __name__ == "__main__":
    # 타겟 채널 ID (예: 뜬뜬 채널 ID. 실제 채널 ID로 변경해야 함)
    # 채널 URL이 핸들(@ddeunddeun) 형태인 경우 소스코드 등에서 고유 UC_ ID를 추출해야 합니다.
    TARGET_CHANNEL_ID = "UCDNvRZRgvkBTUkQzFoT_8rA" # 뜬뜬 DdeunDdeun ID
    
    print("1. 업로드 재생목록 ID 조회 중...")
    uploads_playlist_id = get_uploads_playlist_id(TARGET_CHANNEL_ID)
    
    print("2. 재생목록 내 영상 ID 수집 중...")
    all_video_ids = get_video_ids_from_playlist(uploads_playlist_id)
    print(f"총 {len(all_video_ids)}개의 영상 ID를 찾았습니다.")
    
    print("3. 메타데이터 수집 및 쇼츠 필터링 중...")
    filtered_videos = get_video_details(all_video_ids)
    print(f"쇼츠를 제외한 총 {len(filtered_videos)}개의 유효 영상을 확보했습니다.")
    
    print("4. 자막 데이터 추출 및 병합 중 (시간이 소요됩니다)...")
    final_data = append_transcripts(filtered_videos)
    
    output_filename = "ddeunddeun_raw_data.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n데이터 수집 완료. '{output_filename}'에 저장되었습니다.")