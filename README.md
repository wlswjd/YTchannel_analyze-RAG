# YTchannel_analyze-RAG

유튜브 **Data API로 메타데이터 수집** → **자막(대사) 수집** → **청크·검색·간단 분석(Streamlit)** 까지 한 레포에서 다룹니다.

## 폴더 구조

| 경로 | 역할 |
|------|------|
| `scripts/` | 실행 스크립트 (`paths.py`가 프로젝트 루트 `ROOT` 기준) |
| `data/raw/` | `*_raw_data.json` 수집본 (git 제외 권장) |
| `data/processed/` | 청크 `.jsonl` 산출물 (git 제외) |
| `.env` | `YOUTUBE_API_KEY` (git 제외) |
| `chroma_db/` | (선택) 벡터 DB를 쓸 때 로컬 저장소 — 아직 없어도 됨 |

실행 예는 **프로젝트 루트**에서 `python scripts/…` 또는 `.venv/bin/python scripts/…` 로 통일합니다.

## 스크립트 요약

| 스크립트 | 설명 |
|----------|------|
| `scripts/data_collector.py` | 채널 업로드 목록 + 영상 메타 + (선택) `youtube_transcript_api` 자막 → `data/raw/*_raw_data.json` |
| `scripts/transcript_rescrape.py` | 실패/빈 자막만 **yt-dlp**로 재수집, 같은 JSON에 덮어쓰기 |
| `scripts/chunk_data.py` | Raw JSON → `data/processed/<이름>_chunks.jsonl` (RAG용 텍스트 청크) |
| `scripts/streamlit_app.py` | 키워드 검색(청크) + 월별 업로드/조회수 차트 |
| `scripts/test_api.py` | `@핸들` → 채널 id 확인용 스니펫 |

## 파이프라인 방향

1. **수집·정제** → `data/raw/*.json`
2. **청크** → `python scripts/chunk_data.py` → `data/processed/*_chunks.jsonl`
3. **지금 앱**: Streamlit은 청크에 대해 **키워드(부분 문자열) 검색** — 임베딩·벡터DB **없이** 동작합니다.
4. **다음 단계(선택)**: 청크 → **임베딩** → **Chroma / FAISS 등 벡터 스토어** → 필요하면 LLM으로 근거 인용. 의미 검색·RAG 품질을 올릴 때 이 단계를 붙이면 됩니다.

## 권장 작업 순서 (수집 끝난 뒤)

1. Raw JSON에 자막이 채워졌는지 확인  
2. **청크 생성**

```bash
.venv/bin/python scripts/chunk_data.py --input data/raw/ddeunddeun_raw_data.json
```

출력: `data/processed/ddeunddeun_raw_data_chunks.jsonl`  
(`--input` 생략 시 `data/raw/ddeunddeun_raw_data.json` 우선, 없으면 레거시로 루트 동일 파일명을 찾습니다.)

3. **앱에서 검색·분석**

```bash
.venv/bin/streamlit run scripts/streamlit_app.py
```

- **검색 탭**: 청크에 포함된 대사에서 키워드 매칭 → 영상 링크·제목·발췌  
- **분석 탭**: `published_at` 기준 월별 업로드 수·조회수 합 (간단 성장 그래프)

## 다른 채널 (예: ssookssook)

1. `scripts/data_collector.py`에서 `TARGET_CHANNEL_ID`, `output_filename` (예: `ssookssook_raw_data.json`) 수정 후 실행 → `data/raw/`에 저장  
2. API 여유가 있으면 `python scripts/transcript_rescrape.py --input data/raw/ssookssook_raw_data.json`  
3. `python scripts/chunk_data.py --input data/raw/ssookssook_raw_data.json`  
4. Streamlit에서 Raw JSON / 청크 jsonl 경로를 해당 파일로 지정  

## 자막 재수집(이어받기)

`scripts/transcript_rescrape.py`는 실패·플레이스홀더만 다시 시도합니다.

```bash
python scripts/transcript_rescrape.py --input data/raw/ddeunddeun_raw_data.json
```

배치로 나누려면 `--batch 1 --batch-size 41` 등 (남은 대상만 자동 집계).

### 저장 주기

기본 `--save-every 1` (영상마다 저장).

---

구버전 안내: 예전에는 `--batch-size 41`을 1~5번 각각 돌리는 예시를 적었는데, **같은 명령을 반복**해도 됩니다(매번 “남은 재수집 대상”만 처리).
