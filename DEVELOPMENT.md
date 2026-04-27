# YTchannel_analyze RAG — 개발 진행 기록 & 로드맵

## 1. 프로젝트 한 줄 요약

특정 유튜브 채널(뜬뜬·쑥쑥·15야)의 영상을 수집하고, RAG(검색 + LLM)로
**"그 ○○○이랑 △△ 먹던 편 뭐였지?"** 같은 자연어 질문에 답해주는 Streamlit 봇.

## 2. 현재 데이터 파이프라인

```
data_collector.py        ── 메타데이터(제목·설명·태그·통계) + 자막 1차 시도
        │
        ▼
data/raw/<channel>_raw_data.json
        │
        ├── enrich_metadata.py   ── 이미 모은 raw에 description/tags 보강(YouTube API)
        ├── transcript_rescrape.py ── 빈 자막을 yt-dlp로 재수집
        │
        ▼
chunk_data.py            ── 영상 1개당
                            ① 메타 청크(제목+설명+태그)  ★검색 신뢰도 가장 높음
                            ② 자막 청크(1200자 슬라이딩, 200자 overlap)
        │
        ▼
data/processed/<channel>_chunks.jsonl
        │
        ▼
build_vector_db.py       ── ChromaDB(local persistent)
                            모델: jhgan/ko-sroberta-multitask
        │
        ▼
chroma_db/
        │
        ▼
app.py (Streamlit)       ── 의미검색 → 영상 단위 점수 집계 → Gemini로 답변 생성
```

## 3. 그동안의 개발 흐름 (시간순)

| 단계 | 한 일 | 핵심 파일 |
|------|------|----------|
| 1 | 채널 업로드 목록 + 메타 + 자막 수집 | `data_collector.py` |
| 2 | YouTube 자막 차단·실패 케이스를 yt-dlp로 재수집 | `transcript_rescrape.py` |
| 3 | 자막을 1200자 단위로 청크 분할 | `chunk_data.py` |
| 4 | 키워드 매칭만 하던 초기 검색 (단순 문자열) | `app.py: _keyword_search` 초기 버전 |
| 5 | ChromaDB 도입, 한국어 임베딩 모델로 의미 검색 | `vector_store.py`, `build_vector_db.py` |
| 6 | 채널 분석(통계 + Plotly 차트 + Gemini 인사이트) | `app.py: compute_analytics`, `llm.py: generate_analytics_summary` |
| 7 | Intent 라우팅(분석 vs 에피소드 검색) + 날짜·채널 추출 | `llm.py: detect_intent` |
| **8 (이번 개선)** | **description/tags 수집·메타 청크 도입·점수 집계 개선·LLM 프롬프트 개선** | `enrich_metadata.py` 신규 외 다수 |

## 4. 이번 개선(2026-04-27)에서 해결한 문제

### 문제 1 — 답변이 "정확히 일치하는 영상은 없지만…"으로 도망감

**원인**
- 검색 후보가 자막(transcript) 청크에서만 뽑혀 나옴
- 출연자 이름(예: "최예나")이 자막에는 음차(ex. "예나")로만 나와 매칭 약함
- LLM 프롬프트에 "확신 낮으면 솔직히 밝혀줘"가 강해서 회피로 빠짐

**해결**
- 영상 **description**과 **tags**도 수집(`data_collector.py`, `enrich_metadata.py` 신규)
- 영상별 **메타 청크**(제목 + 설명 + 태그) 추가 → 검색 시 가중치 +0.15
- 같은 영상의 여러 청크는 **video_id 단위로 집계**, 매치 개수 보너스
- LLM 프롬프트를 "찾으시는 영상은 **[제목](링크)** 인 것 같아요" 형식으로 강제

### 문제 2 — 검색이 느림

**원인**
- 검색마다 `SentenceTransformer`를 새로 인스턴스화 → 매번 3~5초 모델 로딩

**해결**
- `vector_store.get_embedder()` 모듈 레벨 싱글턴 캐시
- Streamlit 단에서도 `@st.cache_resource`로 앱 시작 시 1회만 로드
- ChromaDB query를 1회 호출로 후보 넉넉히 가져온 뒤 video 단위 집계

### 문제 3 — 모바일 앱에서 보이는 짧은 영상 요약을 못 가져옴

**원인**
- 그 짧은 요약은 영상의 **description** 앞부분
- 기존 `data_collector.py`가 `snippet.description`을 무시했음

**해결**
- `snippet.description`, `snippet.tags`, `snippet.channelTitle`,
  `statistics.commentCount` 모두 수집
- 기존 데이터를 다시 긁지 않아도 되도록 `enrich_metadata.py` 별도 제공

## 5. 검색 정확도 검증 (이번 개선 후)

| 질문 | Top-1 결과 | 점수 |
|------|-----------|------|
| 최예나랑 삼겹살 먹는편 | 묵은지 고추장 삼겹살 (실비집 EP.10) | 0.78 |
| 이서진이랑 떡국 끓여먹던 편 | 새해 인사는 핑계고 (EP.35) | 0.73 |
| 연말 홈파티 | EP.10 연말 홈파티와 단짝 코미디언 | 0.80 |
| 추석에 모인 편 | EP.57 추석에 놀러온 건 핑계고 | 0.82 |

> ※ "최예나"는 자막의 음차 외에 description/tags에 직접 명시가 없는 케이스라
> 완벽 매칭은 어려움(데이터 한계). 다만 점수 1·2위 사이가 분명히 벌어지므로
> LLM이 1위를 자신있게 추천하도록 프롬프트로 보완.

## 6. 디렉토리 구조

```
YTchannel_analyze RAG/
├── data/
│   ├── raw/                    # *_raw_data.json (메타 + 자막)
│   └── processed/              # *_chunks.jsonl (메타 청크 + 자막 청크)
├── chroma_db/                  # 벡터 인덱스 (채널별 collection)
├── .hf_cache/                  # SentenceTransformer 캐시
├── scripts/
│   ├── app.py                  # Streamlit 진입점
│   ├── llm.py                  # Gemini 호출 + 룰베이스 fallback
│   ├── vector_store.py         # ChromaDB 인덱싱·검색·집계
│   ├── data_collector.py       # 신규 채널 1차 수집
│   ├── enrich_metadata.py      # 기존 데이터에 description/tags 보강 (신규)
│   ├── transcript_rescrape.py  # 빈 자막 yt-dlp 재수집
│   ├── chunk_data.py           # 메타+자막 청크 생성
│   ├── build_vector_db.py      # 청크 → Chroma
│   ├── channels.py             # 채널 정의
│   └── paths.py                # 경로 헬퍼
└── DEVELOPMENT.md              # ← 이 문서
```

## 7. 자주 쓰는 커맨드

```bash
# 새 채널 데이터를 처음부터 (data_collector.py 안의 TARGET_CHANNEL_ID 수정)
.venv/bin/python scripts/data_collector.py

# 기존 raw에 description/tags 보강
.venv/bin/python scripts/enrich_metadata.py --all
.venv/bin/python scripts/enrich_metadata.py --input data/raw/ddeunddeun_raw_data.json

# 자막이 빈 영상만 yt-dlp로 재수집
.venv/bin/python scripts/transcript_rescrape.py --input data/raw/ddeunddeun_raw_data.json

# 청크 재생성 (raw → processed)
.venv/bin/python scripts/chunk_data.py --input data/raw/ddeunddeun_raw_data.json

# 벡터DB 전체 재빌드 (모든 채널)
.venv/bin/python scripts/build_vector_db.py

# 특정 채널만 재빌드
.venv/bin/python scripts/build_vector_db.py --channels ddeunddeun

# Streamlit 실행
.venv/bin/streamlit run scripts/app.py
```

## 8. 앞으로 수정·확장할 점 (우선순위 순)

### 단기 (며칠 안에 가능)

- [ ] **댓글(top comments) 수집** — `commentThreads.list` API로 영상별 인기 댓글 50개 수집.
      댓글에는 "이거 ○○○ 나온 편이죠?" 같은 검색 핵심 단서가 잔뜩 있음.
      메타 청크에 합쳐 넣으면 출연자 이름 매칭이 크게 좋아짐.
- [ ] **출연자 자동 추출** — description의 `#`해시태그 + 정규식으로 인물 리스트를
      별도 메타 필드(`guests: ["유재석", "김원희", ...]`)로 추출 → 청크에 명시 포함.
- [ ] **검색 결과 디버그 패널** — Streamlit 사이드바에 "검색 점수·매치 청크 종류" 토글로
      표시해서 왜 그 영상이 골라졌는지 사용자가 확인할 수 있게.
- [ ] **Hybrid retrieval (BM25 + 벡터)** — 인물 이름·고유명사처럼 정확 매칭이 필요한 토큰을
      위해 `rank_bm25` 같은 가벼운 라이브러리로 키워드 점수를 함께 계산하고 RRF로 결합.
- [ ] **Streamlit 자동 재시작 안내** — 데이터/모델 변경 후 사용자가 수동 새로고침해야 하는
      문제. 캐시 키에 데이터 mtime을 넣어 자동 invalidate.

### 중기 (1~2주)

- [ ] **썸네일 카드 UI** — 현재 결과는 텍스트 리스트. `thumbnail_url`을 활용해
      카드형 그리드로 보여주면 인지성 ↑.
- [ ] **타임스탬프 단위 자막 청크** — 현재는 자막을 글자 수로 자름. 시간 정보를 살려서
      "이 장면은 27분 13초쯤" 같이 답하도록 개선. (`youtube-transcript-api`가 시간 줌)
- [ ] **다중 채널 비교 분석** — 분석 모드에서 두 채널 동시에 비교 (월별 업로드, 평균 조회수).
- [ ] **Re-ranking 단계 추가** — 의미 검색으로 top 30 가져온 뒤 Cross-encoder로
      재정렬 (예: `bongsoo/kpf-cross-encoder-v1`). 정확도가 한 계단 더 올라감.
- [ ] **Description 길면 길수록 임베딩이 약해지는 문제** — 메타 청크가 너무 길면 핵심
      신호가 묽어짐. 첫 N문장 + 태그만 사용하는 옵션 추가.

### 장기 / 아이디어

- [ ] **에피소드 게스트 그래프** — 누가 자주 같이 출연했는지 네트워크 그래프.
- [ ] **자동 요약 생성** — 영상마다 Gemini로 한 줄 요약을 만들어 description 대신 사용
      (description이 광고·해시태그·크레딧 위주인 경우 대비).
- [ ] **LLM 답변 캐싱** — 같은 질문 반복 시 즉시 응답. (질문 → 영상id 매핑 sqlite).
- [ ] **다국어 지원** — 영어 질문도 받기 (`paraphrase-multilingual` 임베딩 함께 운영).
- [ ] **댓글 토픽 클러스터링** — 분석 모드에서 "이 채널 시청자들이 많이 언급하는 주제 TOP 10".
- [ ] **회차 자동 인덱스(번호 누락)** — `EP.N` 패턴 외에 "n번째", "n화" 등 다양한 변형 인식.

## 9. 알려진 제약·주의사항

- **YouTube Data API quota**: 하루 10,000 unit. 영상 메타 1건 당 1 unit, snippet 포함이면
  좀 더. 채널 전체 재수집은 quota 거의 다 씀.
- **자막 추출 IP 차단**: `youtube_transcript_api` 가 IP rate limit 걸리면 빈 transcript로
  떨어짐 → `transcript_rescrape.py`로 yt-dlp 백업.
- **15야 채널은 자막 거의 없음** — 메타 청크 위주로만 검색 가능. 정상 동작 중.
- **HuggingFace 모델 첫 로드는 인터넷 필요** — `.hf_cache/` 가 비어있으면 첫 실행 시
  ~500MB 다운로드. 이후엔 오프라인도 가능 (`HF_HUB_OFFLINE=1`).
- **Streamlit hot-reload 한계** — 모듈 import 변경 후엔 streamlit 재시작 필요.

## 10. 환경 변수 (`.env`)

```
YOUTUBE_API_KEY=...
GEMINI_API_KEY=...           # 없어도 룰 기반 fallback 동작
GEMINI_MODEL=gemini-flash-latest   # 선택
YTDLP_COOKIES_FROM_BROWSER=chrome  # 선택, 자막 차단 우회용
```
