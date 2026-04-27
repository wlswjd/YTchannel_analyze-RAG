# YTchannel_analyze-RAG

특정 유튜브 채널(뜬뜬·쑥쑥·채널십오야)의 영상을 수집해서, **RAG(Retrieval-Augmented Generation)** 로
*"이서진이랑 떡국 끓여먹던 편 뭐였지?"* 같은 자연어 질문에 답해주는 Streamlit 봇입니다.

---

## RAG 파이프라인

```
사용자 질문
    │
    │ (1) 한국어 임베딩 (jhgan/ko-sroberta-multitask)
    ▼
ChromaDB(local persistent) 의미 검색 ──► R (Retrieval)
    │  └─ 영상별 메타 청크(제목+설명+태그) + 자막 청크 통합 검색
    │  └─ video_id 단위 점수 집계 + 메타 청크 가중치 부스트
    ▼
검색 컨텍스트(후보 + description + 자막 발췌) 프롬프트 주입 ──► A (Augmented)
    ▼
Gemini 답변 생성 ──────────────────────────────────────────► G (Generation)
    │  └─ "찾으시는 영상은 [제목](링크) 인 것 같아요…"
    ▼
Streamlit 채팅 UI 렌더링
```

검색이 실패해도 키워드 가중치(제목 ×6, 설명 ×4, 자막 ×1) 기반 fallback이 돌고,
LLM 키가 없으면 룰 기반 fallback 답변이 나갑니다.

---

## 폴더 구조

| 경로 | 역할 |
|------|------|
| `scripts/` | 실행 스크립트 (`paths.py`가 프로젝트 루트 `ROOT` 기준) |
| `data/raw/` | `*_raw_data.json` 수집본 (메타 + 자막) |
| `data/processed/` | 청크 `.jsonl` (메타 청크 + 자막 청크) |
| `chroma_db/` | ChromaDB persistent 벡터 인덱스 |
| `.hf_cache/` | SentenceTransformer 모델 캐시 |
| `.env` | `YOUTUBE_API_KEY`, `GEMINI_API_KEY` |
| `DEVELOPMENT.md` | 상세 개발 기록 + 로드맵 |

---

## 스크립트 요약

| 스크립트 | 설명 |
|----------|------|
| `scripts/data_collector.py` | YouTube Data API → 채널 업로드 목록 + **메타(제목/설명/태그/통계)** + 자막 1차 수집 |
| `scripts/enrich_metadata.py` | 이미 모은 raw에 **description/tags** 보강 (재수집 없이) |
| `scripts/transcript_rescrape.py` | 실패·빈 자막만 **yt-dlp**로 재수집, 같은 JSON에 덮어쓰기 |
| `scripts/chunk_data.py` | Raw JSON → 영상별 ① **메타 청크**(제목+설명+태그) ② 자막 청크(1200자, 200 overlap) |
| `scripts/build_vector_db.py` | 청크를 임베딩해서 ChromaDB persistent collection에 upsert |
| `scripts/vector_store.py` | 임베딩 모델 싱글턴 + 의미 검색 + video_id 단위 점수 집계 |
| `scripts/llm.py` | Gemini 호출(thinking off, 토큰 자동 보정) + 의도 분석 + 룰 기반 fallback |
| `scripts/app.py` | Streamlit 챗봇 UI: 에피소드 검색 / 채널 분석 라우팅, 사이드바 도구 |
| `scripts/channels.py` | 채널 정의 (`뜬뜬`, `쑥쑥`, `채널십오야`) |
| `scripts/paths.py` | 프로젝트 루트 기준 경로 헬퍼 |

---

## 권장 실행 순서

### 0) 환경 변수

`.env`:

```
YOUTUBE_API_KEY=...
GEMINI_API_KEY=...                  # 없어도 룰 기반 fallback 동작
GEMINI_MODEL=gemini-flash-latest    # 선택
YTDLP_COOKIES_FROM_BROWSER=chrome   # 선택, 자막 차단 우회용
```

### 1) 데이터 수집 (새 채널)

```bash
.venv/bin/python scripts/data_collector.py
```

`scripts/data_collector.py` 안의 `TARGET_CHANNEL_ID`, `output_filename` 을 새 채널 값으로 수정 후 실행.

### 2) (옵션) 기존 데이터 보강 / 자막 재수집

```bash
# 모든 raw에 description/tags 보강
.venv/bin/python scripts/enrich_metadata.py --all

# 빈 자막만 yt-dlp로 재시도
.venv/bin/python scripts/transcript_rescrape.py --input data/raw/ddeunddeun_raw_data.json
```

### 3) 청크 생성

```bash
.venv/bin/python scripts/chunk_data.py --input data/raw/ddeunddeun_raw_data.json
```

영상 1편당 메타 청크 1개 + 자막 청크 N개가 생성되어 `data/processed/<이름>_chunks.jsonl` 로 저장됩니다.

### 4) 벡터 DB 빌드

```bash
# 모든 채널 한 번에
.venv/bin/python scripts/build_vector_db.py

# 특정 채널만
.venv/bin/python scripts/build_vector_db.py --channels ddeunddeun
```

처음 실행 시 약 500MB 임베딩 모델이 `.hf_cache/` 로 다운로드됩니다.
이후엔 오프라인 가능 (`HF_HUB_OFFLINE=1`).

### 5) 챗봇 실행

```bash
.venv/bin/streamlit run scripts/app.py
```

브라우저에서:

- **에피소드 검색** — *"이서진이랑 떡국 끓여먹던 편이 뭐였지?"* → 의미 검색 + Gemini 추천
- **채널 분석** — *"뜬뜬 채널 25년 4월부터 26년 4월까지 분석해줘"* → 차트 + Gemini 인사이트
- **사이드바** — 채널 선택, 데이터 현황, 예시 질문, 대화 초기화

---

## 자세한 개발 기록

전체 개발 흐름·설계 의사결정·로드맵은 [`DEVELOPMENT.md`](./DEVELOPMENT.md) 참고.
