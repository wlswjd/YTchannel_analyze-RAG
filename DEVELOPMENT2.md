# DEVELOPMENT2.md — RAG 개선 작업 로그

> 작업일: 2026-05-09
> 기반: CLAUDE.md의 RAG 학습 Day 1~6 개선사항

---

## 작업 개요

한국어 유튜브 채널 RAG 챗봇의 검색 품질 개선을 위해 총 5단계 작업을 진행했습니다.
기존 Dense 전용 검색(ChromaDB)에서 **Dense + BM25 Hybrid Search(RRF)** 구조로 전환.

---

## 단계별 진행 내역

### 1단계 ✅ 메타데이터 보강 — `chunk_data.py`

**목적:** ChromaDB `where` 필터 활용을 위한 메타데이터 정형화

**변경사항:**
- `from datetime import datetime` 추가
- `_parse_upload_date(published_at)` 함수 신규 추가
  - ISO 8601 / YYYYMMDD 등 다양한 형식 처리
  - 파싱 실패 시 `0` 반환
  - 지원 포맷: `%Y-%m-%dT%H:%M:%SZ`, `%Y-%m-%dT%H:%M:%S`, `%Y-%m-%d`, `%Y%m%d`
- `base_meta`에 `upload_date` 정수 필드 추가 (예: `20240315`)
- `chunk_kind` 필드명 → `chunk_type` 로 변경
- 청크 타입 값 변경:
  - `"meta"` → `"metadata"`
  - `"transcript"` → `"transcript"` (유지)

---

### 1단계 연계 ✅ `vector_store.py` — upsert + search 업데이트

**변경사항 (upsert_chunks_jsonl):**
- `chunk_type` 결정 로직 (구 jsonl 하위 호환 폴백 포함):
  ```python
  chunk_type = row.get("chunk_type") or (
      "metadata" if row.get("chunk_kind") == "meta" or row.get("chunk_index", 0) == -1
      else "transcript"
  )
  ```
- ChromaDB 메타에 `upload_date` (int) 추가
- `chunk_kind` → `chunk_type` 교체
- 카운터도 `chunk_type == "metadata"` 기준으로 변경

**변경사항 (semantic_search):**
- boost 판단: `chunk_type == "metadata"` 우선, `chunk_kind == "meta"` 폴백
  ```python
  kind = meta.get("chunk_type") or meta.get("chunk_kind") or "transcript"
  if kind in ("metadata", "meta"):
      score += META_CHUNK_BOOST
  ```
- **기존 video_id 집계 로직 및 match_count 보너스 유지**

---

### 2단계 ✅ HNSW 파라미터 명시 — `vector_store.py`

**변경사항 (upsert_chunks_jsonl의 컬렉션 생성):**

```python
col = client.get_or_create_collection(
    name=name,
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 16,
        "hnsw:construction_ef": 100,
        "hnsw:search_ef": 30,
        "model": model_name,
        "channel": channel_id,
    },
)
```

**주의:** HNSW 파라미터는 컬렉션 생성 시점에만 적용됨. 기존 chroma_db에 반영하려면 재인덱싱(`build_vector_db.py`) 필요.

---

### 3단계 ✅ Hybrid Search 모듈 — `hybrid_search.py` 신규

**파일:** `scripts/hybrid_search.py`

**검색 흐름:**
```
Dense 검색 (ChromaDB, Top-30)
        +
BM25 검색 (채널별 pkl, Top-30)
        ↓
   RRF 결합 (k=60)
        ↓
video_id 단위 집계
(RRF 점수 합산 + 메타 청크 가중치 +0.15 + match_count 보너스 최대 +0.10)
        ↓
   Top-K 영상 반환
```

**주요 구성:**

| 함수 | 역할 |
|------|------|
| `_tokenize(text)` | 한국어+영숫자 정규식 토큰화 (형태소 분석 없이) |
| `build_bm25_index(channel_id, chunks_jsonl)` | BM25Okapi 인덱스 빌드 후 pickle 저장 |
| `_load_bm25(channel_id)` | pkl 로드, 없으면 `None` |
| `hybrid_search(channel_ids, query, n_results)` | 메인 검색 함수 |

**파라미터:**
- BM25: `k1=1.5, b=0.75`
- RRF: `k=60`
- Top-N: `30` (Dense/BM25 각각)
- BM25 인덱스 저장 위치: `data/bm25/bm25_{channel_id}.pkl`

**폴백:** BM25 pkl 없는 채널 → Dense 전용 자동 폴백 (검색 중단 없음)

**의존성 추가 (`requirements.txt`):**
```
rank-bm25==0.2.2
```

---

### 4단계 ✅ app.py 검색 함수 교체

**변경사항:**
```python
# Before
from vector_store import get_embedder, semantic_search

results = semantic_search(channel_ids=channel_ids, query=query, n_results=n)

# After
from hybrid_search import hybrid_search
from vector_store import get_embedder

results = hybrid_search(channel_ids=channel_ids, query=query, n_results=n)
```

**최종 폴백 체인:**
```
hybrid_search (Dense + BM25 RRF)
  └─ BM25 pkl 없으면 → Dense 전용 (hybrid_search 내부 자동)
       └─ ChromaDB 연결 실패 → _keyword_search (app.py)
```

---

### 5단계 ✅ BM25 빌드 스크립트 — `build_bm25_index.py` 신규

**파일:** `scripts/build_bm25_index.py`

**사용법:**
```bash
# 전체 채널 빌드
python scripts/build_bm25_index.py

# 특정 채널만
python scripts/build_bm25_index.py --channels ddeunddeun ssookssook
```

**구조:** `build_vector_db.py`와 동일한 패턴. `chunks_path(ch)` 없으면 SKIP.

---

## 변경된 파일 목록

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `scripts/chunk_data.py` | 수정 | `upload_date` 추가, `chunk_kind` → `chunk_type`, "meta" → "metadata" |
| `scripts/vector_store.py` | 수정 | `chunk_type`/`upload_date` 저장, HNSW 파라미터, boost 판단 업데이트 |
| `scripts/hybrid_search.py` | 신규 | Dense+BM25 Hybrid Search, RRF, video_id 집계 |
| `scripts/build_bm25_index.py` | 신규 | BM25 인덱스 빌드 CLI |
| `scripts/app.py` | 수정 | import 및 검색 함수 교체 |
| `requirements.txt` | 수정 | `rank-bm25==0.2.2` 추가 |

---

## 다음 작업 (미완료)

- [x] **BM25 인덱스 실제 빌드** — 완료 (2026-05-09)
- [x] **재인덱싱** — HNSW 파라미터 + 새 메타데이터 반영 완료 (2026-05-09)
- [x] **청크 크기 축소** — `chunk_size=400, overlap=80` 완료 (2026-05-09)
  - 뜬뜬: 6,032 → 18,654 자막 청크
  - 쑥쑥: 1,060 → 3,253 자막 청크
  - 채널십오야: 26 → 82 자막 청크
- [ ] **형태소 분석 기반 토큰화** — `_tokenize()` 개선 (konlpy 등)
- [ ] **ChromaDB `where` 필터** 실제 적용 (날짜 범위 검색 등)
