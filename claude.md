# YT RAG (YTchannel_analyze-RAG) — 작업 컨텍스트

## 프로젝트 개요
한국어 유튜브 채널(뜬뜬, 쑥쑥, 채널십오야) 자막 + 메타데이터 기반 RAG 챗봇.
- 데이터 수집: `data_collector.py`, `transcript_rescrape.py`, `enrich_metadata.py`
- 청킹: `chunk_data.py` — 메타 청크(제목+설명+태그) + 자막 청크(1200자, overlap 200)
- 인덱싱: `build_vector_db.py` — ChromaDB raw client 사용 (LangChain 아님)
- 검색 + LLM: `app.py` (Streamlit) + `vector_store.py` (검색) + `llm.py` (Gemini)
- 임베딩 모델: `jhgan/ko-sroberta-multitask`
- ChromaDB 컬렉션: 채널별 분리 (ddeunddeun, ssookssook, sibsibya 등)

## 현재 검색 구조 (vector_store.py 추정)
```python
# 의미 검색 → video_id 단위 점수 집계 → 메타 청크 가중치 +0.15
# 자세한 흐름은 vector_store.py 참고
```
→ 순수 Dense 검색만 사용. 도메인 특화 로직(메타 청크 가중치, video_id 집계)은 잘 구현되어 있음.
→ Hybrid Search, Re-ranking, ChromaDB where 필터 미적용.

## 적용할 개선사항 (RAG 학습 Day 1~6 기반)

### 1. 메타데이터 보강 (chunk_data.py 또는 build_vector_db.py)
현재 메타데이터 필드를 명확히 정리하고, ChromaDB의 where 필터로 활용 가능하도록 정형화.

권장 메타데이터 구성:
- `channel`: "ddeunddeun" / "ssookssook" / "sibsibya" 등
- `video_id`: 영상 고유 ID (이미 있을 가능성 높음)
- `video_title`: 영상 제목
- `upload_date`: YYYYMMDD 정수 (예: 20240315). 누락 시 0
  → ChromaDB의 비교 연산자($gte, $lte)는 정수에만 작동
- `chunk_type`: "metadata" / "transcript"
  → 기존 메타 청크 가중치 로직과 연동 가능
- `view_count`: 조회수 (정수, 분석 모드에서 활용)
- `timestamp_start`, `timestamp_end`: 자막 청크의 경우 시간 정보 (있다면)

### 2. HNSW 파라미터 명시 (vector_store.py)
- 현재: ChromaDB 컬렉션 생성 시 metadata 인자에 HNSW 설정 없을 가능성
- 추가: `metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 100, "hnsw:search_ef": 30}`
- 주의: normalize_embeddings 적용 여부 확인 필요 (적용되어 있으면 cosine 효과는 미미)

### 3. Hybrid Search 도입 (신규 모듈 권장)
**중요**: 기존 video_id 단위 집계 로직 + 메타 청크 가중치 +0.15 로직과 통합해야 함.

권장 구조:
1. Dense 검색: ChromaDB로 청크 단위 Top-N 가져오기
2. BM25 검색: 같은 청크들에 대해 BM25 점수 (채널별 별도 인덱스)
3. RRF 결합 (k=60): 청크 단위 통합 순위
4. **video_id 단위 집계**: RRF 결합 후 같은 video_id의 청크들 점수 합산 + 메타 청크 가중치
5. 최종 Top-K 영상 반환

기존 점수 집계 로직을 RRF 결합 후에 적용하는 형태로 통합.

라이브러리:
- `rank-bm25` (requirements.txt 추가 필요)
- BM25 인덱스는 채널별로 pickle 분리 (예: `bm25_ddeunddeun.pkl`)

### 4. (선택) 청크 크기 축소
- 현재: chunk_size=1200, overlap=200 (chunk_data.py)
- 문제: ko-sroberta 토큰 한계(약 300~400자) 크게 초과 → 자막 뒷부분이 임베딩에 반영 안 됨
- 개선: chunk_size=400, overlap=80
- 단, 재인덱싱 + BM25 재빌드 필요 → 마크 RAG처럼 일단 보류 가능

## 결정된 파라미터
- 청크: chunk_size=400, overlap=80 (재인덱싱 시)
- HNSW: space=cosine, M=16, construction_ef=100, search_ef=30
- Hybrid: RRF k=60, Top-N=30 → Top-K=5 (청크 단위)
- video_id 집계 후: Top 5 영상 반환 (기존 로직 유지)
- BM25: k1=1.5, b=0.75

## 작업 순서 권장
1. 메타데이터 보강 — chunk_data.py에서 청크 만들 때 메타데이터 필드 정리
   - 신규 데이터부터 적용 (기존 chroma_db는 그대로)
2. HNSW 파라미터 명시 — vector_store.py / build_vector_db.py
3. Hybrid Search 모듈 분리 — `hybrid_search.py` 신규 파일
   - 기존 video_id 집계 로직과 통합되도록 설계
4. app.py / vector_store.py 검색 함수 교체
5. 폴백 처리 — BM25 인덱스 없으면 기존 Dense 검색

## 주의사항
- LangChain Chroma 아닌 raw chromadb 사용 중 → 마크 RAG와 코드 패턴 다름
- 기존 도메인 특화 로직(video_id 집계, 메타 청크 가중치)을 깨지 않도록 주의
- Intent 라우팅(분석 vs 에피소드 검색)은 그대로 유지, 검색 부분만 교체
- 채널별로 BM25 인덱스 분리 (마크 RAG의 컬렉션별 분리와 같은 패턴)
- 한국어 토큰화는 일단 단순 정규식 (마크 RAG와 동일), 형태소 분석은 추후

## 참고
- 마크 RAG 적용 사례: 같은 학습 기반으로 마크 RAG에 Hybrid Search 적용 완료
  → `hybrid_search.py` 모듈 구조 참고 가능
- 미니 프로젝트(`rag-study/day7_mini/`): 모든 기법 통합 예시