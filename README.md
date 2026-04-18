# YTchannel_analyze-RAG

유튜브 자막 수집이 중간에 멈췄을 때, 기존 JSON을 이어서 재수집할 수 있습니다.

## 재수집(이어받기) 빠른 사용법

`transcript_rescrape.py`는 기존 JSON을 읽고, 실패/빈 자막 항목만 다시 시도해 같은 JSON에 덮어씁니다.

```bash
python transcript_rescrape.py --input /path/to/yesterday_data.json
```

### 로컬 JSON을 읽어서 다른 파일로 저장하고 싶을 때

```bash
python transcript_rescrape.py \
  --input /path/to/yesterday_data.json \
  --output /path/to/yesterday_data.updated.json
```

### 배치로 나눠서 재시도(차단/429 완화)

예: 실패 대상이 205개면 41개씩 5번 실행

```bash
python transcript_rescrape.py --input /path/to/yesterday_data.json --batch-size 41 --batch 1
python transcript_rescrape.py --input /path/to/yesterday_data.json --batch-size 41 --batch 2
python transcript_rescrape.py --input /path/to/yesterday_data.json --batch-size 41 --batch 3
python transcript_rescrape.py --input /path/to/yesterday_data.json --batch-size 41 --batch 4
python transcript_rescrape.py --input /path/to/yesterday_data.json --batch-size 41 --batch 5
```

### 저장 주기 조절

기본값은 `--save-every 1`이라서 영상 1개 처리할 때마다 즉시 저장합니다.

```bash
python transcript_rescrape.py --input /path/to/yesterday_data.json --save-every 1
```
