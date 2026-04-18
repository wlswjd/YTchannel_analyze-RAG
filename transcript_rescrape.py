import argparse
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path


WORKDIR = Path(__file__).resolve().parent


def _vtt_to_text(vtt: str) -> str:
    # Remove WEBVTT headers, timestamps, and cue settings. Keep human-readable lines.
    lines = []
    for raw in vtt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper() == "WEBVTT":
            continue
        if line.startswith("NOTE"):
            continue
        if "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            # cue number
            continue
        # common vtt tags
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    # De-dup consecutive identical lines (VTT often repeats)
    out = []
    prev = None
    for l in lines:
        if l == prev:
            continue
        out.append(l)
        prev = l
    return " ".join(out).strip()


def _run_yt_dlp_fetch_vtt(video_id: str, langs: str) -> str | None:
    """
    Returns transcript text if subtitles found, otherwise None.
    Uses yt-dlp to write VTT files into workspace temp folder.
    """
    tmp_dir = WORKDIR / ".tmp_subs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = WORKDIR / ".yt-dlp-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Avoid reading stale VTT from a previous run for the same video_id.
    for old in tmp_dir.glob(f"{video_id}*.vtt"):
        try:
            old.unlink()
        except OSError:
            pass

    url = f"https://www.youtube.com/watch?v={video_id}"
    outtmpl = str(tmp_dir / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        langs,
        "--sub-format",
        "vtt",
        "--retries",
        "10",
        "--sleep-interval",
        "3",
        "--max-sleep-interval",
        "7",
        "--output",
        outtmpl,
        "--cache-dir",
        str(cache_dir),
        url,
    ]

    # Optional: authenticated cookies can greatly reduce throttling.
    # - If you place a cookies.txt in the project root, it will be used.
    # - Or set YTDLP_COOKIES_FROM_BROWSER (e.g. "chrome", "brave", "safari") to use browser cookies.
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    else:
        cookies_txt = WORKDIR / "cookies.txt"
        if cookies_txt.exists():
            cmd.extend(["--cookies", str(cookies_txt)])

    # Avoid local proxy envs interfering; also helps reproducibility.
    env = os.environ.copy()
    for k in list(env.keys()):
        if "proxy" in k.lower():
            env.pop(k, None)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if err:
            last = "\n".join(err.splitlines()[-5:])
            print(f"    yt-dlp 실패 ({video_id}, langs={langs}):\n    {last}")
        return None

    # yt-dlp may create one of:
    #   <id>.ko.vtt, <id>.ko-KR.vtt, <id>.en.vtt, <id>.en.auto.vtt, etc.
    candidates = sorted(tmp_dir.glob(f"{video_id}.*.vtt"), key=lambda p: p.name)
    if not candidates:
        print(
            f"    yt-dlp는 성공했는데 VTT를 못 찾음 ({video_id}, langs={langs}). "
            f"tmp={tmp_dir}"
        )
        return None

    # Prefer KO, then EN, then anything.
    def rank(p: Path) -> tuple[int, int]:
        name = p.name.lower()
        if ".ko." in name or ".ko-kr." in name:
            return (0, 0 if ".auto." not in name else 1)
        if ".en." in name:
            return (1, 0 if ".auto." not in name else 1)
        return (2, 0 if ".auto." not in name else 1)

    candidates.sort(key=rank)
    vtt = candidates[0].read_text(encoding="utf-8", errors="replace")
    text = _vtt_to_text(vtt)
    return text or None


def main():
    parser = argparse.ArgumentParser(
        description="재수집 대상만 yt-dlp로 자막을 채우고 JSON에 반영합니다. "
        "ddeunddeun_raw_data.json을 읽고 같은 파일에 덮어씁니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKDIR / "ddeunddeun_raw_data.json",
        help="입력/출력 JSON 경로 (기본: ddeunddeun_raw_data.json)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        metavar="N",
        help="재수집 대상 목록을 batch-size개씩 나눌 때 몇 번째 묶음인지 (1부터). "
        "예: 205개, batch-size 41이면 1~5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=41,
        help="한 번에 처리할 재수집 대상 개수 (기본: 41)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = input_path

    data = json.loads(input_path.read_text(encoding="utf-8"))

    # Only retry ones that previously errored or were placeholders.
    def needs_retry(t: str) -> bool:
        if not t:
            return True
        if "자막 추출 중 에러" in t:
            return True
        if "자막이 제공되지 않는 영상" in t:
            return True
        # 새 플레이스홀더 (yt-dlp 실패 시)
        if "자막이 제공되지 않거나 수집할 수 없는 영상" in t:
            return True
        return False

    all_retry_targets = [v for v in data if needs_retry(v.get("transcript", ""))]
    total_retry = len(all_retry_targets)
    print(f"총 {len(data)}개 중 재수집 대상 {total_retry}개")

    if args.batch is not None:
        if args.batch < 1:
            raise SystemExit("--batch는 1 이상이어야 합니다.")
        start = (args.batch - 1) * args.batch_size
        end = min(start + args.batch_size, total_retry)
        if start >= total_retry:
            print(f"배치 {args.batch}: 범위 밖입니다 (재수집 대상 {total_retry}개, 시작 인덱스 {start}).")
            return
        targets = all_retry_targets[start:end]
        print(
            f"배치 {args.batch}: 재수집 대상 중 인덱스 {start + 1}~{end} "
            f"({len(targets)}개 처리)"
        )
    else:
        targets = all_retry_targets
        print(f"전체 재수집 대상 {len(targets)}개를 한 번에 처리합니다.")

    # Manual `yt-dlp --sub-langs ko`가 잘 되는 경우가 많아, 먼저 단일 언어(ko, en)만 시도.
    # `ko.*`는 트랙이 많아질 수 있어 429에 더 잘 걸립니다.
    lang_attempts = ["ko", "en", "ko.*", "en.*"]

    ok = 0
    none = 0
    for idx, v in enumerate(targets, start=1):
        vid = v["video_id"]
        title = v.get("title", "")

        text = None
        for attempt_langs in lang_attempts:
            # retry with exponential backoff on throttling/temporary errors
            for retry in range(4):
                text = _run_yt_dlp_fetch_vtt(vid, langs=attempt_langs)
                if text:
                    break
                time.sleep(min(60.0, (2.0 ** retry) * random.uniform(2.0, 4.0)))
            if text:
                break

        if text:
            v["transcript"] = text
            ok += 1
            print(f"[{idx}/{len(targets)}] OK: {title} ({vid})")
        else:
            v["transcript"] = "자막이 제공되지 않거나 수집할 수 없는 영상입니다."
            none += 1
            print(f"[{idx}/{len(targets)}] NONE: {title} ({vid})")

        # Gentle pacing to reduce blocking
        time.sleep(random.uniform(4.0, 7.0))

        # checkpoint every 10
        if idx % 10 == 0:
            output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: OK {ok}, NONE {none}")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()

