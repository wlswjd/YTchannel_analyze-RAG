import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, resolve_raw_json  # noqa: E402


def _vtt_to_text(vtt: str) -> str:
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
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    out = []
    prev = None
    for l in lines:
        if l == prev:
            continue
        out.append(l)
        prev = l
    return " ".join(out).strip()


def _run_yt_dlp_fetch_vtt(video_id: str, langs: str) -> str | None:
    tmp_dir = ROOT / ".tmp_subs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / ".yt-dlp-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

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

    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    else:
        cookies_txt = ROOT / "cookies.txt"
        if cookies_txt.exists():
            cmd.extend(["--cookies", str(cookies_txt)])

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

    candidates = sorted(tmp_dir.glob(f"{video_id}.*.vtt"), key=lambda p: p.name)
    if not candidates:
        print(
            f"    yt-dlp는 성공했는데 VTT를 못 찾음 ({video_id}, langs={langs}). "
            f"tmp={tmp_dir}"
        )
        return None

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
        description="재수집 대상만 yt-dlp로 자막을 채우고 JSON에 반영합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="입력/출력 JSON (예: data/raw/ssookssook_raw_data.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 JSON 경로 (기본: --input과 동일)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        metavar="N",
        help="재수집 대상을 batch-size개씩 나눌 때 몇 번째 묶음인지 (1부터).",
    )
    parser.add_argument("--batch-size", type=int, default=41)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="N개 처리마다 중간 저장",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path

    data = json.loads(input_path.read_text(encoding="utf-8"))

    def needs_retry(t: str) -> bool:
        if not t:
            return True
        if "자막 추출 중 에러" in t:
            return True
        if "자막이 제공되지 않는 영상" in t:
            return True
        if "자막이 제공되지 않거나 수집할 수 없는 영상" in t:
            return True
        return False

    all_retry_targets = [v for v in data if needs_retry(v.get("transcript", ""))]
    total_retry = len(all_retry_targets)
    print(f"총 {len(data)}개 중 재수집 대상 {total_retry}개")
    print(f"입력: {input_path}")
    print(f"출력: {output_path}")

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

    lang_attempts = ["ko", "en"]

    ok = 0
    none = 0
    for idx, v in enumerate(targets, start=1):
        vid = v["video_id"]
        title = v.get("title", "")

        text = None
        for attempt_langs in lang_attempts:
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

        time.sleep(random.uniform(4.0, 7.0))

        if args.save_every > 0 and idx % args.save_every == 0:
            output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: OK {ok}, NONE {none}")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
