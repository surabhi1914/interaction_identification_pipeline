#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests

DEFAULT_PROJECT = Path(
    "/rs1/researchers/a/amallav/species_classification/ecological_informatics"
)
COLUMNS = [
    "observation_id",
    "photo_id",
    "photo_url",
    "download_url",
    "image_size",
    "local_path",
    "status",
    "bytes_downloaded",
    "attempts",
    "error",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    project = Path(os.environ.get("PROJECT_DIR", DEFAULT_PROJECT)).expanduser()
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            os.environ.get("DATASET_DIR", project / "notebooks/image_extraction/data")
        )
        / "dataset.parquet",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get("IMAGE_DATASET_DIR", project / "image_dataset_raw")
        ),
    )
    p.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(os.environ.get("MANIFEST_DIR", project / "manifest")),
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument(
        "--limit", type=int, default=None, help="Test with e.g. --limit 20 first"
    )
    p.add_argument("--report-every", type=int, default=10)
    p.add_argument(
        "--failure-check",
        type=int,
        default=5,
        help="Stop if first N completed downloads all fail",
    )
    return p.parse_args()


def image_urls(source_url):
    """Generate large and square iNaturalist photo URLs, in priority order."""
    parsed = urlparse(source_url)
    match = re.search(
        r"/(square|small|medium|large|original)\.(jpe?g|png|webp)$",
        parsed.path,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unexpected iNaturalist photo URL format: {source_url}")
    prefix = parsed.path[: match.start()]
    extension = match.group(2)
    return [
        (size, urlunparse(parsed._replace(path=f"{prefix}/{size}.{extension}")))
        for size in ("large", "square")
    ]


def filename_for(obs, photo, url):
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    # Preserve original square test files; never mistake one for a large image.
    return f"obs_{obs}_photo_{photo}{suffix}"


def task(row, output, retries):
    obs, photo, source_url = map(str, row)
    base = dict(observation_id=obs, photo_id=photo, photo_url=source_url)
    attempts_total = 0
    errors = []
    try:
        candidates = image_urls(source_url)
    except ValueError as exc:
        return dict(
            **base,
            download_url="",
            image_size="",
            local_path="",
            status="failed",
            bytes_downloaded=0,
            attempts=0,
            error=str(exc),
        )

    # Prioritize an already-downloaded large version even when square exists.
    for size, url in candidates:
        path = output / filename_for(obs, photo, url)
        if path.is_file() and path.stat().st_size > 0:
            return dict(
                **base,
                download_url=url,
                image_size=size,
                local_path=str(path),
                status="already_exists",
                bytes_downloaded=0,
                attempts=0,
                error="",
            )
        if size == "large":
            break

    for size, url in candidates:
        path = output / filename_for(obs, photo, url)
        if path.is_file() and path.stat().st_size > 0:
            return dict(
                **base,
                download_url=url,
                image_size=size,
                local_path=str(path),
                status="already_exists",
                bytes_downloaded=0,
                attempts=attempts_total,
                error="; ".join(errors),
            )
        for attempt in range(1, retries + 1):
            attempts_total += 1
            tmp = path.with_name(path.name + ".part")
            try:
                with requests.get(
                    url,
                    timeout=(10, 30),
                    stream=True,
                    headers={"User-Agent": "ResearchImageDownloader/1.0"},
                ) as resp:
                    resp.raise_for_status()
                    ct = resp.headers.get("Content-Type", "").split(";")[0].lower()
                    if not ct.startswith("image/"):
                        raise ValueError(
                            f"Non-image response: {ct or 'unknown content type'}"
                        )
                    nbytes = 0
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                nbytes += len(chunk)
                    if nbytes == 0:
                        raise ValueError("Empty download")
                tmp.replace(path)
                return dict(
                    **base,
                    download_url=url,
                    image_size=size,
                    local_path=str(path),
                    status="downloaded",
                    bytes_downloaded=nbytes,
                    attempts=attempts_total,
                    error="; ".join(errors),
                )
            except requests.HTTPError as exc:
                tmp.unlink(missing_ok=True)
                errors.append(f"{size}: {exc}")
                # Missing/forbidden large variants justify trying square immediately.
                if exc.response is not None and exc.response.status_code in (
                    403,
                    404,
                    410,
                ):
                    break
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
            except (requests.Timeout, requests.ConnectionError) as exc:
                tmp.unlink(missing_ok=True)
                errors.append(f"{size}: {exc}")
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
            except (requests.RequestException, ValueError, OSError) as exc:
                tmp.unlink(missing_ok=True)
                errors.append(f"{size}: {exc}")
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        # If the large image couldn't be downloaded, try square.

    return dict(
        **base,
        download_url="",
        image_size="",
        local_path="",
        status="failed",
        bytes_downloaded=0,
        attempts=attempts_total,
        error="; ".join(errors),
    )


def save_progress(path, counts, total, started, last_error, log_path):
    done = sum(counts.values())
    payload = {
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "processed": done,
        "total": total,
        "remaining": total - done,
        "percent": round(100 * done / total, 2) if total else 100,
        "downloaded": counts["downloaded"],
        "already_exists": counts["already_exists"],
        "failed": counts["failed"],
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "last_error": last_error,
        "results_csv": str(log_path),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return payload


def main():
    a = parse_args()
    if (
        a.workers < 1
        or a.retries < 1
        or a.report_every < 1
        or a.failure_check < 1
        or (a.limit is not None and a.limit < 1)
    ):
        raise SystemExit(
            "Workers, retries, report-every, failure-check and limit must be positive."
        )
    dataset, output, manifest = (
        a.dataset.resolve(),
        a.output.resolve(),
        a.manifest_dir.resolve(),
    )
    if not dataset.is_file():
        raise SystemExit(f"Dataset not found: {dataset}")
    output.mkdir(parents=True, exist_ok=True)
    manifest.mkdir(parents=True, exist_ok=True)
    (manifest / "failed").mkdir(parents=True, exist_ok=True)
    print(
        f"Dataset: {dataset}\nOutput: {output}\nManifest directory: {manifest}",
        flush=True,
    )
    df = pd.read_parquet(dataset, columns=["obs_id", "photo_id", "photo_url"])
    for c in ["obs_id", "photo_id", "photo_url"]:
        if c not in df.columns:
            raise SystemExit(f"Missing dataset column: {c}")
    valid = df.dropna(subset=["obs_id", "photo_id", "photo_url"]).copy()
    valid["photo_url"] = valid["photo_url"].astype(str).str.strip()
    valid = valid[valid["photo_url"].str.match(r"^https?://", case=False, na=False)]
    valid = valid.drop_duplicates(subset=["obs_id", "photo_id"])
    if a.limit:
        valid = valid.head(a.limit)
    total = len(valid)
    if not total:
        raise SystemExit(
            "No valid image URLs found. Check photo_url values in the parquet dataset."
        )
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = manifest / "download_progress.json"
    csv_path = manifest / f"download_results_{stamp}.csv"
    counts = Counter()
    sizes = Counter()
    started = time.monotonic()
    rows = (
        (str(obs), str(photo), str(url))
        for obs, photo, url in valid.itertuples(index=False, name=None)
    )
    print(
        f"Unique valid images: {total:,} | workers: {a.workers} | attempts per image: {a.retries}",
        flush=True,
    )
    print(f"Live status: {progress_path}\nPer-image results: {csv_path}", flush=True)
    last_error = ""
    abort = False
    with open(csv_path, "w", newline="", buffering=1) as logfile:
        writer = csv.DictWriter(logfile, fieldnames=COLUMNS)
        writer.writeheader()
        logfile.flush()
        save_progress(progress_path, counts, total, started, last_error, csv_path)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            pending = {}
            exhausted = False

            def fill():
                nonlocal exhausted
                while not exhausted and len(pending) < a.workers * 2:
                    try:
                        item = next(rows)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[pool.submit(task, item, output, a.retries)] = item

            fill()
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = dict(
                            observation_id=item[0],
                            photo_id=item[1],
                            photo_url=item[2],
                            download_url="",
                            image_size="",
                            local_path="",
                            status="failed",
                            bytes_downloaded=0,
                            attempts=0,
                            error=f"Unexpected: {exc}",
                        )
                    writer.writerow(result)
                    logfile.flush()
                    counts[result["status"]] += 1
                    if result["status"] in ("downloaded", "already_exists"):
                        sizes[result["image_size"]] += 1
                    if result["status"] == "failed":
                        last_error = result["error"]
                    state = save_progress(
                        progress_path, counts, total, started, last_error, csv_path
                    )
                    n = state["processed"]
                    if n % a.report_every == 0 or n == total:
                        print(
                            f"[{state['percent']:6.2f}%] {n:,}/{total:,} | downloaded={counts['downloaded']:,} | existing={counts['already_exists']:,} | failed={counts['failed']:,} | remaining={total - n:,}",
                            flush=True,
                        )
                    if (
                        n >= a.failure_check
                        and counts["downloaded"] + counts["already_exists"] == 0
                    ):
                        abort = True
                if abort:
                    print(
                        f"STOPPING EARLY: first {sum(counts.values())} attempts completed with ZERO available images. Last error: {last_error}",
                        flush=True,
                    )
                    for future in pending:
                        future.cancel()
                    break
                fill()
    print(f"Final progress: {progress_path}\nResults CSV: {csv_path}", flush=True)
    if abort:
        raise SystemExit(2)
    print("\nFINAL SUMMARY", flush=True)
    print(f"Total images: {total}")
    print(f"Downloaded: {counts['downloaded']}")
    print(f"Already existing: {counts['already_exists']}")
    print(f"Failed: {counts['failed']}")
    print(f"Large images available: {sizes['large']}")
    print(f"Square fallbacks available: {sizes['square']}")
    print(f"Processed: {sum(counts.values())}")
    print("DOWNLOAD JOB FINISHED", flush=True)


if __name__ == "__main__":
    main()
