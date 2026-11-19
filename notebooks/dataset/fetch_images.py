import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# ============================================================
# Configuration
# ============================================================
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", "/rs1/researchers/a/amallav/species_classification/")
).resolve()

DATASET_DIR = Path(
    os.environ.get("DATASET_DIR", PROJECT_DIR / "notebooks/image_extraction/data")
).resolve()
DATASET_PATH = DATASET_DIR / "dataset.parquet"

OUTPUT_DIR = Path(
    os.environ.get("IMAGE_DATASET_DIR", PROJECT_DIR / "image_dataset_raw")
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_DIR = Path(os.environ.get("MANIFEST_DIR", PROJECT_DIR / "manifest")).resolve()
MANIFEST_FAILED_DIR = Path(
    os.environ.get("MANIFEST_FAILED_DIR", PROJECT_DIR / "manifest/failed")
).resolve()

MANIFEST_PATH = MANIFEST_DIR / "download_image_manifest_{timestamp}.parquet"
FAILED_PATH = MANIFEST_FAILED_DIR / "failed_download_image_{timestamp}.parquet"

print("Prject directory", PROJECT_DIR)
print("Output image directory", OUTPUT_DIR, "| exists:", OUTPUT_DIR.exists())
print("Manifest directory", MANIFEST_DIR, "| exists:", MANIFEST_DIR.exists())
print(
    "Manifest Failed directory",
    MANIFEST_FAILED_DIR,
    "| exists:",
    MANIFEST_FAILED_DIR.exists(),
)

MAX_WORKERS = 12
TIMEOUT = 30
MAX_RETRIES = 3


# ============================================================
# Download one image
# ============================================================


def download_image(
    observation_id,
    photo_id,
    photo_url,
    output_dir=OUTPUT_DIR,
    max_retries=MAX_RETRIES,
):
    observation_id = str(observation_id)
    photo_id = str(photo_id)

    suffix = Path(photo_url.split("?")[0]).suffix or ".jpg"

    filename = f"obs_{observation_id}_photo_{photo_id}{suffix}"
    output_path = output_dir / filename

    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "observation_id": observation_id,
            "photo_id": photo_id,
            "photo_url": photo_url,
            "local_path": str(output_path),
            "status": "already_exists",
            "bytes_downloaded": 0,
            "attempts": 0,
            "error": None,
        }

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(photo_url, timeout=TIMEOUT, stream=True) as response:
                response.raise_for_status()

                bytes_downloaded = 0
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

            return {
                "observation_id": observation_id,
                "photo_id": photo_id,
                "photo_url": photo_url,
                "local_path": str(output_path),
                "status": "downloaded",
                "bytes_downloaded": bytes_downloaded,
                "attempts": attempt,
                "error": None,
            }

        except requests.RequestException as error:
            if attempt == max_retries:
                return {
                    "observation_id": observation_id,
                    "photo_id": photo_id,
                    "photo_url": photo_url,
                    "local_path": None,
                    "status": "failed",
                    "bytes_downloaded": 0,
                    "attempts": attempt,
                    "error": str(error),
                }

            time.sleep(2 ** (attempt - 1))

    # Fallback return if retry loop exits without returning
    return {
        "observation_id": observation_id,
        "photo_id": photo_id,
        "photo_url": photo_url,
        "local_path": None,
        "status": "failed",
        "bytes_downloaded": 0,
        "attempts": max_retries,
        "error": "Exhausted retries without explicit error capture",
    }


# ============================================================
# Bulk download
# ============================================================


def download_dataset(
    df,
    observation_id_col="obs_id",
    photo_id_col="photo_id",
    url_col="photo_url",
    output_dir=OUTPUT_DIR,
    max_workers=MAX_WORKERS,
):

    required_columns = {
        observation_id_col,
        photo_id_col,
        url_col,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    total_input_rows = len(df)

    # Count missing URLs before filtering
    missing_url_count = df[url_col].isna().sum()

    download_df = (
        df[
            [
                observation_id_col,
                photo_id_col,
                url_col,
            ]
        ]
        .dropna(subset=[url_col])
        .drop_duplicates(
            subset=[
                observation_id_col,
                photo_id_col,
            ]
        )
        .copy()
    )

    # Rename columns to fixed names so itertuples access is predictable
    download_df = download_df.rename(
        columns={
            observation_id_col: "observation_id",
            photo_id_col: "photo_id",
            url_col: "photo_url",
        }
    )

    total_to_process = len(download_df)

    print("=" * 60)
    print("IMAGE DOWNLOAD STARTED")
    print("=" * 60)

    print(f"Input rows:            {total_input_rows:,}")
    print(f"Missing URLs:          {missing_url_count:,}")
    print(f"Unique images:         {total_to_process:,}")
    print(f"Workers:               {max_workers}")
    print()

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_image,
                row.observation_id,
                row.photo_id,
                row.photo_url,
                output_dir,
            ): (
                row.observation_id,
                row.photo_id,
            )
            for row in download_df.itertuples(index=False)
        }

        for count, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            try:
                result = future.result()

            except Exception as error:
                observation_id, photo_id = futures[future]

                result = {
                    "observation_id": str(observation_id),
                    "photo_id": str(photo_id),
                    "photo_url": None,
                    "local_path": None,
                    "status": "failed",
                    "bytes_downloaded": 0,
                    "attempts": 0,
                    "error": f"Unexpected error: {error}",
                }

            results.append(result)

            if count % 100 == 0 or count == total_to_process:
                print(f"Processed {count:,} / {total_to_process:,}")

    results_df = pd.DataFrame(results)

    return results_df, {
        "total_input_rows": total_input_rows,
        "missing_url_count": missing_url_count,
        "total_to_process": total_to_process,
    }


# ============================================================
# Summary
# ============================================================


def print_download_summary(results_df, input_stats):

    downloaded = (results_df["status"] == "downloaded").sum()

    already_exists = (results_df["status"] == "already_exists").sum()

    failed = (results_df["status"] == "failed").sum()

    total_processed = len(results_df)

    successful_available = downloaded + already_exists

    success_rate = (
        successful_available / total_processed * 100 if total_processed > 0 else 0
    )

    failure_rate = failed / total_processed * 100 if total_processed > 0 else 0

    total_bytes = results_df["bytes_downloaded"].sum()

    total_mb = total_bytes / (1024**2)
    total_gb = total_bytes / (1024**3)

    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print(f"Total input rows:          {input_stats['total_input_rows']:,}")

    print(f"Rows with missing URL:     {input_stats['missing_url_count']:,}")

    print(f"Unique images processed:   {total_processed:,}")

    print()
    print("RESULTS")
    print("-" * 60)

    print(f"Downloaded successfully:   {downloaded:,}")
    print(f"Already existed:           {already_exists:,}")
    print(f"Failed:                    {failed:,}")

    print()
    print(f"Images available locally:  {successful_available:,}")

    print(f"Success rate:              {success_rate:.2f}%")

    print(f"Failure rate:              {failure_rate:.2f}%")

    print()
    print("DATA DOWNLOADED")
    print("-" * 60)

    if total_gb >= 1:
        print(f"New data downloaded:       {total_gb:.2f} GB")
    else:
        print(f"New data downloaded:       {total_mb:.2f} MB")

    if failed > 0:
        print()
        print("FAILURES")
        print("-" * 60)

        print(
            results_df.loc[
                results_df["status"] == "failed",
                ["observation_id", "photo_id", "error"],
            ]
            .head(10)
            .to_string(index=False)
        )

        if failed > 10:
            print(f"\nShowing first 10 of {failed:,} failures.")

    print()
    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    photos = pd.read_parquet(DATASET_PATH)
    # Example:
    # photos = pd.DataFrame(
    #     {
    #         "observation_id": [
    #             10001,
    #             10001,
    #         ],
    #         "photo_id": [
    #             421440,
    #             421441,
    #         ],
    #         "photo_url": [
    #             (
    #                 "https://inaturalist-open-data.s3.amazonaws.com/"
    #                 "photos/421440/large.jpg"
    #             ),
    #             (
    #                 "https://inaturalist-open-data.s3.amazonaws.com/"
    #                 "photos/421441/large.jpg"
    #             ),
    #         ],
    #     }
    # )

    results, input_stats = download_dataset(photos)

    # --------------------------------------------------------
    # Save full manifest
    # --------------------------------------------------------

    results.to_parquet(
        MANIFEST_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save failures separately
    # --------------------------------------------------------

    failed_df = results[results["status"] == "failed"].copy()

    if not failed_df.empty:
        failed_df.to_parquet(
            FAILED_PATH,
            index=False,
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_download_summary(
        results,
        input_stats,
    )

    print()
    print(f"Manifest: {MANIFEST_PATH}")

    if not failed_df.empty:
        print(f"Failed downloads: {FAILED_PATH}")
