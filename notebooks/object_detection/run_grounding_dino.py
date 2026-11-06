#!/usr/bin/env python3
"""Grounding DINO inference on one 8,000-row dataset partition.

Run independently with --batch-id 0, 1, ...; checkpoint every --save-every images.
"""

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DEFAULT_PROJECT = Path(
    "/rs1/researchers/a/amallav/species_classification/ecological_informatics"
)
INPUT_NAMES = [f"dataset_animalia_with_prompts_{i}.parquet" for i in (1, 2, 3)]
EXTRA_COLUMNS = [
    "source_row",
    "img_width",
    "img_height",
    "n_detections",
    "max_score",
    "mean_score",
    "detection_flag",
    "prompt_used",
    "rank",
    "score",
    "group",
    "label",
    "x0",
    "y0",
    "x1",
    "y1",
    "box_area_frac",
    "aspect_ratio",
    "edge_proximity_frac",
    "center_offset_frac",
    "score_rank_gap",
    "is_valid_box",
]


# +------------------+------------------------+----------------------------------------------------+
# | Argument         | Default                | Purpose                                            |
# +------------------+------------------------+----------------------------------------------------+
# | --project-dir    | project directory      | Root location for project files                    |
# | --input-dir      | Automatically derived  | Directory containing the three Parquet datasets    |
# | --model-dir      | Automatically derived  | Directory containing your Hugging Face model       |
# | --output-dir     | Automatically derived  | Where detection results are saved                  |
# | --batch-id       | Required               | Selects which dataset partition to process         |
# | --batch-size     | 8000                   | Number of dataset rows assigned to a job           |
# | --save-every     | 100                    | Number of attempted images between saves           |
# | --top-n          | 5                      | Maximum number of detections retained per image    |
# | --box-threshold  | 0.35                   | Detection box-score filtering threshold            |
# | --text-threshold | 0.25                   | Text-alignment threshold used during postprocessing|
# +------------------+----------------------+------------------------------------------------------+


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT)
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--batch-id", type=int, required=True, help="Zero-based 8000-row partition"
    )
    p.add_argument("--batch-size", type=int, default=8000)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    return p.parse_args()


def atomic_parquet(df, path):
    """
    Safely save detection results without leaving a corrupted checkpoint.
    First, we write the DataFrame to a temporary (.tmp) file. Once writing completes successfully, we rename it to the final .parquet file.
    Reason:If the HPC job crashes during writing, the incomplete temporary file is ignored during resume, and the last completed checkpoint remains safe.
    """
    temp = path.with_name(path.name + ".tmp")
    df.to_parquet(temp, index=False)
    os.replace(temp, path)


# +----------------------+--------------------------------------------+--------------------------------------------------------------+
# | Metric               | Calculation                                | Interpretation                                               |
# +----------------------+--------------------------------------------+--------------------------------------------------------------+
# | box_area_frac        | Box area / image area                      | Fraction of image occupied by the detection                  |
# | aspect_ratio         | Box width / box height                     | Whether the box is wide, tall, or square                     |
# | edge_proximity_frac  | Nearest edge distance / image diagonal     | How close the box is to an image boundary                    |
# | center_offset_frac   | Center-to-center distance / image diagonal | How far the box center is from the image center              |
# | is_valid_box         | Coordinate validity checks                 | Whether the box has positive dimensions and fits inside      |
# |                      |                                            | the image                                                    |
# +----------------------+--------------------------------------------+--------------------------------------------------------------+


def box_metrics(coords, width, height):
    x0, y0, x1, y1 = coords
    w, h = x1 - x0, y1 - y0
    diag = np.hypot(width, height)
    center_dist = np.hypot((x0 + x1) / 2 - width / 2, (y0 + y1) / 2 - height / 2)
    return {
        "box_area_frac": round(float(w * h / (width * height)), 6),
        "aspect_ratio": round(float(w / h), 4) if h > 0 else 0.0,
        "edge_proximity_frac": round(
            float(min(x0, y0, width - x1, height - y1) / diag), 4
        ),
        "center_offset_frac": round(float(center_dist / diag), 4),
        "is_valid_box": bool(
            w > 0 and h > 0 and x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height
        ),
    }


def detect(row, model, processor, device, options, post_params):
    path = Path(str(row["image_local_path"]))
    prompt = str(row["grounding_dino_prompt"]).strip()
    if pd.isna(row["image_local_path"]) or not path.is_file():
        raise FileNotFoundError(str(path))
    if pd.isna(row["grounding_dino_prompt"]) or not prompt:
        raise ValueError("Missing grounding_dino_prompt")

    with Image.open(path) as img:
        image = img.convert("RGB")
    width, height = image.size
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    kwargs = {
        "text_threshold": options.text_threshold,
        "target_sizes": [(height, width)],
    }
    kwargs["box_threshold" if "box_threshold" in post_params else "threshold"] = (
        options.box_threshold
    )
    if "input_ids" in post_params:
        kwargs["input_ids"] = inputs["input_ids"]
    result = processor.post_process_grounded_object_detection(outputs, **kwargs)[0]
    boxes = result["boxes"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()
    labels = result.get("text_labels", result.get("labels", []))
    order = np.argsort(scores)[::-1]
    boxes, scores = boxes[order], scores[order]
    labels = [str(labels[i]) for i in order]
    count = len(scores)
    original = row.to_dict()
    stats = {
        "img_width": width,
        "img_height": height,
        "n_detections": count,
        "max_score": float(scores.max()) if count else None,
        "mean_score": float(scores.mean()) if count else None,
        "detection_flag": count > 0,
        "prompt_used": prompt,
    }
    if count == 0:
        return [{**original, **stats, **dict.fromkeys(EXTRA_COLUMNS[8:], None)}]
    gap = round(float(scores[0] - scores[1]), 4) if count > 1 else None
    items = []
    for i in range(min(options.top_n, count)):
        box = [float(v) for v in boxes[i]]
        x0, y0, x1, y1 = box
        items.append(
            {
                **original,
                **stats,
                "rank": i + 1,
                "score": float(scores[i]),
                "group": path.stem,
                "label": labels[i],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                **box_metrics(box, width, height),
                "score_rank_gap": gap,
            }
        )
    return items


def update_manifest(df, batch_dir):
    """
    Build a row-level processing manifest from saved checkpoint files.

    A row is completed only if its detection output exists on disk.
    Failed rows remain retryable. Pending rows have not been saved.
    """
    manifest = df[["source_row", "photo_id", "image_local_path"]].copy()

    manifest["status"] = "pending"
    manifest["n_detections"] = pd.NA
    manifest["saved_boxes"] = 0
    manifest["checkpoint_file"] = pd.NA
    manifest["error"] = pd.NA

    # Read successfully saved detection checkpoints
    parts = []

    for path in sorted(batch_dir.glob("detections_*.parquet")):
        part = pd.read_parquet(path, columns=["source_row", "n_detections", "rank"])

        part["checkpoint_file"] = path.name
        parts.append(part)

    if parts:
        saved = pd.concat(parts, ignore_index=True)

        # Summarize multiple detection rows into one row per image
        summary = saved.groupby("source_row").agg(
            n_detections=("n_detections", "first"),
            saved_boxes=("rank", "count"),
            checkpoint_file=("checkpoint_file", "first"),
        )

        summary["status"] = np.where(
            summary["n_detections"] > 0, "completed", "no_detection"
        )

        manifest = manifest.set_index("source_row")

        for col in ["status", "n_detections", "saved_boxes", "checkpoint_file"]:
            manifest.loc[manifest.index.intersection(summary.index), col] = summary.loc[
                manifest.index.intersection(summary.index), col
            ].to_numpy()

        manifest = manifest.reset_index()

    # Record the latest failure for rows without saved detections
    error_path = batch_dir / "errors.csv"

    if error_path.exists() and error_path.stat().st_size > 0:
        errors = pd.read_csv(error_path)

        if not errors.empty:
            latest = errors.drop_duplicates(
                subset=["source_row"], keep="last"
            ).set_index("source_row")

            pending = manifest["status"] == "pending"
            failed = pending & manifest["source_row"].isin(latest.index)

            manifest.loc[failed, "status"] = "failed"
            manifest.loc[failed, "error"] = (
                manifest.loc[failed, "source_row"].map(latest["error"]).to_numpy()
            )

    # Publish the updated manifest atomically
    atomic_parquet(manifest, batch_dir / "manifest.parquet")

    return manifest


def main():
    a = parse_args()
    if a.batch_id < 0 or a.batch_size <= 0 or a.save_every <= 0 or a.top_n <= 0:
        raise ValueError("Invalid batch-id, batch-size, save-every, or top-n")
    project = a.project_dir.resolve()
    input_dir = (
        a.input_dir or project / "notebooks" / "object_detection" / "data"
    ).resolve()
    model_dir = (a.model_dir or project / "models" / "grounding-dino-base").resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Grounding DINO config missing: {model_dir}")

    if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
        raise FileNotFoundError(f"No local model weights found in {model_dir}")
    out = (
        a.output_dir
        or project / "notebooks" / "object_detection" / "output" / "batch_runs"
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = [input_dir / name for name in INPUT_NAMES]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"Input dataset not found: {f} (set --input-dir)")

    # Preserve original order; split BEFORE any filtering to retain stable row IDs.
    dataset = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    for c in ("photo_id", "image_local_path", "grounding_dino_prompt"):
        if c not in dataset.columns:
            raise ValueError(f"Missing required column: {c}")
    start, end = a.batch_id * a.batch_size, (a.batch_id + 1) * a.batch_size
    if start >= len(dataset):
        raise ValueError(f"Batch {a.batch_id} outside {len(dataset)}-row dataset")
    df = dataset.iloc[start:end].copy()
    df["source_row"] = np.arange(start, start + len(df), dtype=np.int64)
    batch_dir = out / f"batch_{a.batch_id:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_dir": str(model_dir),
        "inputs": [str(p) for p in files],
        "batch_id": a.batch_id,
        "batch_size": a.batch_size,
        "top_n": a.top_n,
        "box_threshold": a.box_threshold,
        "text_threshold": a.text_threshold,
        "total_rows": len(dataset),
    }
    config_path = batch_dir / "run_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            raise ValueError(
                "Run configuration changed; use a new output directory to avoid mixing results"
            )
    else:
        config_path.write_text(json.dumps(config, indent=2))

    # A successfully processed input row is always represented in one part,
    # including the no-detections placeholder. Errors may be retried on rerun.
    done = set()
    for part in sorted(batch_dir.glob("detections_*.parquet")):
        done.update(
            pd.read_parquet(part, columns=["source_row"])["source_row"]
            .astype(int)
            .tolist()
        )
    todo = df.loc[~df["source_row"].isin(done)]
    # Create or refresh the processing manifest before inference.
    # Existing checkpoints determine which images are already completed.
    manifest = update_manifest(df, batch_dir)

    print("\nManifest status:")
    print(manifest["status"].value_counts())
    print(
        f"Dataset rows: {len(dataset)}, batch {a.batch_id}: {len(df)}, remaining: {len(todo)}",
        flush=True,
    )
    if todo.empty:
        print("Batch already complete.", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not visible. Request a GPU through SLURM before running."
        )
    device = "cuda"
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_dir), local_files_only=True
    ).to(device)
    model.eval()
    post_params = inspect.signature(
        processor.post_process_grounded_object_detection
    ).parameters
    print(
        f"GPU: {torch.cuda.get_device_name(0)}; parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        flush=True,
    )

    existing_parts = sorted(batch_dir.glob("detections_*.parquet"))

    part_index = (
        max(int(p.stem.split("_")[-1]) for p in existing_parts) + 1
        if existing_parts
        else 0
    )
    result_rows, error_rows, attempted = [], [], 0

    def flush():
        nonlocal part_index, attempted

        # Save successful detections
        if result_rows:
            atomic_parquet(
                pd.DataFrame(result_rows),
                batch_dir / f"detections_{part_index:05d}.parquet",
            )
            result_rows.clear()
            part_index += 1

        # Save processing errors
        if error_rows:
            err_file = batch_dir / "errors.csv"

            pd.DataFrame(error_rows).to_csv(
                err_file, mode="a", index=False, header=not err_file.exists()
            )
            error_rows.clear()

        # Rebuild manifest using saved detection and error records
        update_manifest(df, batch_dir)

        attempted = 0

    try:
        for _, row in tqdm(
            todo.iterrows(), total=len(todo), desc=f"Batch {a.batch_id}"
        ):
            try:
                result_rows.extend(
                    detect(row, model, processor, device, a, post_params)
                )
            except Exception as exc:
                error_rows.append(
                    {
                        "source_row": int(row["source_row"]),
                        "photo_id": row["photo_id"],
                        "image_local_path": str(row["image_local_path"]),
                        "error": repr(exc),
                    }
                )
            attempted += 1
            if attempted >= a.save_every:
                flush()
    finally:
        flush()
    # Final processing summary
    manifest = pd.read_parquet(batch_dir / "manifest.parquet")

    print("\n── Final batch status ──")
    print(
        manifest["status"]
        .value_counts()
        .reindex(["completed", "no_detection", "failed", "pending"], fill_value=0)
    )

    print(f"Total input rows: {len(manifest)}")
    print(f"Manifest saved: {batch_dir / 'manifest.parquet'}")
    counts = manifest["status"].value_counts()

    completed = int(counts.get("completed", 0))
    no_detection = int(counts.get("no_detection", 0))
    failed = int(counts.get("failed", 0))
    pending = int(counts.get("pending", 0))

    print("\nFINAL BATCH SUMMARY")
    print(f"Completed detections : {completed}")
    print(f"No detections        : {no_detection}")
    print(f"Failed               : {failed}")
    print(f"Pending              : {pending}")

    if failed == 0 and pending == 0:
        print("Batch completed successfully.")
    else:
        print("Batch finished with incomplete rows. Retry required.")


if __name__ == "__main__":
    main()
