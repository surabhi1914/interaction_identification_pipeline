#!/usr/bin/env python3
# Extracted from run_pipeline.ipynb, after heading: Species Identification
# Run this script with the same Python environment and working directory
# expected by the original notebook.

# --- Notebook cell 88 ---
from pathlib import Path
import pandas as pd
import numpy as np

from PIL import Image
import torch


# --- Notebook cell 89 ---
PROJECT_PATH = Path(
    "/rs1/researchers/a/amallav/species_classification/ecological_informatics/"
).resolve()
GDINO_PATH = (
    PROJECT_PATH
    / "notebooks/interaction/groundingdino/full_run/grounding_dino_boxes_validated.parquet"
).resolve()

print(f"Path: {GDINO_PATH}")
if GDINO_PATH.exists():
    print("Exists ✔️ ")
else:
    print("Path does not exist ❌")


# --- Notebook cell 90 ---
df_boxes = pd.read_parquet(GDINO_PATH)


# # --- Notebook cell 91 ---

# print("Rows:", len(df_boxes))
# print("Columns:")
# print(df_boxes.columns.tolist())


# # --- Notebook cell 92 ---

# display(df_boxes.head())


# # --- Notebook cell 93 ---
# print(df_boxes.dtypes)

# print("\nMissing values:")
# display(
#     df_boxes.isna()
#     .sum()
#     .sort_values(ascending=False)
#     .head(20)
# )


# # --- Notebook cell 95 ---
# import matplotlib.pyplot as plt
# from PIL import Image

# # Select 10 detections
# df_sample = df_boxes.sample(n=10, random_state=42)

# fig, axes = plt.subplots(2, 5, figsize=(16, 7))

# for ax, (_, row) in zip(axes.flat, df_sample.iterrows()):

#     with Image.open(row["local_file_path"]) as img:
#         img = img.convert("RGB")

#         crop = img.crop((
#             int(row["crop_x0"]),
#             int(row["crop_y0"]),
#             int(row["crop_x1"]),
#             int(row["crop_y1"])
#         ))

#         ax.imshow(crop)
#         ax.set_title(
#             f"Photo: {row['photo_id']}\n"
#             f"Box rank: {row['rank']} | Score: {row['score']:.3f}",
#             fontsize=9
#         )
#         ax.axis("off")

# plt.tight_layout()
# plt.show()


# --- Notebook cell 97 ---
import os
from pathlib import Path

import torch
import open_clip
from PIL import Image

CACHE_DIR = Path(
    "/rs1/researchers/a/amallav/species_classification/"
    "ecological_informatics/models/_hf_cache"
)

MODEL_SPEC = "hf-hub:imageomics/bioclip-2"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMER_OFFLINE"]="1"

device = "cpu"

if torch.cuda.is_available():
    try:
        x = torch.ones(4, device="cuda")
        assert x.sum().item() == 4
        device = "cuda"
    except Exception as e:
        print("CUDA test failed:", e)

print("Device:", device)
print("PyTorch:", torch.__version__)
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU capability:", torch.cuda.get_device_capability(0))


# --- Notebook cell 98 ---
SNAPSHOTS_DIR = (
    CACHE_DIR
    / "hub/models--imageomics--bioclip-2/snapshots"
)
snapshots = sorted(SNAPSHOTS_DIR.glob("*"))

for snapshot in snapshots:
    print("\nSnapshot:", snapshot)
    for file in snapshot.iterdir():
        print(file.name, "exists:", file.exists())


# --- Notebook cell 99 ---
# Load BioCLIP2
import open_clip

MODEL_DIR = snapshots[0]

assert (MODEL_DIR / "open_clip_config.json").is_file(), \
    "OpenCLIP config is missing"

model, _, preprocess = open_clip.create_model_and_transforms(
    f"local-dir:{MODEL_DIR}"
)

tokenizer = open_clip.get_tokenizer(
    f"local-dir:{MODEL_DIR}"
)

model = model.to(device).eval()

print("BioCLIP2 loaded successfully")
print("Device:", device)


# # --- Notebook cell 100 ---
# # Test one GroundingDINO crop
# row = df_boxes.iloc[0]

# with Image.open(row["local_file_path"]) as img:
#     img = img.convert("RGB")

#     crop = img.crop((
#         int(row["crop_x0"]),
#         int(row["crop_y0"]),
#         int(row["crop_x1"]),
#         int(row["crop_y1"])
#     ))

# image_tensor = preprocess(crop).unsqueeze(0).to(device)

# with torch.inference_mode():
#     image_features = model.encode_image(image_tensor)
#     image_features = torch.nn.functional.normalize(
#         image_features, dim=-1
#     )

# print("Image embedding shape:", image_features.shape)
# print("Embedding norm:", image_features.norm(dim=-1).item())

# display(crop)


# # --- Notebook cell 101 ---
# # Check BioCLIP2's taxonomy vocabulary
# import json

# config_path = MODEL_DIR / "open_clip_config.json"

# with open(config_path) as f:
#     config = json.load(f)

# print("Configuration keys:", config.keys())
# print("Text configuration:")
# print(config.get("model_cfg", {}).get("text_cfg", {}))


# # --- Notebook cell 102 ---
# import importlib.metadata

# try:
#     print("pybioclip:", importlib.metadata.version("pybioclip"))
# except importlib.metadata.PackageNotFoundError:
#     print("pybioclip not installed")

TOL_DIR = PROJECT_PATH / "models" / "treeoflife_200m"
# --- Notebook cell 103 ---
EMB_PATH = TOL_DIR / "embeddings" / "txt_emb_bioclip-2.npy"
LABEL_PATH = TOL_DIR / "embeddings" / "txt_emb_bioclip-2.json"

print(EMB_PATH.exists(), LABEL_PATH.exists())


# --- Notebook cell 104 ---
# Load the taxonomy embeddings
import numpy as np
import json

# Load the precomputed embeddings without copying them all into RAM
text_embeddings = np.load(EMB_PATH, mmap_mode="r")
text_embeddings = text_embeddings.T

with open(LABEL_PATH, "r") as f:
    taxon_labels = json.load(f)

print("Text embedding shape:", text_embeddings.shape)
print("Number of taxonomic labels:", len(taxon_labels))
print("First label:", taxon_labels[0])

assert text_embeddings.shape[0] == len(taxon_labels)
assert text_embeddings.shape[1] == model.text_projection.shape[-1]


# # --- Notebook cell 106 ---
# import torch
# import torch.nn.functional as F

# TOP_K = 5
# CHUNK_SIZE = 20000

# best_scores = None
# best_indices = None


# # --- Notebook cell 107 ---

# with torch.inference_mode():

#     for start in range(0, len(text_embeddings), CHUNK_SIZE):

#         end = min(start + CHUNK_SIZE, len(text_embeddings))

#         text_batch = torch.as_tensor(
#             text_embeddings[start:end].copy(),
#             dtype=torch.float32,
#             device=device
#         )

#         text_batch = F.normalize(text_batch, dim=-1)

#         similarities = image_features.float() @ text_batch.T

#         scores, indices = similarities.topk(
#             min(TOP_K, end - start), dim=1
#         )

#         indices += start

#         if best_scores is None:
#             best_scores = scores
#             best_indices = indices
#         else:
#             combined_scores = torch.cat(
#                 [best_scores, scores], dim=1
#             )
#             combined_indices = torch.cat(
#                 [best_indices, indices], dim=1
#             )

#             best_scores, positions = combined_scores.topk(
#                 TOP_K, dim=1
#             )

#             best_indices = combined_indices.gather(
#                 1, positions
#             )


# # --- Notebook cell 108 ---

# top_scores = best_scores[0].cpu().tolist()
# top_indices = best_indices[0].cpu().tolist()


# # --- Notebook cell 109 ---

# for rank, (idx, score) in enumerate(
#     zip(top_indices, top_scores), start=1
# ):
#     names, common_name = taxon_labels[idx]

#     species = f"{names[-2]} {names[-1]}"

#     print(
#         f"{rank}. {species} | "
#         f"Similarity: {score:.4f} | "
#         f"Common name: {common_name}"
#     )


# --- Notebook cell 111 ---
#Configuration and checks
from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm.auto import tqdm

# Configuration
BATCH_SIZE = 32
TOP_K = 5
CHUNK_SIZE = 20_000


# --- Notebook cell 112 ---
OUTPUT_DIR = (
    PROJECT_PATH / "notebooks" / "interaction"/ "bioclip2" /"full_run"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Notebook cell 113 ---
assert device == "cuda", "CUDA is required for this run"

model = model.to(device).eval()

# Taxonomy embeddings: expected (867455, 768)
# Handle either original or transposed orientation.
if text_embeddings.shape[0] != len(taxon_labels):
    assert text_embeddings.shape[1] == len(taxon_labels)
    text_embeddings = text_embeddings.T

assert text_embeddings.shape[0] == len(taxon_labels)

# Transfer normalized taxonomy embeddings to GPU once.
# float32 preserves consistency with the single-crop test.
text_gpu = torch.empty(
    text_embeddings.shape,
    dtype=torch.float32,
    device=device
)


# --- Notebook cell 114 ---

with torch.inference_mode():
    for start in tqdm(
        range(0, len(text_embeddings), CHUNK_SIZE),
        desc="Loading taxonomy embeddings"
    ):
        end = min(start + CHUNK_SIZE, len(text_embeddings))

        chunk = np.ascontiguousarray(
            text_embeddings[start:end],
            dtype=np.float32
        )

        text_gpu[start:end] = torch.from_numpy(chunk).to(device)

    text_gpu = F.normalize(text_gpu, dim=1)

print("BioCLIP2 device:", device)
print("Taxonomy embeddings:", text_gpu.shape)
print("GroundingDINO detections:", len(df_boxes))


# --- Notebook cell 115 ---
# Crop preparation function
def prepare_crop(row):
    """
    Load a validated GroundingDINO crop and
    apply BioCLIP2 image preprocessing.
    """

    with Image.open(row["local_file_path"]) as img:
        img = img.convert("RGB")

        crop = img.crop((
            int(row["crop_x0"]),
            int(row["crop_y0"]),
            int(row["crop_x1"]),
            int(row["crop_y1"])
        ))

        return preprocess(crop)


# --- Notebook cell 116 ---
# BioCLIP2 inference function
def predict_bioclip_batch(batch_df):
    """
    Zero-shot BioCLIP2 inference.

    Returns:
        predictions: list of prediction dictionaries
        errors: list of failed crop dictionaries
    """

    images = []
    valid_indices = []
    errors = []

    # Load crops
    for source_idx, row in batch_df.iterrows():
        try:
            image_tensor = prepare_crop(row)

            images.append(image_tensor)
            valid_indices.append(source_idx)

        except Exception as e:
            errors.append({
                "source_index": source_idx,
                "model_status": "error",
                "model_error": str(e)
            })

    if not images:
        return [], errors

    batch_tensor = torch.stack(images).to(device)

    # BioCLIP2 inference
    with torch.inference_mode():

        image_features_batch = model.encode_image(
            batch_tensor
        )

        image_features_batch = F.normalize(
            image_features_batch.float(),
            dim=-1
        )

        similarities = image_features_batch @ text_gpu.T

        scores, indices = torch.topk(
            similarities,
            k=TOP_K,
            dim=1
        )

        scores = scores.cpu().numpy()
        indices = indices.cpu().numpy()

    predictions = []

    # Convert predictions into records
    for i, source_idx in enumerate(valid_indices):

        for pred_rank in range(TOP_K):

            taxon_idx = int(indices[i, pred_rank])
            score = float(scores[i, pred_rank])

            names, common_name = taxon_labels[taxon_idx]

            genus = names[5]
            species_epithet = names[6]

            species_name = (
                f"{genus} {species_epithet}"
                if genus and species_epithet
                else None
            )

            predictions.append({
                "source_index": source_idx,

                "model_prediction_rank": pred_rank + 1,
                "model_taxon_index": taxon_idx,
                "model_score": score,

                "model_kingdom_name": names[0],
                "model_phylum_name": names[1],
                "model_class_name": names[2],
                "model_order_name": names[3],
                "model_family_name": names[4],
                "model_genus_name": genus,
                "model_species_epithet": species_epithet,
                "model_species_name": species_name,
                "model_common_name": common_name,

                "model_status": "success",
                "model_error": None,
            })

    return predictions, errors


# --- Notebook cell 117 ---
# Preserve original dataframe and column names
df_run = df_boxes.reset_index(drop=True).copy()

all_predictions = []
all_errors = []

for start in tqdm(
    range(0, len(df_run), BATCH_SIZE),
    desc="BioCLIP2 full inference"
):

    end = min(start + BATCH_SIZE, len(df_run))

    batch_df = df_run.iloc[start:end]

    predictions, errors = predict_bioclip_batch(
        batch_df
    )

    all_predictions.extend(predictions)
    all_errors.extend(errors)

df_predictions = pd.DataFrame(all_predictions)

df_errors = pd.DataFrame(
    all_errors,
    columns=[
        "source_index",
        "model_status",
        "model_error"
    ]
)

print("Total input detections:", len(df_run))
print(
    "Successfully processed detections:",
    df_predictions["source_index"].nunique()
)
print("Failed detections:", len(df_errors))
print("Prediction rows:", len(df_predictions))


# --- Notebook cell 119 ---
# Preserve all original source columns
df_source = df_run.reset_index(names="source_index")


# --- Notebook cell 120 ---

# Combine predictions with original data
df_per_box = df_source.merge(
    df_predictions,
    on="source_index",
    how="left",
    validate="one_to_many"
)


# --- Notebook cell 121 ---

# Populate failed crop information
if not df_errors.empty:

    error_lookup = df_errors.set_index(
        "source_index"
    )["model_error"]

    failed_mask = df_per_box["model_status"].isna()

    df_per_box.loc[
        failed_mask, "model_status"
    ] = "error"

    df_per_box.loc[
        failed_mask, "model_error"
    ] = df_per_box.loc[
        failed_mask, "source_index"
    ].map(error_lookup)


# --- Notebook cell 122 ---

# Keep source columns first, model columns second
source_cols = df_boxes.columns.tolist()


# --- Notebook cell 123 ---

model_cols = [
    c for c in df_per_box.columns
    if c.startswith("model_")
]


# --- Notebook cell 124 ---

df_per_box = df_per_box[
    source_cols + model_cols
]


# --- Notebook cell 125 ---

# Export
PER_BOX_PATH = OUTPUT_DIR / "bioclip2_per_box.parquet"


# --- Notebook cell 126 ---

df_per_box.to_parquet(
    PER_BOX_PATH,
    index=False
)

print("Saved:", PER_BOX_PATH)
print("Rows:", len(df_per_box))
print("Columns:", len(df_per_box.columns))


# --- Notebook cell 128 ---
# Sort detections and predictions
df_per_image = (
    df_per_box
    .sort_values(
        [
            "photo_id",
            "rank",
            "model_prediction_rank"
        ],
        ascending=[True, True, True],
        na_position="last"
    )
    .drop_duplicates(
        subset=["photo_id"],
        keep="first"
    )
    .copy()
)


# --- Notebook cell 129 ---

# Keep original source columns unchanged.
# Add explicit selected-box metadata.
df_per_image["model_selected_box_rank"] = (
    df_per_image["rank"]
)


# --- Notebook cell 130 ---

df_per_image["model_selected_box_score"] = (
    df_per_image["score"]
)


# --- Notebook cell 131 ---

# Export
PER_IMAGE_PATH = (
    OUTPUT_DIR / "bioclip2_per_image.parquet"
)


# --- Notebook cell 132 ---
df_per_image.to_parquet(
    PER_IMAGE_PATH,
    index=False
)

print("Saved:", PER_IMAGE_PATH)
print("Images:", len(df_per_image))
print("Columns:", len(df_per_image.columns))


# --- Notebook cell 133 ---
# Final validation
# Input coverage
assert df_per_box[
    ["photo_id", "rank"]
].drop_duplicates().shape[0] == df_run[
    ["photo_id", "rank"]
].drop_duplicates().shape[0]


# --- Notebook cell 134 ---

# All original source columns preserved
assert set(df_boxes.columns).issubset(
    df_per_box.columns
)

assert set(df_boxes.columns).issubset(
    df_per_image.columns
)

# No duplicate image-level records
assert df_per_image["photo_id"].is_unique

# Successful detections have exactly TOP_K predictions
successful = df_per_box[
    df_per_box["model_status"] == "success"
]

counts = successful.groupby(
    ["photo_id", "rank"], dropna=False
).size()

assert counts.eq(TOP_K).all()

# All rows accounted for, including failed crops
successful_crops = df_predictions["source_index"].nunique()

assert (
    successful_crops + len(df_errors)
    == len(df_run)
)

assert (
    len(df_per_box)
    == successful_crops * TOP_K + len(df_errors)
)

print("Validation passed")

print("\nSource detections:", len(df_run))
print("Successful detections:", successful_crops)
print("Failed detections:", len(df_errors))
print("Per-box rows:", len(df_per_box))
print("Per-image rows:", len(df_per_image))

print("\nFinal files:")
print(PER_BOX_PATH)
print(PER_IMAGE_PATH)


# --- Notebook cell 135 ---

print(
    df_per_image[
        [
            "photo_id",
            "rank",
            "score",
            "model_prediction_rank",
            "model_species_name",
            "model_score",
            "model_status"
        ]
    ].head(10).to_string(index=False)
)
