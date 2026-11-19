# Species Identification Pipeline Architecture

## Purpose

Identify animal species in iNaturalist photographs using a two-stage pipeline: **GroundingDINO** locates organisms, and **BioCLIP2** identifies them from image crops. The current scope is **research-grade Animalia** observations.

## Data Flow

```text
iNaturalist observations + photographs urls
        |
        v
Filter & prepare data (research-grade, Animalia, usable images)
        |
        v
Fetch photographs from amazon s3 bucket for the curated dataset
        |
        v
Llama (prompt generation using the scientific name and common name)
        |
        v
GroundingDINO (detect organism regions; up to 5 candidate boxes/image)
        |
        v
Crop detected regions from the original photograph
        |
        v
BioCLIP2 (zero-shot identification for each crop, Top 5 classification for each crop)
        |
        v
Evaluation framework (ongoing)
Compare predictions with iNaturalist reference taxonomy (evaluation)
```

| Stage | Primary file | Reads / needs | Writes / produces |
|---|---|---|---|
| Fetch observations | [`notebooks/dataset/fetch_observations.py`](notebooks/dataset/fetch_observations.py) | iNaturalist API; project ID 41347, research-grade filter | observation dataset + photograph urls dataset|
| Download photographs | [`notebooks/image_extraction/fetch_images.py`](notebooks/image_extraction/fetch_images.py) | Prepared Parquet with `obs_id`, `photo_id`, `photo_url` | Images under `image_dataset_raw/`|
| Generate detection prompts | [`notebooks/prompt_engineering/generate_prompts.py`](notebooks/prompt_engineering/generate_prompts.py) | Animalia dataset with `scientific_name`, `common_name`; local Llama 3.1 model | `dataset_animalia_with_prompts_3.parquet` and readable CSV in `notebooks/prompt_engineering/output/` |
| Detect organism regions | [`notebooks/object_detection/run_grounding_dino.py`](notebooks/object_detection/run_grounding_dino.py) | Three prompted dataset splits, each carrying `photo_id`, `image_local_path`, `grounding_dino_prompt`; local GroundingDINO weights | Per-batch `detections_*.parquet`, `manifest.parquet`, `run_config.json`, optionally `errors.csv` under `notebooks/object_detection/output/batch_runs/batch_*/` |
| Identify species and evaluate | [`notebooks/species_identification/run_bioclip2.py`](notebooks/species_identification/run_bioclip2.py) | Classification-input Parquet with valid crops; cached BioCLIP2 weights | Per-crop prediction and Top-5 candidate Parquets; consolidated predictions, evaluation CSVs, summary JSON, Excel workbook under `notebooks/species_identification/output/bioclip2`| 
|Output evaluation | [notebooks/species_identification/test_bioclip.ipynb](https://github.com/surabhi1914/interaction_identification_pipeline/tree/main/notebooks/species_identification/test_bioclip.ipynb) | ongoing |


*Note*:
 > Model-generated columns use the `model_` prefix where applicable.

## HPC Cluster Code Location
`/rs1/researchers/a/amallav/species_classification/ecological_informatics/notebooks
