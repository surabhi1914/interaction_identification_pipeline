# Species Identification Pipeline Architecture

## Purpose

Identify animal species in iNaturalist photographs using a two-stage pipeline: **GroundingDINO** locates organisms, and **BioCLIP2** identifies them from image crops. The current scope is **research-grade Animalia** observations.

## Data Flow

```text
iNaturalist observations + photographs
        |
        v
Filter & prepare data (research-grade, Animalia, usable images)
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
# Components and their path
- dataset preparation : [notebook/dataset](notebooks/dataset)
- prompt generation : [notebook/prompt_engineering/generate_prompts.py](notebooks/prompt_engineering/generate_prompts.py)
- object detection : [notebooks/object_detection/run_grounding_dino.py](notebooks/object_detection/run_grounding_dino.py)
- species classification : [notebooks/species_classification](https://github.com/surabhi1914/interaction_identification_pipeline/tree/main/notebooks/species_identification)
- output evaluation : [notebooks/species_identification/test_bioclip.ipynb](https://github.com/surabhi1914/interaction_identification_pipeline/tree/main/notebooks/species_identification/test_bioclip.ipynb)

*Note*:
 > Model-generated columns use the `model_` prefix where applicable.

## HPC Cluster Code Location
`/rs1/researchers/a/amallav/species_classification/ecological_informatics/notebooks
