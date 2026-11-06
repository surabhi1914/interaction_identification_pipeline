# Ecological Informatics: Interaction Identification Pipeline

A machine learning pipeline for identifying species and understanding ecological interactions, with a focus on predator-prey relationships, using images from the [iNaturalist](https://www.inaturalist.org/) *Who Eats Whom* project.

## Pipeline

1. **Data Collection:** Extract observations, images, and taxonomic metadata from iNaturalist.
2. **GroundingDINO:** Detect and localize organisms within images.
3. **BioCLIP2:** Perform zero-shot species identification.
4. **Vision-Language Model (VLM):** Generate descriptions of observable organism behaviors and interactions.
5. **Interaction Classification:** Analyze behavioral descriptions to identify predator-prey relationships.

## Technologies

Python, PyTorch, Hugging Face Transformers, GroundingDINO, BioCLIP2

## Status

Work in progress. The pipeline is being developed and evaluated for automated ecological interaction identification.
