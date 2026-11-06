#importing libraries
import pandas as pd
import numpy as np

import transformers
import torch
import accelerate

from pathlib import Path
import os

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. Use a GPU compute node.")
    
print("GPU:", torch.cuda.get_device_name(0))

# Set configuration paths

PROJECT_DIR = Path(
    os.getenv(
        "PROJECT_DIR",
        "/rs1/researchers/a/amallav/species_classification/ecological_informatics"
    )
)

DATASET_DIR = Path(
    os.getenv(
        "DATASET_DIR",
        str(PROJECT_DIR / "notebooks/prompt_engineering/data")
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(PROJECT_DIR / "notebooks/prompt_engineering/output")
    )
)

DATASET_PATH = DATASET_DIR / "dataset_animalia_split3.parquet"

MODEL_ID = Path(
    os.getenv(
        "MODEL_ID",
        str(PROJECT_DIR / "models/Llama-3.1-8B-Instruct")
    )
)

OUTPUT_PATH_P = OUTPUT_DIR / "dataset_animalia_with_prompts_3.parquet"
OUTPUT_PATH_C = OUTPUT_DIR / "dataset_animalia_with_prompts_readable_3.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert DATASET_PATH.is_file(), f"Dataset not found: {DATASET_PATH}"
assert (MODEL_ID / "config.json").is_file(), f"Model not found: {MODEL_ID}"

print("Dataset:", DATASET_PATH)
print("Model:", MODEL_ID)
print("Output:", OUTPUT_DIR)


# Loading the llama model

pipeline = transformers.pipeline(
    task = "text-generation",
    model=MODEL_ID,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.bfloat16},
    temperature=0.3,
    top_p=0.9,
    truncation=True,
    max_length=500,
)

df = pd.read_parquet(DATASET_PATH)
print(df.head())
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

## Checking for missing columns:
required_columns = ["scientific_name", "common_name"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")
    



SYSTEM_PROMPT = """

You are an expert in biological morphology and visual object descriptions.

Your task is to generate a concise, visually grounded text prompt for GroundingDINO, an open-vocabulary object detection model.

OBJECTIVE:
Describe the target organism using visible physical features that help distinguish it from other objects or organisms in an image.

INSTRUCTIONS:
1. Identify the organism using its scientific name, common name, and taxonomic classification.

2. Begin with a recognizable object category whenever possible, such as bird, butterfly, beetle, spider, fish, mushroom, flowering plant, leaf, or tree.

3. Include the most useful visible identifying characteristics:
   - Color and patterns
   - Body shape and structure
   - Distinctive visible anatomical features
   - Wings, legs, antennae, fins, tails, petals, leaves, or caps
   - Other externally visible morphological traits

4. Prioritize distinctive visual features over general biological descriptions.

5. Use taxonomic context to determine the appropriate object category and morphology, not to produce a taxonomic explanation.

6. Do not invent species-specific features. If distinguishing characteristics are uncertain, use reliable visual features of the broader taxonomic group.

7. Do not include:
   - Habitat or geographic distribution
   - Feeding behavior or ecological relationships
   - Conservation status
   - Internal anatomy or microscopic characteristics
   - Nonvisual biological information
   - Explanations or introductory text

OUTPUT REQUIREMENTS:
- Generate ONE concise detection prompt.
- Use a short noun phrase, not a complete sentence.
- Prefer 5-18 words.
- Put the recognizable object category near the beginning.
- Include up to three informative visual characteristics.
- Avoid repeating the same description in different words.
- Use plain, concrete visual language.
- End with a period.
- Output ONLY the detection prompt.

EXAMPLES:

Input:
scientific_name: Ardea alba
common_name: Great Egret
taxon_class: Aves

Expected Output:
white bird with long slender neck, long legs, and pointed yellow bill.

Input:
scientific_name: Danaus plexippus
common_name: Monarch Butterfly
taxon_class: Insecta

Expected Output:
orange butterfly with black wing veins, dark wing borders, and white spots.

Input:
scientific_name: Vulpes vulpes
common_name: Red Fox
taxon_class: Mammalia

Expected Output:
reddish-brown fox with pointed ears, elongated snout, and bushy tail.

Input:
scientific_name: Rosa rubiginosa
common_name: Sweet Briar
taxon_class: Magnoliopsida

Expected Output:
pink rose flowers with five petals, yellow centers, and green serrated leaves.

Input:
scientific_name: Amanita muscaria
common_name: Fly Agaric
taxon_class: Agaricomycetes

Expected Output:
red mushroom with white cap spots, broad rounded cap, and pale stalk.
"""


def _safe_str(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()


def build_user_prompt(row: pd.Series) -> str:
    """Build the user prompt from a dataframe row."""

    scientific_name = _safe_str(row.get("scientific_name"))
    common_name = _safe_str(row.get("common_name"))

    return f"""Generate a GroundingDINO detection prompt for the target organism.

            Scientific name: {scientific_name}
            Common name: {common_name}
            
            Describe the organism using its most recognizable externally visible features."""


def generate_prompt_for_row(row: pd.Series) -> str:
    """Call the LLM pipeline and return the generated prompt text for this row."""
    user_prompt = build_user_prompt(row)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    outputs = pipeline(messages, max_new_tokens=500, do_sample=False, return_full_text = True)
    generated = outputs[0]["generated_text"][-1]["content"]
    return generated.strip()


# Iterate through the dataframe, generate prompt for each row, add new column

grounding_dino_prompts = []
for idx, row in df.iterrows():
    x = generate_prompt_for_row(row)
    grounding_dino_prompts.append(x)
    print(f"Processed {idx}: prompt as: {x}")

df["grounding_dino_prompt"] = grounding_dino_prompts

print(df[["common_name", "scientific_name", "grounding_dino_prompt"]].head(10))

df.to_parquet(OUTPUT_PATH_P, index=False)
df.to_csv(OUTPUT_PATH_C,index=False)
