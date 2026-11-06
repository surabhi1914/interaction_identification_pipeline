#!/usr/bin/env python3
"""BioCLIP2 classification of validated Grounding DINO crops.

One output row per Grounding DINO crop (Top-1 BioCLIP species and hierarchy).
A separate Top-K table stores all BioCLIP species candidates per crop.
No crops are saved to disk: PIL crops are created in memory.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT = Path('/rs1/researchers/a/amallav/species_classification/ecological_informatics')
RANKS = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species')
KEYS = ('source_row', 'rank')
MODEL_FIELDS = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus',
                'species', 'species_epithet', 'common_name', 'score')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir', type=Path, default=PROJECT)
    p.add_argument('--input', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--model', default='hf-hub:imageomics/bioclip-2')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--save-every', type=int, default=25,
                   help='Number of attempted CROPS between checkpoints')
    p.add_argument('--max-crops', type=int, default=None,
                   help='Pilot only; process the first N crops')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def atomic_parquet(frame, path):
    temp = path.with_name(path.name + '.tmp')
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def normalize_name(value):
    """Compare scientific-name strings without case/whitespace sensitivity.

    Does NOT equate synonyms, taxonomic revisions or infraspecific names.
    """
    if value is None or pd.isna(value):
        return None
    clean = ' '.join(str(value).split()).casefold()
    return clean or None


def compare(gt, predicted):
    a, b = normalize_name(gt), normalize_name(predicted)
    return pd.NA if a is None or b is None else a == b


def make_prediction(row, classifier, top_k):
    """Return (single crop summary, all top-k species rows)."""
    from bioclip import Rank

    # Preserve the exact image orientation used in Grounding DINO.
    with Image.open(str(row['image_local_path'])) as src:
        img = src.convert('RGB')

    if (img.width != int(row['img_width']) or img.height != int(row['img_height'])):
        raise ValueError('Image dimensions changed after detection')

    coords = tuple(int(row[c]) for c in ('crop_x0', 'crop_y0', 'crop_x1', 'crop_y1'))
    x0, y0, x1, y1 = coords
    if not (0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height):
        raise ValueError(f'Invalid crop coordinates: {coords}')

    crop = img.crop(coords)
    preds = classifier.predict(images=[crop], rank=Rank.SPECIES,
                               k=top_k, batch_size=1)
    if not preds:
        raise ValueError('BioCLIP2 returned zero predictions')
    preds = sorted(preds, key=lambda p: float(p['score']), reverse=True)

    result = row.to_dict()  # RETAIN ALL input columns
    best = preds[0]
    result['bioclip_status'] = 'completed'
    result['model_species_epithet'] = best.get('species_epithet')
    result['model_common_name'] = best.get('common_name')
    result['model_score'] = float(best['score'])
    result['model_top2_score'] = float(preds[1]['score']) if len(preds) > 1 else np.nan
    result['model_top1_top2_gap'] = (float(preds[0]['score']) - float(preds[1]['score'])
                                     if len(preds) > 1 else np.nan)
    result['model_topk_json'] = json.dumps(
        [{**{f: p.get(f) for f in MODEL_FIELDS if f != 'score'},
          'score': float(p['score'])} for p in preds],
        ensure_ascii=False)

    for rank in RANKS:
        predicted = best.get(rank)
        result[f'model_{rank}_name'] = predicted
        result[f'{rank}_match'] = compare(row.get(f'{rank}_name'), predicted)
    result['model_species_topk_match'] = (
        pd.NA if normalize_name(row.get('species_name')) is None else
        any(normalize_name(row.get('species_name')) == normalize_name(p.get('species'))
            for p in preds))
    result['model_species_gt_rank'] = next(
        (i for i, p in enumerate(preds, 1)
         if normalize_name(row.get('species_name')) is not None and
         normalize_name(row.get('species_name')) == normalize_name(p.get('species'))),
        pd.NA)

    detailed = []
    for i, p in enumerate(preds, start=1):
        item = {'source_row': int(row['source_row']), 'rank': int(row['rank']),
                'model_prediction_rank': i, 'model_score': float(p['score'])}
        for field in MODEL_FIELDS:
            if field != 'score':
                name = f'model_{field}_name' if field in RANKS else f'model_{field}'
                item[name] = p.get(field)
        for rank in RANKS:
            item[f'{rank}_match'] = compare(row.get(f'{rank}_name'), p.get(rank))
        detailed.append(item)
    return result, detailed


# ------------------------- Evaluation helpers -------------------------
# Evaluation unit definitions:
#   image = unique source_row
#   crop  = unique (source_row, rank), where rank is Grounding DINO rank
#   candidate = unique (source_row, rank, model_prediction_rank)
# Observed accuracy uses *one predetermined crop per image*: DINO rank 1.

DISTANCE_LABELS = {
    0: 'Same species',
    1: 'Same genus, different species',
    2: 'Same family, different genus',
    3: 'Same order, different family',
    4: 'Same class, different order',
    5: 'Same phylum, different class',
    6: 'Same kingdom, different phylum',
    7: 'No shared known level',
}


def taxonomic_relationship(row):
    """Compare hierarchies from kingdom down, only as far as data support.

    Returns (deepest shared rank, integer distance, relationship, error category).
    Missing data yields an unresolved distance if it blocks identifying the
    first disagreement. A matching species requires all higher ranks to match.
    Distances: 0=same species, 1=same genus, ..., 6=same kingdom,
    7=different kingdoms. Only a *known* mismatch fixes distance.
    """
    deepest = None
    for level, rank in enumerate(RANKS):
        gt = normalize_name(row.get(f'{rank}_name'))
        pred = normalize_name(row.get(f'model_{rank}_name'))
        if gt is None or pred is None:
            return deepest, pd.NA, 'Unresolved taxonomy', 'Unresolved taxonomy'
        if gt != pred:
            distance = 7 - level
            label = DISTANCE_LABELS[distance]
            if distance in (1, 2):
                category = 'Near-miss error'
            elif distance in (3, 4):
                category = 'Moderate mismatch'
            else:
                category = 'Distant misclassification'
            return deepest, distance, label, category
        deepest = rank
    return deepest, 0, DISTANCE_LABELS[0], 'Correct species'


def add_taxonomy_diagnostics(frame):
    """Append taxonomic distance and relationship for each row."""
    if frame.empty:
        return frame.copy()
    out_frame = frame.copy()
    diagnostics = out_frame.apply(taxonomic_relationship, axis=1,
                                  result_type='expand')
    diagnostics.columns = ['deepest_shared_rank', 'taxonomic_distance',
                           'taxonomic_relationship', 'error_category']
    for name in diagnostics:
        out_frame[name] = diagnostics[name].to_numpy()
    return out_frame


def boolean_accuracy(series):
    """Null comparisons are *excluded*, never silently counted as incorrect."""
    known = series.dropna().astype(bool)
    count = len(known)
    correct = int(known.sum())
    return {'evaluated': count, 'correct': correct,
            'incorrect': count - correct,
            'accuracy': float(correct / count) if count else np.nan,
            'error_rate': float((count - correct) / count) if count else np.nan}


def save_evaluations(main, top, expected_input, out):
    """Write reproducible crop-, image-, and candidate-level evaluations."""
    if main.empty:
        raise ValueError('No classified crops available for evaluation')
    if top.empty:
        raise ValueError('Top-K data is missing; cannot evaluate prediction ranks')

    # Attach true labels to every BioCLIP candidate, keeping candidate rows.
    gt_columns = [*KEYS] + [f'{rank}_name' for rank in RANKS]
    top_eval = top.merge(main[gt_columns], on=list(KEYS), how='left',
                         validate='many_to_one', indicator=True)
    if not top_eval['_merge'].eq('both').all():
        raise ValueError('Top-K contains a candidate without a matching crop')
    top_eval = top_eval.drop(columns='_merge')
    for rank in RANKS:
        top_eval[f'{rank}_match'] = [
            compare(a, b) for a, b in zip(top_eval[f'{rank}_name'],
                                          top_eval[f'model_{rank}_name'])
        ]
    top_eval = add_taxonomy_diagnostics(top_eval)
    atomic_parquet(top_eval, out / 'bioclip2_topk_evaluated.parquet')

    # Keep image-level reported accuracy strictly on the chosen DINO rank-1 crop.
    rank1 = main.loc[main['rank'] == 1].copy()
    if rank1['source_row'].duplicated().any():
        raise ValueError('Multiple rank-1 crops for an image')

    populations = {
        'all_crops': main,
        'dino_rank1_images': rank1,
    }
    rows = []
    missing_rows = []
    for population, subset in populations.items():
        for rank in RANKS:
            gt = subset[f'{rank}_name'].map(normalize_name)
            pred = subset[f'model_{rank}_name'].map(normalize_name)
            missing_rows.append({
                'population': population,
                'rank': rank,
                'total_records': len(subset),
                'missing_ground_truth': int(gt.isna().sum()),
                'missing_prediction': int(pred.isna().sum()),
                'missing_either': int((gt.isna() | pred.isna()).sum()),
                'missing_either_pct': float(100 * (gt.isna() | pred.isna()).mean())
                    if len(subset) else np.nan,
            })
            stats = boolean_accuracy(subset[f'{rank}_match'])
            rows.append({'population': population, 'rank': rank, **stats,
                         'accuracy_pct': 100 * stats['accuracy'],
                         'error_rate_pct': 100 * stats['error_rate']})
    pd.DataFrame(rows).to_csv(out / 'rank_accuracy.csv', index=False)
    pd.DataFrame(missing_rows).to_csv(out / 'rank_missingness.csv', index=False)

    # Phylum-level error breakdown: support and accuracy by true phylum.
    phylum = rank1.copy()
    phylum = phylum.loc[phylum['phylum_match'].notna()].copy()
    if not phylum.empty:
        phylum['phylum_match'] = phylum['phylum_match'].astype(bool)
        per_phylum = (phylum.groupby('phylum_name', dropna=False)
                      .agg(images_evaluated=('source_row', 'size'),
                           correct=('phylum_match', 'sum'))
                      .reset_index())
        per_phylum['incorrect'] = per_phylum['images_evaluated'] - per_phylum['correct']
        per_phylum['accuracy'] = per_phylum['correct'] / per_phylum['images_evaluated']
        per_phylum.to_csv(out / 'phylum_per_group.csv', index=False)
        (phylum.groupby(['phylum_name', 'model_phylum_name'], dropna=False)
         .size().reset_index(name='images')
         .sort_values('images', ascending=False)
         .to_csv(out / 'phylum_confusions.csv', index=False))

    # Hierarchical accuracy: all ranks from kingdom through this rank match.
    hierarchical_rows = []
    for population, subset in populations.items():
        for i, rank in enumerate(RANKS):
            flags = subset[[f'{r}_match' for r in RANKS[:i + 1]]]
            comparable = flags.notna().all(axis=1)
            n = int(comparable.sum())
            correct = int(flags.loc[comparable].astype(bool).all(axis=1).sum())
            hierarchical_rows.append({
                'population': population, 'through_rank': rank,
                'evaluated': n, 'correct': correct,
                'hierarchical_accuracy': correct / n if n else np.nan,
            })
    pd.DataFrame(hierarchical_rows).to_csv(out / 'hierarchical_accuracy.csv', index=False)

    # Evaluated Top-1 image-level summary. Failed crops and no-detection
    # images are NOT part of this observed-accuracy denominator.
    species_stats = boolean_accuracy(rank1['species_match'])
    image_summary = pd.DataFrame([{
        'total_images_with_rank1_prediction': int(rank1['source_row'].nunique()),
        'total_images_evaluated': species_stats['evaluated'],
        'correctly_identified_true': species_stats['correct'],
        'falsely_identified_false': species_stats['incorrect'],
        'observed_accuracy': species_stats['accuracy'],
        'observed_error_rate': species_stats['error_rate'],
        'excluded_missing_species_labels': int(rank1['species_match'].isna().sum()),
        'input_crops': int(len(expected_input)),
        'classified_crops': int(len(main)),
        'unclassified_crops': int(len(expected_input) - len(main)),
    }])
    image_summary.to_csv(out / 'observed_image_accuracy.csv', index=False)

    # Relationship distribution for observed rank-1 image predictions.
    relation_summary = (
        rank1.groupby(['taxonomic_relationship', 'error_category'],
                      dropna=False).size().reset_index(name='images')
    )
    relation_summary['fraction_of_rank1_images'] = (
        relation_summary['images'] / len(rank1)
    )
    relation_summary.to_csv(out / 'taxonomic_relationships.csv', index=False)

    # Numeric distances are conditional on sufficient taxonomic labels.
    distance_rows = []
    for prediction_rank, subset in top_eval.groupby('model_prediction_rank'):
        distances = pd.to_numeric(subset['taxonomic_distance'], errors='coerce').dropna()
        distance_rows.append({
            'prediction_rank': int(prediction_rank),
            'candidate_rows': len(subset),
            'distance_evaluated': len(distances),
            'distance_unresolved': len(subset) - len(distances),
            'mean_taxonomic_distance': float(distances.mean()) if len(distances) else np.nan,
            'median_taxonomic_distance': float(distances.median()) if len(distances) else np.nan,
        })
    pd.DataFrame(distance_rows).sort_values('prediction_rank').to_csv(
        out / 'distance_by_prediction_rank.csv', index=False)

    all_distances = pd.to_numeric(top_eval['taxonomic_distance'], errors='coerce').dropna()
    confidence = pd.DataFrame([{
        'crops_with_top1': len(main),
        'crops_with_top2': int(main['model_top2_score'].notna().sum()),
        'mean_top1_score': float(main['model_score'].mean()),
        'mean_top2_score': float(main['model_top2_score'].mean()),
        'mean_top1_top2_gap': float(main['model_top1_top2_gap'].mean()),
        'median_top1_top2_gap': float(main['model_top1_top2_gap'].median()),
    }])
    confidence.to_csv(out / 'confidence_summary.csv', index=False)

    # Top-k retrieval accuracy and MRR: one vote per crop, correct species
    # counted at its earliest BioCLIP candidate rank.
    known = main.loc[main['model_species_topk_match'].notna()]
    topk_flags = known['model_species_topk_match'].astype(bool)
    gt_ranks = pd.to_numeric(known['model_species_gt_rank'], errors='coerce')
    retrieval = pd.DataFrame([{
        'evaluated_crops': len(known),
        'topk_species_accuracy': float(topk_flags.mean()) if len(known) else np.nan,
        'mrr_at_k': float((1.0 / gt_ranks).fillna(0).mean()) if len(known) else np.nan,
    }])
    retrieval.to_csv(out / 'topk_retrieval.csv', index=False)

    # Oracle: uses ground truth to check *any crop*; never report as deployed accuracy.
    oracle_rows = []
    for rank in RANKS:
        valid = main.loc[main[f'{rank}_match'].notna()]
        if valid.empty:
            continue
        by_image = valid.groupby('source_row')[f'{rank}_match'].agg(
            lambda s: bool(s.astype(bool).any()))
        oracle_rows.append({'rank': rank, 'images_evaluated': len(by_image),
                            'oracle_any_crop_top1_accuracy': float(by_image.mean())})
    pd.DataFrame(oracle_rows).to_csv(out / 'oracle_any_crop_top1.csv', index=False)

    summary = {
        'observed_image_accuracy': image_summary.iloc[0].to_dict(),
        'overall_topk_candidate_mean_taxonomic_distance': (
            float(all_distances.mean()) if len(all_distances) else None),
        'overall_topk_candidate_distances_evaluated': len(all_distances),
        'overall_topk_candidate_distances_unresolved': len(top_eval) - len(all_distances),
        'distance_scale': DISTANCE_LABELS,
    }
    (out / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2,
                                        default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    print('Image-level species accuracy:', image_summary.iloc[0].to_dict(), flush=True)



def write_combined_workbook(main, top_eval, out):
    """One Excel workbook containing all predictions and every evaluation table.

    Grain:
      Predictions = one row per Grounding DINO crop.
      Top5_Candidates = one row per BioCLIP2 species candidate per crop.
      Other tabs = summary metrics derived from the same completed checkpoints.

    Uses xlsxwriter to create Excel. This is a reporting export; Parquet files
    remain the authoritative lossless outputs for further scientific analysis.
    """
    summary_path = out / 'evaluation_summary.json'
    summary = json.loads(summary_path.read_text())
    observed = summary['observed_image_accuracy']
    overview_rows = [
        ('Total images evaluated', observed['total_images_evaluated'],
         'Unique image source rows with a DINO rank-1 crop and comparable species labels'),
        ('Correctly identified (True)', observed['correctly_identified_true'],
         'BioCLIP2 Top-1 species matches observer label on DINO rank-1 crop'),
        ('Falsely identified (False)', observed['falsely_identified_false'],
         'BioCLIP2 Top-1 species differs from comparable observer label'),
        ('Observed accuracy', observed['observed_accuracy'],
         'Correct / evaluated; excludes missing reference/predicted species'),
        ('Observed error rate', observed['observed_error_rate'],
         'Incorrect / evaluated; excludes missing reference/predicted species'),
        ('Images with Rank-1 crop prediction', observed['total_images_with_rank1_prediction'],
         'May exceed evaluated images due to missing taxonomy'),
        ('Crops submitted', observed['input_crops'], 'All validated crop rows in this run'),
        ('Crops classified', observed['classified_crops'], 'Successful crop-level predictions'),
        ('Crops unclassified', observed['unclassified_crops'], 'Crop attempts without saved predictions'),
        ('Mean Top-K taxonomic distance',
         summary['overall_topk_candidate_mean_taxonomic_distance'],
         'Candidate-row weighted, evaluable candidates only; ordinal rank scale'),
        ('Top-K distances evaluated',
         summary['overall_topk_candidate_distances_evaluated'],
         'Candidate rows with sufficiently complete reference and model hierarchy'),
        ('Top-K distances unresolved',
         summary['overall_topk_candidate_distances_unresolved'],
         'Candidate rows without enough known taxonomy for exact distance'),
    ]
    overview = pd.DataFrame(overview_rows, columns=['metric', 'value', 'interpretation'])
    tables = [
        ('Overview', overview),
        ('Predictions', main),
        ('Top5_Candidates', top_eval),
    ]
    reports = [
        ('Image_Accuracy', 'observed_image_accuracy.csv'),
        ('Rank_Accuracy', 'rank_accuracy.csv'),
        ('Rank_Nulls', 'rank_missingness.csv'),
        ('Hierarchical', 'hierarchical_accuracy.csv'),
        ('Phylum_By_Group', 'phylum_per_group.csv'),
        ('Phylum_Confusions', 'phylum_confusions.csv'),
        ('Relationships', 'taxonomic_relationships.csv'),
        ('Distance_By_Rank', 'distance_by_prediction_rank.csv'),
        ('Confidence', 'confidence_summary.csv'),
        ('TopK_Retrieval', 'topk_retrieval.csv'),
        ('Any_Crop_Oracle', 'oracle_any_crop_top1.csv'),
    ]
    for sheet_name, filename in reports:
        file_path = out / filename
        if file_path.exists():
            tables.append((sheet_name, pd.read_csv(file_path)))

    report_path = out / 'all_results_and_metrics.xlsx'
    temp_path = out / 'all_results_and_metrics.tmp.xlsx'
    try:
        with pd.ExcelWriter(temp_path, engine='xlsxwriter',
                            engine_kwargs={'options': {'strings_to_urls': False,
                                                        'nan_inf_to_errors': True}}) as writer:
            for sheet_name, table in tables:
                if len(table) + 1 > 1_048_576 or len(table.columns) > 16_384:
                    raise ValueError(f'{sheet_name} exceeds Excel worksheet limits')
                export = table.copy()
                # Excel has no native pd.NA; represent missing data as blank cells.
                export = export.astype(object).where(pd.notna(export), None)
                export.to_excel(writer, sheet_name=sheet_name, index=False)
                sheet = writer.sheets[sheet_name]
                sheet.freeze_panes(1, 0)
                sheet.autofilter(0, 0, len(export), max(0, len(export.columns)-1))
                sheet.set_row(0, 26, writer.book.add_format({
                    'bold': True, 'font_color': 'white', 'bg_color': '#17365D',
                    'text_wrap': True, 'valign': 'vcenter'}))
                sheet.set_column(0, max(0, len(export.columns)-1), 18)
                for index, column in enumerate(export.columns):
                    if 'prompt' in column or 'error' in column:
                        sheet.set_column(index, index, 40)
                    elif column in ('image_local_path', 'photo_url', 'obs_url'):
                        sheet.set_column(index, index, 34)
        os.replace(temp_path, report_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    print(f'Combined Excel report: {report_path}', flush=True)

def compile_outputs(out, expected_input):
    """Consolidate checkpoint outputs and compute descriptive/evaluation tables."""
    files = sorted(out.glob('predictions_*.parquet'))
    if not files:
        print('No prediction checkpoints to consolidate.')
        return
    main = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if main.duplicated(list(KEYS)).any():
        raise ValueError('Duplicate crop keys in prediction parts')

    top_files = sorted(out.glob('topk_*.parquet'))
    top = (pd.concat([pd.read_parquet(p) for p in top_files], ignore_index=True)
           if top_files else pd.DataFrame())
    if not top.empty:
        if top.duplicated([*KEYS, 'model_prediction_rank']).any():
            raise ValueError('Duplicate Top-K candidate keys')
        atomic_parquet(top, out / 'bioclip2_topk_all.parquet')

    main = add_taxonomy_diagnostics(main)
    atomic_parquet(main, out / 'bioclip2_predictions_all.parquet')
    save_evaluations(main, top, expected_input, out)
    # Build one human-readable file from the same checkpointed predictions
    # and metric CSVs. Individual Parquet/CSV files are kept unchanged.
    top_eval = pd.read_parquet(out / 'bioclip2_topk_evaluated.parquet')
    write_combined_workbook(main, top_eval, out)


def main():
    a = arguments()
    if a.top_k < 1 or a.save_every < 1 or (a.max_crops is not None and a.max_crops < 1):
        raise ValueError('top-k, save-every and max-crops must be positive')
    project = a.project_dir.resolve()
    input_path = (a.input or project / 'notebooks/object_detection/output/analysis/'
                  'bioclip2_input_detections.parquet').resolve()
    out = (a.output_dir or project / 'notebooks/species_identification/output/bioclip2').resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(input_path)
    required = set(KEYS) | {'image_local_path', 'img_width', 'img_height',
                             'crop_x0', 'crop_y0', 'crop_x1', 'crop_y1', 'crop_valid'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.duplicated(list(KEYS)).any():
        raise ValueError('Input contains duplicate (source_row, rank) pairs')
    if not df['crop_valid'].fillna(False).astype(bool).all():
        raise ValueError('Input has invalid crops; fix these before identification')
    if a.max_crops is not None:
        df = df.head(a.max_crops).copy()

    # Resume guard: do not mix incompatible settings / input selection.
    config = {'input': str(input_path), 'model': a.model, 'top_k': a.top_k,
              'max_crops': a.max_crops, 'total_crops': len(df),
              'source_row_rank_keys_hash': __import__('hashlib').sha256(
                  pd.util.hash_pandas_object(df[list(KEYS)], index=False).values.tobytes()
              ).hexdigest()}
    cfg = out / 'run_config.json'
    if cfg.exists():
        if json.loads(cfg.read_text()) != config:
            raise ValueError('Different run configuration: choose another output-dir')
    else:
        cfg.write_text(json.dumps(config, indent=2))

    parts = sorted(out.glob('predictions_*.parquet'))
    orphan_topk = {p.name.replace('topk_', 'predictions_', 1) for p in out.glob('topk_*.parquet')} - {p.name for p in parts}
    if orphan_topk:
        raise RuntimeError(f'Orphan Top-K checkpoint(s): {sorted(orphan_topk)}')
    # Paired checkpoint files must both exist before resuming.
    for part in parts:
        partner = out / part.name.replace('predictions_', 'topk_', 1)
        if not partner.exists():
            raise RuntimeError(f'Missing Top-K checkpoint paired with {part.name}: {partner}')
    done = set()
    for p in parts:
        pkeys = pd.read_parquet(p, columns=list(KEYS))
        done.update(map(tuple, pkeys[list(KEYS)].itertuples(index=False, name=None)))
    todo = df.loc[~pd.MultiIndex.from_frame(df[list(KEYS)]).isin(done)]
    print(f'Input crops={len(df)} | completed={len(done)} | remaining={len(todo)}', flush=True)
    if todo.empty:
        compile_outputs(out, df)
        return
    if not a.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; submit through SLURM or use --cpu for testing')

    # Do not access Hugging Face online from the GPU compute node.
    # Use the shared Hugging Face cache populated on the login node.
    # Set the cache location before starting Python via the SLURM script.
    hf_home = os.environ.get("HF_HOME")
    
    if not hf_home or not Path(hf_home).is_dir():
        raise RuntimeError(
            "HF_HOME is not configured or the cache directory does not exist. "
            "Set HF_HOME in the SLURM script."
        )
    
    print(f"Using Hugging Face cache: {hf_home}", flush=True)
    
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from bioclip import TreeOfLifeClassifier
    classifier = TreeOfLifeClassifier(
        device='cpu' if a.cpu else 'cuda', model_str=a.model)

    existing = [int(p.stem.rsplit('_', 1)[-1]) for p in parts]
    part_no = max(existing, default=-1) + 1
    successful, topk, errors = [], [], []
    attempted = 0

    def flush():
        nonlocal part_no, attempted
        if successful:
            atomic_parquet(pd.DataFrame(successful), out / f'predictions_{part_no:05d}.parquet')
            atomic_parquet(pd.DataFrame(topk), out / f'topk_{part_no:05d}.parquet')
            successful.clear()
            topk.clear()
            part_no += 1
        if errors:
            err = out / 'errors.csv'
            pd.DataFrame(errors).to_csv(err, mode='a', header=not err.exists(), index=False)
            errors.clear()
        attempted = 0

    try:
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc='BioCLIP2'):
            try:
                record, candidates = make_prediction(row, classifier, a.top_k)
                successful.append(record)
                topk.extend(candidates)
            except Exception as exc:
                errors.append({'source_row': row['source_row'], 'rank': row['rank'],
                               'photo_id': row.get('photo_id'), 'error': repr(exc)})
            attempted += 1
            if attempted >= a.save_every:
                flush()
    finally:
        flush()
    compile_outputs(out, df)
    print(f'BioCLIP2 outputs saved under {out}', flush=True)


if __name__ == '__main__':
    main()
