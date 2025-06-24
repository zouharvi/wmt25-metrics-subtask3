"""
TODO: documentation
TODO: Update TER computation script (?)
TODO: Use different tokenizer depending on the language pair

Modified by Vilém Zouhar on 19th June 2025
- switch to TSV format with added post_edit column 
- switch to precomputed baseline0

Modified by Vilém Zouhar on 17th June 2025
- update loading to current references.pe format
- update loading to merge safely, now the script operates a singular data frame
- add subsampling
- add types to functions
- miscellaneous quality of life
"""

from typing import Dict, List
from huggingface_hub import login
import numpy as np
import os
login(token=os.getenv("HF_TOKEN"))

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import TER

COMET_MODEL_PATH = "Unbabel/wmt22-cometkiwi-da"
LANGPAIRS = ["en-zh", "en-cs", "en-is", "en-ja", "en-ru", "en-uk"]


def read_predictions(path) -> Dict[str, pd.DataFrame]:
    # Read the TSV file
    df = pd.read_csv(path, sep='\t', header=0)

    # Ensure 'seg_id' is treated as an integer
    df['segment_id'] = df['segment_id'].astype(int)

    # Create a dictionary to hold language-pair-specific DataFrames
    lang_pair_dfs = {}

    # Group by language pair
    for (source_lang, target_lang), group_df in df.groupby(['source_lang', 'target_lang']):
        lang_pair_dfs[source_lang + "-" + target_lang] = group_df

    return lang_pair_dfs


def merge_dfs_dicts(ref_dfs_dict: Dict[str, pd.DataFrame], pred_dfs_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    for lang in ref_dfs_dict.keys():
        ref_df = ref_dfs_dict[lang]
        pred_df = pred_dfs_dict[lang]

        # check length
        if len(ref_df) != len(pred_df):
            raise ValueError(f"Length mismatch for {lang}: {len(ref_df)} (correct) vs {len(pred_df)} (submitted)")

        # ensure that (segment_id, doc_id, source_lang, target_lang) are the same
        for rowA, rowB in zip(ref_df.itertuples(index=False), pred_df.itertuples(index=False)):
            if (
                rowA.segment_id != rowB.segment_id or
                rowA.doc_id != rowB.doc_id or
                rowA.source_lang != rowB.source_lang or
                rowA.target_lang != rowB.target_lang
            ):
                raise ValueError(f"Mismatch in {lang} for seg_id {rowA.seg_id}: {rowA} vs {rowB}")
        
        # copy post-edit to our dataframe that we know is correct
        ref_df['post_edit'] = pred_df['post_edit'].copy()
    
    return ref_dfs_dict


def load_comet_model(model_name=COMET_MODEL_PATH):
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model


def compute_comet_scores(model, sources, predictions, batch_size=8) -> List[float]:
    data = [{"src": s, "mt": p} for s, p in zip(sources, predictions)]
    comet_scores = model.predict(data, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    return comet_scores.scores


def compute_corpus_ter(references, predictions) -> float:
    ter = TER()
    score = ter.corpus_score(predictions, [references]).score
    return score


def evaluate_predictions(dfs_dict: Dict[str, pd.DataFrame], subsample: int = 100) -> Dict[str, Dict[str, float]]:
    results = {}

    model = load_comet_model(COMET_MODEL_PATH)

    for lang_pair, df in dfs_dict.items():
        # subsample
        df = df.sample(n=subsample, random_state=0) if subsample and len(df) > subsample else df

        # Extract relevant fields
        sources = df['source_segment'].tolist()
        hypothesis = df['hypothesis_segment'].tolist()
        hypothesis_comet = df['overall_comet22'].astype(float).tolist()
        postedit = df['post_edit'].tolist()

        # Compute COMET scores for the predicted corrections
        postedit_comet = compute_comet_scores(model, sources, postedit)

        # Compute delta COMET
        avg_delta_comet = np.average([pred - baseline0 for pred, baseline0 in zip(postedit_comet, hypothesis_comet)])

        # Compute Gain-to-Edit Ratio
        corpus_ter = compute_corpus_ter(postedit, hypothesis)
        # negative infinity is not supported by codabench
        gain_to_edit_ratio = avg_delta_comet / (corpus_ter / 100) if corpus_ter > 0 else -999

        # Save result
        results[lang_pair] = {
            'avg_delta_comet': avg_delta_comet,
            'gain_to_edit_ratio': gain_to_edit_ratio
        }

    return results


def main():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("input_dir")
    args.add_argument("output_dir")
    args = args.parse_args()

    print('Reading references...')
    ref_dfs_dict = read_predictions(args.input_dir + "/baseline0_predictions_with_scores.tsv")

    print('Reading predictions...')
    dfs_dict = read_predictions(args.input_dir + "/res/predictions.tsv")

    print("Merging predictions with references...")
    dfs_dict = merge_dfs_dicts(ref_dfs_dict, dfs_dict)

    print("Run Evaluation...")
    results_dict = evaluate_predictions(dfs_dict)

    print("Raw results:", results_dict)

    print('Write scores to the output file...')
    with open(args.output_dir + '/scores.txt', 'w', encoding='utf-8') as out:
        for lang_pair, results in results_dict.items():
            delta_comet_avg = results.get('avg_delta_comet', 0.0)
            gain_to_edit_ratio = results.get('gain_to_edit_ratio', 0.0)
    
            out.write(f"{lang_pair}_delta_comet_avg: {delta_comet_avg:.4f}\n")
            out.write(f"{lang_pair}_gain_to_edit_ratio: {gain_to_edit_ratio:.4f}\n")

if __name__ == "__main__":
    main()
