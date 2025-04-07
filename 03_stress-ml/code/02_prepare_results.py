import os.path as op
import pickle
from glob import glob

import numpy as np
import pandas as pd

WORKING_DIR = "..."  # Set your WORKING_DIR here.
RESULTS_DIR = op.join(WORKING_DIR, "results")

# Metric is_greater map.
METRIC_MAP = {
    "corr": True,
    "r2": True,
    "explained_variance": True,
    "mae": False,
    "mse": False,
}


def compute_perm_pvalue(actual_score, perm_scores, is_greater=True):
    n_permutations = len(perm_scores)
    if is_greater:
        return (np.sum(perm_scores >= actual_score) + 1) / (n_permutations + 1)
    else:
        return (np.sum(perm_scores <= actual_score) + 1) / (n_permutations + 1)


def make_results_dict(results, metric_map):
    score_dict = {}
    for metric, is_greater in metric_map.items():
        actual_score = results["actual_df"][metric].mean()
        perm_scores = results["perm_df"][metric]
        p_value = compute_perm_pvalue(actual_score, perm_scores, is_greater)
        score_dict[metric] = actual_score
        score_dict[f"{metric}_pval"] = p_value
    return score_dict


if __name__ == "__main__":
    results_df = []
    for results_pkl in glob(op.join(RESULTS_DIR, "*.pkl")):
        with open(results_pkl, "rb") as f:
            results = pickle.load(f)
        configs = op.basename(results_pkl).replace(".pkl", "").split("_")
        model = configs[0]
        corr = True if configs[1] == "correct" else False
        hyp = True if configs[2] == "hyperopt" else False
        hold = True if configs[3] == "holdout" else False
        results_df.append(
            {
                "model": model,
                "confound": corr,
                "hyp": hyp,
                "hold": hold,
                **make_results_dict(results, METRIC_MAP),
            }
        )
    results_df = pd.DataFrame(results_df).sort_values(by="model")
    results_df.to_csv(op.join(RESULTS_DIR, "results.csv"), index=False)
