import os
import os.path as op
from glob import glob

import numpy as np
import pandas as pd

WORKING_PATH = "..."  # Set your working path here.

CORTICAL_AREAS = [
    "rostralanteriorcingulate",
    "caudalanteriorcingulate",
    "posteriorcingulate",
    "parahippocampal",
    "lateralorbitofrontal",
    "medialorbitofrontal",
    "insula",
    "precuneus",
]
SUBCORTICAL_AREAS = [
    "Thalamus-Proper",
    "Caudate",
    "Accumbens-area",
    "Putamen",
    "Hippocampus",
    "Amygdala",
]
RESULT_TYPES = [
    "tstat",
    "tstat_uncp",
    "tstat_fdrp",
    "tstat_fwep",
    "tstat_mcfwep",
    "cohen",
]
CONTRASTS = ["average", "men > women", "women > men", "men", "women"]
RESULTS_PATH = op.join(WORKING_PATH, "results")
REPORTS_PATH = op.join(WORKING_PATH, "reports")
os.makedirs(REPORTS_PATH, exist_ok=True)


def format_with_asterisk(val):
    """Add asterisk to significant values (p < 0.05) with 3 decimal places"""
    if isinstance(val, float) and val < 0.05:
        return f"{val:.3f}*"
    return f"{val:.3f}"


if __name__ == "__main__":
    for result in glob(f"{RESULTS_PATH}/*"):
        if "volume" in result:
            areas = SUBCORTICAL_AREAS
        else:
            areas = CORTICAL_AREAS

        for c, cont in enumerate(CONTRASTS):
            res_mat = np.array([])
            for res in RESULT_TYPES:
                res_file = op.join(
                    result, f"{op.basename(result)}_dat_{res}_c{c + 1}.csv"
                )
                if not op.exists(res_file):
                    continue
                res_mat = (
                    np.vstack(
                        [res_mat, pd.read_csv(res_file, header=None).values.flatten()]
                    )
                    if res_mat.size
                    else pd.read_csv(res_file, header=None).values.flatten()
                )

            # Create DataFrame and format p-values
            result_df = pd.DataFrame(res_mat, columns=areas)
            for i, res_type in enumerate(RESULT_TYPES):
                if any(pval in res_type for pval in ["fdrp", "fwep", "mcfwep", "uncp"]):
                    result_df.iloc[i] = result_df.iloc[i].apply(format_with_asterisk)

            with open(
                op.join(REPORTS_PATH, f"results_{op.basename(result)}.txt"), "a"
            ) as f:
                f.write(f"Contrast: {cont.capitalize()}\n")
                f.write(
                    pd.concat(
                        [pd.DataFrame(RESULT_TYPES, columns=[""]), result_df],
                        axis=1,
                    ).to_string(index=False)
                )
                f.write("\n\n")
