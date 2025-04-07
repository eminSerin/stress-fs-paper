import os.path as op

import pandas as pd

WORK_DIR = "..."  # Set your WORK_DIR here.

if __name__ == "__main__":
    pd.read_csv(f"aseg_stats.txt", sep="\t")
        .rename({"Measure:volume": "id"}, axis=1)
        .to_csv(op.join(WORK_DIR, "data", "fs-measures", "aseg_stats_combined.csv"), index=False)
