import os.path as op

import pandas as pd

WORK_DIR = "..."  # Set your WORK_DIR here.
MEASURES = ["area", "thickness", "volume", "meancurv", "gauscurv", "foldind", "curvind"]

if __name__ == "__main__":
    db_df = pd.DataFrame()
    for m, meas in enumerate(MEASURES):
        lh_df = (
            pd.read_csv(f"aparc_lh_{meas}.txt", sep="\t")
            .rename(columns={f"lh.aparc.{meas}": "id"})
            .drop(columns=["BrainSegVolNotVent", "eTIV"])
        )
        rh_df = (
            pd.read_csv(f"aparc_rh_{meas}.txt", sep="\t")
            .rename(columns={f"rh.aparc.{meas}": "id"})
            .drop(columns=["BrainSegVolNotVent", "eTIV"])
        )
        meas_df = lh_df.merge(rh_df, on="id")
    db_df = pd.concat([db_df, meas_df])
    db_df.to_csv(
        op.join(WORK_DIR, "data", "aparcstats2table", "aparcstats2table_combined.csv"),
        index=False,
    )
