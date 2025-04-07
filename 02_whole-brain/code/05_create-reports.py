import os
import os.path as op
import shutil
from glob import glob

FS_PATH = "..."  # Set your FS_PATH here.
WORKING_DIR = "..."  # Set your WORKING_DIR here.
if __name__ == "__main__":
    for direction in ["pos", "abs"]:
        reports = glob(
            f"{FS_PATH}/*.glmdir/contrast_*/cache.th30.{direction}.sig.cluster.summary"
        )
        out_path = f"{WORKING_DIR}/results/reports" + f".{direction}"
        os.makedirs(out_path, exist_ok=True)
        for report in reports:
            contrast = op.basename(op.dirname(report)).split("_")[-1]
            run_name = op.basename(op.dirname(op.dirname(report))).replace(
                ".glmdir", ""
            )
            out_file = op.join(out_path, f"{run_name}_contrast-{contrast}.txt")
            shutil.copy(report, out_file)
