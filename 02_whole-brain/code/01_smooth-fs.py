import os
import os.path as op
import subprocess
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm

# Create environment with SUBJECTS_DIR set
ENV = os.environ.copy()
ENV["SUBJECTS_DIR"] = "..."  # Set your SUBJECTS_DIR here.
ENV["FREESURFER_HOME"] = "..."  # Set your FREESURFER_HOME here.


N_JOBS = 10
FS_PATH = "..."  # Set your FS_PATH here.


def smooth_fs(subj):
    smooth_cmd = f"recon-all -s {subj} -qcache -fwhm 10"
    subprocess.run(smooth_cmd, shell=True, env=ENV, check=True)


def process_subj(subj_path):
    subj = op.basename(subj_path)
    if not len(glob(op.join(subj_path, "surf", "*.fwhm10.*"))) == 20:
        smooth_fs(subj)
    else:
        print(f"Skipping: {subj}. Preprocessed files already exists.")


if __name__ == "__main__":
    # Process subjects in parallel
    Parallel(n_jobs=N_JOBS)(
        delayed(process_subj)(subj_path)
        for subj_path in tqdm(glob(op.join(FS_PATH, "sub-*")))
    )
