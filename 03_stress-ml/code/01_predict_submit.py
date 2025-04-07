import os
import os.path as op
import subprocess

WORKING_DIR = "..."  # Set your WORKING_DIR here.
PY_BIN = "..."  # Set your Python executable here.
LOG_DIR = op.join(WORKING_DIR, "slurm/logs")
os.makedirs(LOG_DIR, exist_ok=True)

SCRIPT = op.join(WORKING_DIR, "code/01_predict.py")

CLUSTER_CONFIG = {
    "time": "24:00:00",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 32,
    "mem": "64G",
}


def submit_slurm_script(
    est,
    hyperopt,
    correct,
    holdout,
):
    slurm_config = """#!/bin/bash
    
#SBATCH --job-name=stress-ml_{est:s}_{correct:s}_{hyperopt:s}_{holdout:s}
#SBATCH --output={slurm_out:s}/{est:s}_{correct:s}_{hyperopt:s}_{holdout:s}.out
#SBATCH --time={time:s}
#SBATCH --nodes={nodes:d}
#SBATCH --ntasks={ntasks:d}
#SBATCH --cpus-per-task={cpus_per_task:d}
#SBATCH --mem={mem:s}

# Load any required modules or environments

{python_exec:s} \\
    {script_file:s} \\
    --est={est:s} \\
    --{hyperopt:s} \\
    --{correct:s} \\
    --{holdout:s}
"""
    slurm_config = slurm_config.format(
        python_exec=PY_BIN,
        script_file=SCRIPT,
        slurm_out=LOG_DIR,
        est=est,
        hyperopt=hyperopt,
        correct=correct,
        holdout=holdout,
        **CLUSTER_CONFIG,
    )
    slurm_file = op.join(
        LOG_DIR,
        f"{est}_{correct}_{hyperopt}_{holdout}.sh",
    )
    with open(slurm_file, "w") as f:
        f.write(slurm_config)
    subprocess.run(f"sbatch {slurm_file}", shell=True)


ESTIMATORS = (
    "ridge",
    # "svr",
    # "histboost",
    "pls",
)

for est in ESTIMATORS:
    for hyperopt in ["hyperopt", "no-hyperopt"]:
        for correct in ["correct", "no-correct"]:
            for holdout in ["holdout", "no-holdout"]:
                submit_slurm_script(est, hyperopt, correct, holdout)
