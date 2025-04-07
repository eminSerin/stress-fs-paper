import os
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

FS_PATH = "..."  # Set your FS_PATH here.
FIG_PATH = "..."  # Set your FIG_PATH here.
os.makedirs(FIG_PATH, exist_ok=True)

RESULTS_MAP = {
    "left_cacc_volume_men": f"{FS_PATH}/whole-brain/lh.volume.10_increase.glmdir/contrast_men/cache.th30.abs.sig.cluster.mgh",
    "left_cacc_area_men": f"{FS_PATH}/whole-brain/lh.area.10_increase.glmdir/contrast_men/cache.th30.abs.sig.cluster.mgh",
}

# Load the fsaverage mesh and sulcal data.
FSAVERAGE = load_fsaverage(mesh="fsaverage7")
FSAVERAGE_DATA = load_fsaverage_data(mesh="fsaverage7", data_type="sulcal")

DB = pd.read_csv("...")  # Set your behavioral data here.

sns.set_theme(font="Helvetica")


def plot_brain_correlations(data, x, y, y_label, out_path=None):
    # Set style and color palette
    sns.set_style(
        "whitegrid",
        {"axes.grid": False, "xtick.bottom": False, "ytick.left": False},
    )
    palette = {"male": "#2E86AB", "female": "#F24236"}

    # Create figure
    plt.figure(figsize=(12, 11), dpi=300)

    # Create jointplot with enhanced styling
    g = sns.jointplot(
        data=data,
        x=x,
        y=y,
        hue="sex",
        kind="scatter",
        height=10,
        ratio=8,
        palette=palette,
        marginal_kws=dict(common_norm=False, fill=True),
        joint_kws=dict(alpha=0.7, s=100),
    )

    # Add regression lines with confidence intervals
    for sex_group in data["sex"].unique():
        subset = data[data["sex"] == sex_group]
        sns.regplot(
            data=subset,
            x=x,
            y=y,
            scatter=False,
            line_kws={"linestyle": "--", "linewidth": 2},
            ci=95,
            ax=g.ax_joint,
            color=palette[sex_group],
        )

    # Add overall regression line
    sns.regplot(
        data=data,
        x=x,
        y=y,
        scatter=False,
        color="black",
        line_kws={"linewidth": 2},
        ci=95,
        ax=g.ax_joint,
    )

    # Customize labels and appearance
    g.ax_joint.set_xlabel(
        "Cortisol increase (nmol/l)".title(), fontsize=20, fontweight="bold"
    ).set_visible(False)  ## To set the visibility off.
    g.ax_joint.set_ylabel(y_label, fontsize=20, fontweight="bold").set_visible(
        False
    )  ## To set the visibility off.
    g.ax_joint.tick_params(labelsize=18)
    # Remove axis values
    # g.ax_joint.set_xticklabels([])
    # g.ax_joint.set_yticklabels([])

    # Customize legend
    g.ax_joint.legend(
        title="Sex",
        title_fontsize=12,
        fontsize=11,
        bbox_to_anchor=(0.95, 0.15),
        frameon=True,
        edgecolor="black",
    ).set_visible(False)

    # Make axes thicker
    g.ax_joint.spines["bottom"].set_linewidth(2)
    g.ax_joint.spines["left"].set_linewidth(2)
    g.ax_joint.tick_params(width=2)

    # Adjust layout and save
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=500, bbox_inches="tight")
        plt.close()


class ConfoundRegressor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self._weights = None
        self._mdl = LinearRegression()
        self._encoder = OneHotEncoder(sparse_output=False, drop="first")

    def _reshape_confound(self, confound):
        if isinstance(confound, pd.DataFrame) or isinstance(confound, pd.Series):
            confound = confound.values
        if confound.ndim == 1:
            confound = confound.reshape(-1, 1)
        return confound

    def _convert_categorical(self, X):
        """Convert categorical columns to one-hot encoding"""
        return self._encoder.fit_transform(self._reshape_confound(X))

    def fit(self, X, y=None, confound=None):
        """Fit confound regressor."""
        if confound is None:
            raise ValueError("Confound matrix must be provided.")
        confound = self._convert_categorical(confound)
        self._mdl.fit(confound, X)
        return self

    def transform(self, X, confound=None):
        """Remove confound effects."""
        if confound is None:
            raise ValueError("Confound matrix must be provided.")
        confound = self._convert_categorical(confound)
        return X - self._mdl.predict(confound)

    def fit_transform(self, X, y=None, confound=None):
        """Fit confound regressor and remove confound effects."""
        self.fit(X, y, confound)
        return self.transform(X, confound)


if __name__ == "__main__":
    roi_map = {}
    for sig_roi, sig_map in RESULTS_MAP.items():
        print(f"Extracting {sig_roi}...")
        mask = np.abs(nib.load(sig_map).get_fdata()) > 1.96
        roi_map[sig_roi] = []
        measure = sig_roi.split("_")[2]
        for subj in tqdm(DB["id"]):
            img = nib.load(
                op.join(FS_PATH, f"{subj}/surf/lh.{measure}.fwhm10.fsaverage.mgh")
            ).get_fdata()
            roi = img[mask].sum()
            roi_map[sig_roi].append(roi)
        roi_map[sig_roi] = np.array(roi_map[sig_roi])

    y_label_map = {
        "left_cacc_volume_men": "Left cACC volume (mm³)",
        "left_cacc_area_men": "Left cACC surface area (mm²)",
    }

    for sig_roi, y_label in y_label_map.items():
        plot_data = pd.concat(
            [DB[["id", "sex", "age", "increase", "cycle"]], pd.DataFrame(roi_map)],
            axis=1,
        )
        plot_data["sex"] = plot_data["sex"].map({0: "male", 1: "female"})
        plot_brain_correlations(
            plot_data,
            "increase",
            sig_roi,
            y_label,
            op.join(FIG_PATH, f"{sig_roi}.pdf"),
        )
