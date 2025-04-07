import os
import os.path as op
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


def plot_shap_beeswarm(shap_values, show_labels=True, out_file=None):
    # Set a high-quality, publication-ready style
    plt.rcParams.update(
        {
            "font.size": 32,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "figure.figsize": (15, 9),
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    # Create a figure explicitly to ensure high resolution output
    _ = plt.figure()
    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        color=plt.get_cmap("coolwarm"),
        # log_scale=True,
        show=False,  # Delay display to allow further customization
        # color_bar=False,
    )

    if not show_labels:
        # Remove labels and tick values from both x and y axes
        ax = plt.gca()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])  # Remove x-axis tick labels but keep ticks
        ax.set_yticks([])

    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file, dpi=1000, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_shap_cv(shap_cv_df, n_features=20, show_labels=True, out_file=None):
    # Set style parameters
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "figure.figsize": (12, 9),
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    # Sort features by mean absolute SHAP value and select top n_featuress
    mean_abs_shap = shap_cv_df.abs().mean().sort_values(ascending=False)
    shap_cv_df = shap_cv_df[mean_abs_shap.index]
    plot_df = shap_cv_df.iloc[:, :n_features].reset_index(drop=True)
    plot_df = plot_df.melt(var_name="Brain Measure", value_name="SHAP Value")

    _ = plt.figure()
    strip_ax = sns.stripplot(
        data=plot_df,
        y="Brain Measure",
        x="SHAP Value",
        color="#3B4BC0",  # Match the blue color from your example
        alpha=0.5,
    )
    strip_ax.tick_params(axis="x", labelsize=12)
    mean_shap_per_measure = plot_df.groupby("Brain Measure")["SHAP Value"].mean()
    measure_names = plot_df["Brain Measure"].unique()
    ax = plt.gca()
    for idx, measure in enumerate(measure_names):
        mean_value = mean_shap_per_measure[measure]
        # Draw horizontal line at each measure's y-position with its mean x-value
        ax.vlines(
            x=mean_value,
            ymin=idx - 0.2,
            ymax=idx + 0.2,
            colors="#F24236",
            linewidth=2,
            zorder=5,
        )
    plt.xlabel("SHAP value (impact on model output)")

    if not show_labels:
        # Remove labels and tick values from both x and y axes
        ax = plt.gca()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])  # Remove x-axis tick labels but keep ticks
        ax.set_yticklabels([])

    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.75)
    sns.despine(left=True)
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file, dpi=1000, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


FIG_PATH = "..."  # Set your FIG_PATH here.
os.makedirs(FIG_PATH, exist_ok=True)

FEATURE_NAMES = (
    pd.read_csv(
        "..."  # Set your DATA_PATH here.
    )
    .iloc[:, 6:]
    .columns
)

if __name__ == "__main__":
    # Set results file and load them
    results_fname = "..."  # Set your RESULTS_PATH here.
    with open(results_fname, "rb") as f:
        results = pickle.load(f)

    # Get SHAP values
    shap_value = results["shap_values"]["total"]  # Total SHAP values
    shap_values_cv = pd.DataFrame(
        np.array(
            [np.median(np.abs(x.values), axis=0) for x in results["shap_values"]["cv"]]
        ),
        columns=FEATURE_NAMES,
    )  # SHAP values for the CV folds

    # Update feature names for clarity on the beeswarm plot
    for i, f in enumerate(FEATURE_NAMES):
        shap_value.feature_names[i] = f

    print("Plotting SHAP beeswarm plots...")

    plot_shap_beeswarm(
        shap_value,
        show_labels=False,
        out_file=op.join(FIG_PATH, "shap_beeswarm-noLabels.pdf"),
    )
    plot_shap_beeswarm(
        shap_value,
        show_labels=True,
        out_file=op.join(FIG_PATH, "shap_beeswarm-withLabels.pdf"),
    )
    plot_shap_cv(
        shap_values_cv,
        show_labels=False,
        out_file=op.join(FIG_PATH, "shap_beeswarm-cv-noLabels.pdf"),
    )
    plot_shap_cv(
        shap_values_cv,
        show_labels=True,
        out_file=op.join(FIG_PATH, "shap_beeswarm-cv-withLabels.pdf"),
    )
