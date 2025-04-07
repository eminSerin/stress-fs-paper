import os
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.plotting import plot_surf_stat_map
from tqdm import tqdm

WORKING_DIR = "..."  # Set your WORKING_DIR here.
FS_DIR = "..."  # Set your FS_DIR here.

# Plotting settings.
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "figure.figsize": (10, 8),
    }
)

RESULTS_MAP = {
    "left_cacc_volume_men": f"{FS_DIR}/whole-brain/lh.volume.10_increase.glmdir/contrast_men/cache.th30.abs.sig.cluster.mgh",
    "left_cacc_area_men": f"{FS_DIR}/whole-brain/lh.area.10_increase.glmdir/contrast_men/cache.th30.abs.sig.cluster.mgh",
    "right_precentral_thickness_average": f"{FS_DIR}/whole-brain/rh.thickness.10_increase.glmdir/contrast_average/cache.th30.abs.sig.cluster.mgh",
    "right_precentral_thickness_women": f"{FS_DIR}/whole-brain/rh.thickness.10_increase.glmdir/contrast_women/cache.th30.abs.sig.cluster.mgh",
    "right_precentral_thickness_womenGTmen": f"{FS_DIR}/whole-brain/rh.thickness.10_increase.glmdir/contrast_women>men/cache.th30.abs.sig.cluster.mgh",
}

FIGURE_PATH = f"{WORKING_DIR}/figures"
os.makedirs(FIGURE_PATH, exist_ok=True)

# Load the fsaverage mesh and sulcal data.
FSAVERAGE = load_fsaverage(mesh="fsaverage7")
FSAVERAGE_DATA = load_fsaverage_data(mesh="fsaverage7", data_type="sulcal")


def plot_cluster(
    img_path,
    threshold=1.96,
    hemi="left",
    view="medial",
    cmap="coolwarm",
    out_file=None,
    show=True,
):
    # Generate a high-resolution, publication-quality surface statistical map using matplotlib
    cluster_img = nib.load(img_path)
    plot_surf_stat_map(
        surf_mesh=FSAVERAGE["inflated"],
        stat_map=cluster_img.get_fdata(),
        bg_map=FSAVERAGE_DATA,
        threshold=threshold,
        view=view,
        hemi=hemi,
        colorbar=False,  # Enable the colorbar for clarity
        cmap=cmap,
    )
    if show:
        plt.show()
    if out_file is not None:
        plt.savefig(
            op.join(FIGURE_PATH, f"{out_file}.png"),
            dpi=1000,
            bbox_inches="tight",
        )
    plt.close()


# Plot the clusters.
for result, path in tqdm(RESULTS_MAP.items()):
    if "right" in result:
        plot_cluster(path, hemi="right", view="lateral", out_file=result, show=False)
    else:
        plot_cluster(path, out_file=result, show=False)
