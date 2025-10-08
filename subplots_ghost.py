import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

def show_image(ax, data, title, vmin=None, vmax=None, cmap="gray"):
    ax.imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def extract_slices(img):
    """Extract central sagittal, coronal, and axial slices from a 3D image"""
    return {
        "Sagittal": img[img.shape[0] // 2, :, :],
        "Coronal":  img[:, img.shape[1] // 2, :],
        "Axial":    img[:, :, img.shape[2] // 2]
    }

# --- Base directory for images ---
base_dir = r"Ghost Models"


dir_mask = r"Ghost Models\graz_c_04_susc_mask.nii.gz"

# --- Filenames ---
filenames = {
    "Susceptibility":  "createdQSM_ghost.nii.gz",
    "Diif_susc":       "difference_QSM_created_ghost.nii.gz",
    "r2s":             "createdR2S_ghost.nii.gz",
    "r2s_diff":        "difference_r2s_created_ghost.nii.gz",
}

filenames_ghost = {
    "Susceptibility_ghost":  "susceptibility_ghost.nii.gz",
    "r2star_ghost":          "r2star_ghost.nii.gz",
}

# --- Load mask ---
mask = nib.load(dir_mask).get_fdata()
mask = (mask > 0).astype(np.float32)  # binarize just in case

# --- Load and mask images ---
images = {name: nib.load(os.path.join(base_dir, fname)).get_fdata() * mask
          for name, fname in filenames.items()}

images_ori = {name: nib.load(os.path.join(base_dir, fname)).get_fdata() * mask
              for name, fname in filenames_ghost.items()}

# --- Extract slices ---
slices = {name: extract_slices(img) for name, img in images.items()}
slices_ghost = {name: extract_slices(img) for name, img in images_ori.items()}

# --- Plot 6 rows × 3 cols ---
fig, axes = plt.subplots(6, 3, figsize=(15, 20))
fig.suptitle("Ghost Dataset: Visualization QSM / RS* (Model No-Add / Best Validation Loss)" , fontsize=16)

# Susceptibility row
show_image(axes[0, 0], slices["Susceptibility"]["Sagittal"], "Created QSM (Sagittal)", vmin=-0.005, vmax=0.08373, cmap="gray")
show_image(axes[0, 1], slices["Susceptibility"]["Coronal"],  "Created QSM (Coronal)",  vmin=-0.005, vmax=0.08373, cmap="gray")
show_image(axes[0, 2], slices["Susceptibility"]["Axial"],    "Created QSM (Axial)",    vmin=-0.005, vmax=0.08373, cmap="gray")

show_image(axes[1, 0], slices_ghost["Susceptibility_ghost"]["Sagittal"], "Reference QSM (Sagittal)", vmin=-0.005, vmax=0.08373, cmap="gray")
show_image(axes[1, 1], slices_ghost["Susceptibility_ghost"]["Coronal"],  "Reference QSM (Coronal)",  vmin=-0.005, vmax=0.08373, cmap="gray")
show_image(axes[1, 2], slices_ghost["Susceptibility_ghost"]["Axial"],    "Reference QSM (Axial)",    vmin=-0.005, vmax=0.08373, cmap="gray")

# Diif_susc row
show_image(axes[2, 0], slices["Diif_susc"]["Sagittal"], "Difference Created QSM (Sagittal)", vmin=-0.05, vmax=0.05, cmap="viridis")
show_image(axes[2, 1], slices["Diif_susc"]["Coronal"],  "Difference Created QSM (Coronal)",  vmin=-0.05, vmax=0.05, cmap="viridis")
show_image(axes[2, 2], slices["Diif_susc"]["Axial"],    "Difference Created QSM (Axial)",    vmin=-0.05, vmax=0.05, cmap="viridis")

# r2s row
show_image(axes[3, 0], slices["r2s"]["Sagittal"], "Created R2* (Sagittal)", vmin=0, vmax=50, cmap="gray")
show_image(axes[3, 1], slices["r2s"]["Coronal"],  "Created R2* (Coronal)",  vmin=0, vmax=50, cmap="gray")
show_image(axes[3, 2], slices["r2s"]["Axial"],    "Created R2* (Axial)",    vmin=0, vmax=50, cmap="gray")

show_image(axes[4, 0], slices_ghost["r2star_ghost"]["Sagittal"], "Reference R2*  (Sagittal)", vmin=0, vmax=50, cmap="gray")
show_image(axes[4, 1], slices_ghost["r2star_ghost"]["Coronal"],  "Reference R2*  (Coronal)",  vmin=0, vmax=50, cmap="gray")
show_image(axes[4, 2], slices_ghost["r2star_ghost"]["Axial"],    "Reference R2*  (Axial)",    vmin=0, vmax=50, cmap="gray")

# r2s_diff row
show_image(axes[5, 0], slices["r2s_diff"]["Sagittal"], "Difference Created R2* (Sagittal)", vmin=-5, vmax=5, cmap="viridis")
show_image(axes[5, 1], slices["r2s_diff"]["Coronal"],  "Difference R2* (Coronal)",  vmin=-5, vmax=5, cmap="viridis")
show_image(axes[5, 2], slices["r2s_diff"]["Axial"],    "Difference R2* (Axial)",    vmin=-5, vmax=5, cmap="viridis")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
fig.savefig(r"Subplots\no_add_best_ghost")

# -----------------------------
# Histograms of Difference Volumes (Styled)
# -----------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Histogramm Ghost Dataset Difference QSM / R2* (Model No-Add / Best Validation Loss)", fontsize=16)

# Histogram for QSM Difference
qsm_diff = images["Diif_susc"][mask > 0].flatten()
axes2[0].hist(qsm_diff, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
axes2[0].set_title("Histogram: QSM Difference", fontsize=13)
axes2[0].set_xlabel("Difference Value")
axes2[0].set_ylabel("Voxel Count")
axes2[0].grid(True, linestyle='--', alpha=0.5)

# Histogram for R2* Difference
r2s_diff = images["r2s_diff"][mask > 0].flatten()
axes2[1].hist(r2s_diff, bins=100, color='darkorange', edgecolor='black', alpha=0.7)
axes2[1].set_title("Histogram: R2* Difference", fontsize=13)
axes2[1].set_xlabel("Difference Value")
axes2[1].set_ylabel("Voxel Count")
axes2[1].grid(True, linestyle='--', alpha=0.5)

fig2.savefig(r"Subplots\no_add_best_ghost_histo")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
