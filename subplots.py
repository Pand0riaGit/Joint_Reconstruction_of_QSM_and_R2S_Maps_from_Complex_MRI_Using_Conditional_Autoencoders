import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_laplace
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import torch

# -----------------------------
# Metric Functions
# -----------------------------
def compute_mae(pred, target):
    return np.mean(np.abs(pred - target))

def compute_ssim_3d(pred, target):
    ssim_vals = []
    for i in range(pred.shape[2]):
        ssim_val = ssim(target[:, :, i], pred[:, :, i], data_range=target.max() - target.min())
        ssim_vals.append(ssim_val)
    return np.mean(ssim_vals)

def compute_pearson(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    r, _ = pearsonr(pred_flat, target_flat)
    return r

def compute_hfen(pred, target, sigma=1.5):
    pred_edges = gaussian_laplace(pred, sigma=sigma)
    target_edges = gaussian_laplace(target, sigma=sigma)
    return np.sqrt(np.sum((pred_edges - target_edges) ** 2)) / np.sqrt(np.sum(target_edges ** 2))

# -----------------------------
# Load NIfTI files
# -----------------------------
base_dir = r'Created Artificial Data'
files = {
    'Created QSM': 'createdQSM.nii.gz',
    'Created R2*': 'createdR2S.nii.gz',
    'Diff QSM': 'difference_QSM_created.nii.gz',
    'Diff R2*': 'difference_r2s_created.nii.gz',
    'Phase': 'phase.nii.gz',
    'Magnitude': 'magnitude.nii.gz',
    'QSM Ref': 'qsm.nii.gz',
    'R2* Ref': 'r2s.nii.gz',
}

volumes = {}
slices = {}

for name, filename in files.items():
    path = os.path.join(base_dir, filename)
    print(f"Loading: {path}")
    img = nib.load(path)
    data = img.get_fdata()
    volumes[name] = data
    mid_slice = data.shape[2] // 2
    slices[name] = data[:, :, mid_slice]

# -----------------------------
# Compute and print metrics
# -----------------------------
print("\n--------- Metrics ---------")
metrics = {
    'SUS': {
        'pred': volumes['Created QSM'],
        'gt': volumes['QSM Ref']
    },
    'R2': {
        'pred': volumes['Created R2*'],
        'gt': volumes['R2* Ref']
    }
}

for key, pair in metrics.items():
    mae = compute_mae(pair['pred'], pair['gt'])
    ssim_val = compute_ssim_3d(pair['pred'], pair['gt'])
    pearson_val = compute_pearson(pair['pred'], pair['gt'])
    hfen_val = compute_hfen(pair['pred'], pair['gt'])
    print(f"\n--- {key} ---")
    print(f"MAE:     {mae:.4f}")
    print(f"SSIM:    {ssim_val:.4f}")
    print(f"Pearson: {pearson_val:.4f}")
    print(f"HFEN:    {hfen_val:.4f}")

# -----------------------------
# Difference Value Summary Table
# -----------------------------
diff_data = {
    'QSM Difference': volumes['Diff QSM'],
    'R2* Difference': volumes['Diff R2*']
}

stats = {}
for name, vol in diff_data.items():
    vol_flat = vol.flatten()
    stats[name] = {
        'Min': np.min(vol_flat),
        'Max': np.max(vol_flat),
        'Mean': np.mean(vol_flat),
        'Std': np.std(vol_flat),
        '5th Percentile': np.percentile(vol_flat, 5),
        '95th Percentile': np.percentile(vol_flat, 95)
    }

df = pd.DataFrame(stats).T
print("\n--------- Difference Value Summary ---------")
print(df.to_string(float_format="%.4f"))

# -----------------------------
# Plot middle slices
# -----------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(f"Visualization QSM / R2* / Phase / Magnitude (Model No-Add / Best Validation Loss)\n", fontsize=16)

def show_image(ax, data, title, vmin=None, vmax=None, cmap='gray'):
    data_min = np.min(data)
    data_max = np.max(data)
    scale_info = f"scaled to: [{vmin}, {vmax}]" if vmin is not None and vmax is not None else "no scaling"
    im = ax.imshow(data.T, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\n", fontsize=10)
    ax.axis('off')
    fig.colorbar(im, ax=ax, shrink=0.75)

# QSM row
show_image(axes[0, 0], slices['Created QSM'], 'Created QSM', vmin=-0.4, vmax=0.4, cmap='gray')
show_image(axes[0, 1], slices['QSM Ref'], 'Reference QSM', vmin=-0.4, vmax=0.4, cmap='gray')
show_image(axes[0, 2], slices['Diff QSM'], 'QSM Difference', vmin=-0.4, vmax=0.4, cmap='viridis')

# R2* row
show_image(axes[1, 0], slices['Created R2*'], 'Created R2*', vmin=0, vmax=80, cmap='gray')
show_image(axes[1, 1], slices['R2* Ref'], 'Reference R2*', vmin=0, vmax=80, cmap='gray')
show_image(axes[1, 2], slices['Diff R2*'], 'R2* Difference', vmin=-5, vmax=5, cmap='viridis')

# Phase and Magnitude
show_image(axes[2, 0], slices['Phase'], 'Phase', vmin=-0.14, vmax=0.14, cmap='gray')
show_image(axes[2, 1], slices['Magnitude'], 'Magnitude')
axes[2, 2].axis('off')

fig.savefig(r"Subplots\no_add_best_art")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# -----------------------------
# Bar Chart of Difference Stats
# -----------------------------
# -----------------------------
# Histograms of Difference Volumes
# -----------------------------
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(f"Hisogramm Difference QSM / R2* (Model No-Add / Best Validation Loss)\n", fontsize=16)

# Histogram for QSM Difference
qsm_diff = volumes['Diff QSM'].flatten()
axes[0].hist(qsm_diff, bins=100, color='steelblue', edgecolor='black')
axes[0].set_title("Histogram: QSM Difference", fontsize=13)
axes[0].set_xlabel("Difference Value")
axes[0].set_ylabel("Voxel Count")
axes[0].grid(True, linestyle='--', alpha=0.5)

# Histogram for R2* Difference
r2s_diff = volumes['Diff R2*'].flatten()
axes[1].hist(r2s_diff, bins=100, color='darkorange', edgecolor='black')
axes[1].set_title("Histogram: R2* Difference", fontsize=13)
axes[1].set_xlabel("Difference Value")
axes[1].set_ylabel("Voxel Count")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig3.savefig(r"Subplots\no_add_best_histo")
plt.show()

data = torch.load(os.path.join(base_dir, 'Latest_Model.pt'),map_location=torch.device('cpu'))  # load on CPU

print(data["epoch"])

train_losses = data['train_losses']
val_losses = data['val_losses']
print(train_losses)

# Convert to numpy if needed
train_losses = train_losses if isinstance(train_losses, (list, np.ndarray)) else train_losses.numpy()
val_losses = val_losses if isinstance(val_losses, (list, np.ndarray)) else val_losses.numpy()

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
