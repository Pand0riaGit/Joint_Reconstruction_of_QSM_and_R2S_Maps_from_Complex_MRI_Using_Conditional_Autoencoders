import os
import numpy as np
import nibabel as nib
import torch
from model import Model
from dataset import NiftiDataset
import torch.nn.functional as F


# -------------------------------------------------------
# Device setup
# -------------------------------------------------------
# Clear GPU memory before running this script to avoid OOM.
torch.cuda.empty_cache()
# Use GPU:2 if available, otherwise fall back to CPU.
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------
# File and model paths
# -------------------------------------------------------
# Base directory where ghost phantom data is stored.
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ghost")
# Directory containing the trained model weights.
model_path = os.path.dirname(os.path.realpath(__file__))

# -------------------------------------------------------
# Dataset loader (not directly used in inference here)
# -------------------------------------------------------
# Custom dataset class for NIfTI-based MRI data.
dataset = NiftiDataset(device=device, batch_size=1)

# -------------------------------------------------------
# Load ghost phantom data from NIfTI files
# -------------------------------------------------------
phase = nib.load(os.path.join(base_dir, "phase_ghost.nii.gz")).get_fdata()
magnitude = nib.load(os.path.join(base_dir, "magnitude_ghost.nii.gz")).get_fdata()
qsm = nib.load(os.path.join(base_dir, "susceptibility_ghost.nii.gz")).get_fdata()
r2s = nib.load(os.path.join(base_dir, "r2star_ghost.nii.gz")).get_fdata()

# Load metadata (affine transform + header) for saving results later
qsm_metadata = nib.load(os.path.join(base_dir, "susceptibility_ghost.nii.gz"))
affine = qsm_metadata.affine
header = qsm_metadata.header

# Optional: apply mask (e.g., brain mask) to restrict analysis
mask = nib.load("graz_c_04_susc_mask.nii.gz").get_fdata()

# Debug prints to confirm dataset properties
print("Magnitude Max", magnitude.max(), "Magnitude Min", magnitude.min(), "magnitude size:", magnitude.shape)
print("Phase Max", phase.max(), "Phase Min", phase.min(), "Phase size:", phase.shape)
print("QSM Max", qsm.max(), "QSM Min", qsm.min(), "QSM size:", qsm.shape)
print("R2S Max", r2s.max(), "R2S Min", r2s.min(), "R2S size:", r2s.shape)

# -------------------------------------------------------
# Convert NumPy arrays to Torch complex tensors
# -------------------------------------------------------
# Magnitude and phase are combined into a single complex-valued
# forward simulation volume using polar representation:
#   z = magnitude * exp(i * phase).
magnitude_t = torch.tensor(magnitude, dtype=torch.float32, device=device)
phase_t = torch.tensor(phase, dtype=torch.float32, device=device)

sim_fw_full = torch.polar(magnitude_t, phase_t).to(torch.complex64)  # shape [D,H,W]
print("sim_fw_full shape:", sim_fw_full.shape)

# -------------------------------------------------------
# Load trained model
# -------------------------------------------------------
model = Model().to(device)
checkpoint = torch.load(
    os.path.join(model_path, 'Models/model_add.pt'),
    map_location=device
)
model.load_state_dict(checkpoint)
model.eval()  # inference mode (no dropout/batchnorm updates)

# -------------------------------------------------------
# Patch-based inference function
# -------------------------------------------------------
# Large 3D volumes (256³) cannot be processed in one pass due
# to GPU memory limits. Instead, we:
#   1. Divide the volume into overlapping 3D patches.
#   2. Run the model on each patch.
#   3. Blend patches back together using a Hann window to avoid seams.
#   4. Normalize and crop back to the original shape.
# -------------------------------------------------------
def run_patch_inference(model, volume, patch_size=32, overlap=16, batch_size=1):
    """
    Run patch-based inference on a 3D complex volume.
    Args:
        model: trained PyTorch model
        volume: torch.Tensor [D,H,W] (complex64)
        patch_size: size of cubic patches
        overlap: overlap between adjacent patches (for smooth blending)
        batch_size: how many patches to process simultaneously
    Returns:
        Reconstructed full-volume prediction [D,H,W]
    """
    D, H, W = volume.shape
    stride = patch_size - overlap

    # --- 1. Pad volume so it divides evenly into patches ---
    pad_D = (stride - D % stride) % stride
    pad_H = (stride - H % stride) % stride
    pad_W = (stride - W % stride) % stride
    volume_padded = F.pad(volume, (0, pad_W, 0, pad_H, 0, pad_D))

    Dp, Hp, Wp = volume_padded.shape
    output = torch.zeros((Dp, Hp, Wp), dtype=torch.complex64, device=device)
    weight = torch.zeros((Dp, Hp, Wp), dtype=torch.float32, device=device)

    # --- 2. Define Hann window for patch blending ---
    win = torch.hann_window(patch_size, device=device)
    weight_patch = win[:, None, None] * win[None, :, None] * win[None, None, :]
    weight_patch = weight_patch / weight_patch.max()  # normalize to [0,1]

    patches, coords = [], []

    with torch.no_grad():
        for z in range(0, Dp - patch_size + 1, stride):
            for y in range(0, Hp - patch_size + 1, stride):
                for x in range(0, Wp - patch_size + 1, stride):
                    patch = volume_padded[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    patches.append(patch.unsqueeze(0).unsqueeze(0))  # [1,1,D,H,W]
                    coords.append((z, y, x))

                    # Process patches in batches for efficiency
                    if len(patches) == batch_size:
                        batch = torch.cat(patches, dim=0)
                        preds = model(batch).squeeze(1)  # -> [B,D,H,W]

                        # Blend predictions into output volume
                        for i, (z1, y1, x1) in enumerate(coords):
                            pred = preds[i] * weight_patch
                            output[z1:z1+patch_size, y1:y1+patch_size, x1:x1+patch_size] += pred
                            weight[z1:z1+patch_size, y1:y1+patch_size, x1:x1+patch_size] += weight_patch

                        patches, coords = [], []

        # Process any leftover patches
        if patches:
            batch = torch.cat(patches, dim=0)
            preds = model(batch).squeeze(1)
            for i, (z1, y1, x1) in enumerate(coords):
                pred = preds[i] * weight_patch
                output[z1:z1+patch_size, y1:y1+patch_size, x1:x1+patch_size] += pred
                weight[z1:z1+patch_size, y1:y1+patch_size, x1:x1+patch_size] += weight_patch

    # --- 3. Normalize accumulated patches ---
    weight = torch.clamp(weight, min=1e-6)  # avoid divide by zero
    output = output / weight

    # --- 4. Crop back to original volume size ---
    output = output[:D, :H, :W]

    return output


# -------------------------------------------------------
# Run inference
# -------------------------------------------------------
print("Running patch-based inference...")
yhat = run_patch_inference(model, sim_fw_full, patch_size=128, overlap=16)

# -------------------------------------------------------
# Extract predicted R2* and QSM
# -------------------------------------------------------
# By design, the network outputs complex-valued volumes:
#   - Magnitude → corresponds to R2*
#   - Phase/angle → corresponds to QSM
createdR2S = torch.abs(yhat)
createdQSM = torch.angle(yhat)

print("createdR2S shape:", createdR2S.shape, "Max:", createdR2S.max(), "Min:", createdR2S.min())
print("createdQSM shape:", createdQSM.shape, "Max:", createdQSM.max(), "Min:", createdQSM.min())

# -------------------------------------------------------
# Save predictions as NIfTI files
# -------------------------------------------------------
nib.save(
    nib.Nifti1Image(createdR2S.cpu().detach().numpy(), affine=affine, header=header),
    os.path.join("/home/nikpra/saved/try", 'createdR2S_ghost.nii.gz')
)
nib.save(
    nib.Nifti1Image(createdQSM.cpu().detach().numpy(), affine=affine, header=header),
    os.path.join("/home/nikpra/saved/try", 'createdQSM_ghost.nii.gz')
)

# -------------------------------------------------------
# Compute and save difference maps (prediction - ground truth)
# -------------------------------------------------------
diff_r2s = createdR2S.cpu().detach().numpy() - r2s
nib.save(
    nib.Nifti1Image(diff_r2s, affine=affine, header=header),
    os.path.join("/home/nikpra/saved/try", 'difference_r2s_created_ghost.nii.gz')
)

diff_qsm = createdQSM.cpu().detach().numpy() - qsm
nib.save(
    nib.Nifti1Image(diff_qsm, affine=affine, header=header),
    os.path.join("/home/nikpra/saved/try", 'difference_QSM_created_ghost.nii.gz')
)

# -------------------------------------------------------
# Final cleanup
# -------------------------------------------------------
torch.cuda.empty_cache()  # release GPU memory
