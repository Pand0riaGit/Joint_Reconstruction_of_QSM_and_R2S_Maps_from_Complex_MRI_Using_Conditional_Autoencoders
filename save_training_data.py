import os
import numpy as np
import nibabel as nib
import torch
from model import Model
from dataset import NiftiDataset

# ======================================================
# Device Setup & Memory Management
# ======================================================
# Clear cached memory from previous CUDA sessions to
# prevent out-of-memory issues.
torch.cuda.empty_cache()
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# ======================================================
# Paths
# ======================================================
# Output directory for results and checkpoint location
this_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Created Artificial Data")
model_path = "Models"
model_name = 'model_add.pt'

# ======================================================
# Load Synthetic Dataset
# ======================================================
dataset = NiftiDataset(device=device, batch_size=1)

# Extract forward-simulated phase and magnitude
phase = dataset.sim_fw_full[..., 0]      # simulated phase
magnitude = dataset.sim_fw_full[..., 1]  # simulated magnitude

# Save phase and magnitude maps as NIfTI
nib.save(nib.Nifti1Image(magnitude, np.eye(4)), os.path.join(this_file_path, 'magnitude.nii.gz'))
print("Magnitude Max:", magnitude.max(), "Min:", magnitude.min())

nib.save(nib.Nifti1Image(phase, np.eye(4)), os.path.join(this_file_path, 'phase.nii.gz'))
print("Phase Max:", phase.max(), "Min:", phase.min())

# ======================================================
# Ground Truth Data
# ======================================================
qsm = dataset.sim_gt_full[..., 0]   # susceptibility map
r2s = dataset.sim_gt_full[..., 1]   # R2* map

# Save ground truth susceptibility and R2* maps
nib.save(nib.Nifti1Image(qsm, np.eye(4)), os.path.join(this_file_path, 'qsm.nii.gz'))
print("QSM Max:", qsm.max(), "Min:", qsm.min())

nib.save(nib.Nifti1Image(r2s, np.eye(4)), os.path.join(this_file_path, 'r2s.nii.gz'))
print("R2S Max:", r2s.max(), "Min:", r2s.min())

# ======================================================
# Convert Forward Data into Complex Torch Tensor
# ======================================================
magnitude_t = torch.tensor(magnitude, dtype=torch.float32, device=device)
phase_t = torch.tensor(phase, dtype=torch.float32, device=device)

# Combine into a single complex-valued tensor
sim_fw_full = torch.polar(magnitude_t, phase_t).to(torch.complex64)
print("sim_fw_full shape:", sim_fw_full.shape)

# ======================================================
# Load Pretrained Model
# ======================================================
model = Model().to(device)
checkpoint = torch.load(
    os.path.join(model_path, model_name),
    map_location=device
)
model.load_state_dict(checkpoint)
model.eval()

# ======================================================
# Run Inference
# ======================================================
# Add batch & channel dimensions → [1, 1, D, H, W]
yhat = model(sim_fw_full.unsqueeze(0).unsqueeze(0))

# Extract predictions
createdR2S = torch.abs(yhat).squeeze(0).squeeze(0)   # magnitude → R2*
createdQSM = torch.angle(yhat).squeeze(0).squeeze(0) # phase → QSM

print("createdR2S shape:", createdR2S.shape, "Max:", createdR2S.max(), "Min:", createdR2S.min())
print("createdQSM shape:", createdQSM.shape, "Max:", createdQSM.max(), "Min:", createdQSM.min())

# ======================================================
# Save Model Predictions
# ======================================================
nib.save(nib.Nifti1Image(createdR2S.cpu().detach().numpy(), np.eye(4)),
         os.path.join(this_file_path, 'createdR2S.nii.gz'))
nib.save(nib.Nifti1Image(createdQSM.cpu().detach().numpy(), np.eye(4)),
         os.path.join(this_file_path, 'createdQSM.nii.gz'))

# ======================================================
# Compute and Save Difference Maps
# ======================================================
# Difference to ground truth for quantitative evaluation
diff_r2s = createdR2S.cpu().detach().numpy() - r2s
nib.save(nib.Nifti1Image(diff_r2s, np.eye(4)),
         os.path.join(this_file_path, 'difference_r2s_created.nii.gz'))

diff_qsm = createdQSM.cpu().detach().numpy() - qsm
nib.save(nib.Nifti1Image(diff_qsm, np.eye(4)),
         os.path.join(this_file_path, 'difference_QSM_created.nii.gz'))

# ======================================================
# Clean Up
# ======================================================
torch.cuda.empty_cache()
