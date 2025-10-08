import numpy as np
import nibabel as nib
import torch
from torch.utils.data import IterableDataset
import os


# =======================================================
# NiftiDataset
# =======================================================
# This dataset class prepares synthetic MRI data from
# susceptibility maps. It creates paired phase and magnitude
# images using a physical forward model (dipole convolution
# for phase, exponential decay for magnitude).
#
# The generated data is saved as NIfTI volumes to enable
# later reuse in training or evaluation.
# =======================================================
class NiftiDataset(IterableDataset):
    def __init__(self, device, batch_size, susc_path, mask, output_dir, TE=23/1000, M0=1.0):
        """
        Args:
            device: Torch device (e.g., "cuda:0")
            batch_size: number of samples per batch
            susc_path: path to input susceptibility NIfTI file
            mask: binary mask path for valid brain region
            output_dir: directory where simulated images are saved
            TE: echo time in seconds (default 23 ms)
            M0: equilibrium magnetization (default 1.0)
        """
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.training_dim = 80  # patch size if later patching is added

        # --- Load susceptibility map ---
        self.susc_img = nib.load(susc_path)
        self.susc_data = self.susc_img.get_fdata()   # raw χ map (256³)
        self.affine = self.susc_img.affine           # affine for saving
        self.header = self.susc_img.header           # header metadata

        # --- Load anatomical mask ---
        self.mask = nib.load(mask).get_fdata()

        # --- Store imaging parameters ---
        self.TE = TE
        self.M0 = M0

        # --- Tissue dictionary (χ → R2*) ---
        # Each entry links a susceptibility value (ppm)
        # to a corresponding relaxation rate (s⁻¹).
        self.tissue_properties = {
            "Head of caudate nucleus": {"susceptibility": 0.044, "R2*": 42.3},
            "External globus pallidus": {"susceptibility": 0.143, "R2*": 85.1},
            "Internal globus pallidus": {"susceptibility": 0.118, "R2*": 81.7},
            "Putamen": {"susceptibility": 0.038, "R2*": 49.4},
            "Red nucleus": {"susceptibility": 0.100, "R2*": 65.6},
            "Red nucleus pars dorsomedialis": {"susceptibility": 0.115, "R2*": 66.0},
            "Red nucleus pars caudalis": {"susceptibility": 0.093, "R2*": 65.1},
            "Red nucleus pars oralis": {"susceptibility": 0.120, "R2*": 69.8},
            "Substantia nigra": {"susceptibility": 0.152, "R2*": 81.7},
            "Subthalamic nucleus": {"susceptibility": 0.111, "R2*": 69.3},
            "Pulvinar of thalamus": {"susceptibility": 0.045, "R2*": 43.0},
            "Anteroprinciple nucleus of thalamus": {"susceptibility": 0.032, "R2*": 41.6},
            "Mediodorsal nucleus of thalamus": {"susceptibility": 0.022, "R2*": 42.5},
            "Dorsal nuclei group of thalamus": {"susceptibility": 0.006, "R2*": 38.5},
            "Posterior limb of internal capsule": {"susceptibility": -0.093, "R2*": 30.2},
            "Splenium of corpus callosum": {"susceptibility": -0.043, "R2*": 37.7},
            "Genu of corpus callosum": {"susceptibility": -0.027, "R2*": 38.6},
            "Frontal subcortical WM": {"susceptibility": 0.018, "R2*": 32.7},
            "Frontal deep WM": {"susceptibility": 0.000, "R2*": 35.6},
            "Occipital WM": {"susceptibility": -0.009, "R2*": 34.4},
            "Motor cortex (precentral gyrus)": {"susceptibility": 0.018, "R2*": 33.0},
            "Sensory cortex (postcentral gyrus)": {"susceptibility": 0.014, "R2*": 34.3},
            "Occipital cortex": {"susceptibility": 0.025, "R2*": 34.9},
            "Prefrontal cortex": {"susceptibility": -0.006, "R2*": 23.8},
            "Temporal cortex": {"susceptibility": 0.023, "R2*": 33.2},
            "Cerebrospinal fluid": {"susceptibility": 0.019, "R2*": 1.2},
        }

        # --- Map χ values to nearest R2* values ---
        self.susc_data, self.r2_map = self.map_susc_to_r2(self.susc_data)

        # --- Simulate phase and magnitude images ---
        self.phase, self.magnitude = self.forward_convolution(self.susc_data, self.r2_map)

        # --- Save outputs to NIfTI for later use ---
        self.save_nifti(self.susc_data, os.path.join(output_dir, "susceptibility_ghost.nii.gz"))
        self.save_nifti(self.r2_map, os.path.join(output_dir, "r2star_ghost.nii.gz"))
        self.save_nifti(self.phase, os.path.join(output_dir, "phase_ghost.nii.gz"))
        self.save_nifti(self.magnitude, os.path.join(output_dir, "magnitude_ghost.nii.gz"))

    # ===================================================
    # χ → R2* Mapping
    # ===================================================
    # Each voxel’s susceptibility is replaced with the closest
    # value from the dictionary, and the corresponding R2*
    # value is assigned. This ensures consistency between χ
    # and R2* while discretizing the range of tissue values.
    # ===================================================
    def map_susc_to_r2(self, susc_data):
        sus_values = np.array([props["susceptibility"] for props in self.tissue_properties.values()])
        r2_values = np.array([props["R2*"] for props in self.tissue_properties.values()])

        flat_susc = susc_data.flatten()

        # Find nearest susceptibility dictionary entry
        idx = np.abs(flat_susc[:, None] - sus_values[None, :]).argmin(axis=1)

        # Replace each voxel with nearest χ and corresponding R2*
        mapped_susc = sus_values[idx].reshape(susc_data.shape)
        mapped_r2 = r2_values[idx].reshape(susc_data.shape)

        return mapped_susc, mapped_r2

    # ===================================================
    # Dipole Kernel Generator
    # ===================================================
    # Creates a 3D dipole kernel in k-space aligned to the
    # main magnetic field (assumed along z-axis here).
    # Used for forward convolution to simulate MR phase.
    # ===================================================
    def generate_3d_dipole_kernel(self, data_shape, voxel_size, b_vec):
        fov = np.array(data_shape) * np.array(voxel_size)

        # Create frequency grid
        ry, rx, rz = np.meshgrid(
            np.arange(-data_shape[1] // 2, data_shape[1] // 2),
            np.arange(-data_shape[0] // 2, data_shape[0] // 2),
            np.arange(-data_shape[2] // 2, data_shape[2] // 2)
        )

        # Normalize by field of view
        rx, ry, rz = rx / fov[0], ry / fov[1], rz / fov[2]

        # Compute dipole term
        sq_dist = rx ** 2 + ry ** 2 + rz ** 2
        sq_dist[sq_dist == 0] = 1e-6  # avoid division by zero
        d2 = ((b_vec[0] * rx + b_vec[1] * ry + b_vec[2] * rz) ** 2) / sq_dist
        kernel = (1 / 3 - d2)
        return kernel

    # ===================================================
    # Forward Model Simulation
    # ===================================================
    # 1. Convolve χ with dipole kernel in Fourier space
    #    → generates MR phase (field perturbations).
    # 2. Apply exponential decay using R2* and TE
    #    → generates MR magnitude signal.
    # ===================================================
    def forward_convolution(self, chi_sample, r2_sample):
        scaling = np.sqrt(chi_sample.size)

        # χ → k-space
        chi_fft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(chi_sample))) / scaling

        # Dipole convolution
        dipole_kernel = self.generate_3d_dipole_kernel(chi_sample.shape, voxel_size=1, b_vec=[0, 0, 1])
        chi_fft_t_kernel = chi_fft * dipole_kernel
        tissue_phase = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(chi_fft_t_kernel)))
        tissue_phase = np.real(tissue_phase * scaling)

        # Magnitude decay
        tissue_magnitude = self.M0 * np.exp(-r2_sample * self.TE)

        return tissue_phase, tissue_magnitude

    # ===================================================
    # Save to NIfTI
    # ===================================================
    def save_nifti(self, data, out_path):
        img = nib.Nifti1Image(data.astype(np.float32), affine=self.affine, header=self.header)
        nib.save(img, out_path)

    # ===================================================
    # Iterator (not used in this setup)
    # ===================================================
    def __iter__(self):
        # No patch-wise sampling in this implementation
        return iter([])


# =======================================================
# Example: Generate ghost phantom data
# =======================================================
dataset = NiftiDataset(
    device="cuda:0",
    batch_size=1,
    susc_path="Ghost Models\graz_c_04_susc.nii.gz",
    mask="Ghost Models\graz_c_04_susc_mask.nii.gz",
    output_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "Ghost Models")
)
