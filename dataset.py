import numpy as np
import raster_geometry as rg
import torch
from torch.utils.data import IterableDataset
import os


# =======================================================
# Synthetic NIfTI Dataset (Random Shapes)
# =======================================================
# This dataset class generates artificial 3D susceptibility
# distributions made of cuboids and spheres with random size
# and placement. Each shape is assigned tissue-specific
# magnetic properties (χ, R2*), sampled from a dictionary.
#
# The χ map is converted to phase images via dipole
# convolution in Fourier space, and R2* values generate
# magnitude decay images. The result is a set of realistic
# looking complex-valued MR training samples.
# =======================================================
class NiftiDataset(IterableDataset):
    def __init__(self, device, batch_size):
        super(NiftiDataset).__init__()

        self.device = device
        self.batch_size = batch_size
        self.training_dim = 80  # patch size for training

        # ---------------------------------------------------
        # Tissue dictionary: maps region names to
        # susceptibility (ppm) and relaxation rate R2* (s⁻¹).
        # These values are used to simulate realistic
        # contrasts for different brain structures.
        # ---------------------------------------------------
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
            "Frontal deep WM": {"susceptibility": 0.0, "R2*": 35.6},
            "Occipital WM": {"susceptibility": -0.009, "R2*": 34.4},
            "Motor cortex (precentral gyrus)": {"susceptibility": 0.018, "R2*": 33.0},
            "Sensory cortex (postcentral gyrus)": {"susceptibility": 0.014, "R2*": 34.3},
            "Occipital cortex": {"susceptibility": 0.025, "R2*": 34.9},
            "Prefrontal cortex": {"susceptibility": -0.006, "R2*": 23.8},
            "Temporal cortex": {"susceptibility": 0.023, "R2*": 33.2},
            "Cerebrospinal fluid": {"susceptibility": 0.019, "R2*": 1.2},
        }

        # Imaging parameters
        self.TE = 23 / 1000  # echo time in seconds
        self.M0 = 1.0        # baseline magnetization

        # ---------------------------------------------------
        # Create synthetic χ–R2* ground truth volume
        # ---------------------------------------------------
        self.sim_gt_full = self.simulate_susceptibility_sources(
            simulation_dim=120, rectangles_total=400, spheres_total=200
        )

        # Forward simulate MR signals (phase + magnitude)
        self.sim_fw_full = self.forward_convolution(self.sim_gt_full)

    # ===================================================
    # Generate synthetic susceptibility sources
    # ===================================================
    # Random cuboids and spheres are drawn inside the 3D
    # grid, each assigned with susceptibility and R2* from
    # the tissue dictionary. Shapes shrink in size as more
    # are added to improve coverage.
    # ===================================================
    def simulate_susceptibility_sources(self, simulation_dim=80,
                                        rectangles_total=50, spheres_total=40,
                                        sus_std=1,
                                        shape_size_min_factor=0.01,
                                        shape_size_max_factor=0.5):

        # Output volume has two channels:
        # [:,:,:,0] = susceptibility χ
        # [:,:,:,1] = R2* values
        temp = np.zeros((simulation_dim, simulation_dim, simulation_dim, 2))

        # ----- Add Cuboids -----
        for idx in range(rectangles_total):
            shrink_factor = 1 / ((idx / rectangles_total + 1))
            shape_min = np.floor(simulation_dim * shrink_factor * shape_size_min_factor)
            shape_max = np.floor(simulation_dim * shrink_factor * shape_size_max_factor)

            roi = np.random.choice(list(self.tissue_properties.keys()))
            sus_val = self.tissue_properties[roi]["susceptibility"]
            r2_val = self.tissue_properties[roi]["R2*"]

            # Random dimensions and placement
            dx, dy, dz = np.random.randint(shape_min, shape_max, size=3)
            x, y, z = np.random.randint(simulation_dim, size=3)

            temp[x:min(x+dx, simulation_dim),
                 y:min(y+dy, simulation_dim),
                 z:min(z+dz, simulation_dim), 0] = sus_val
            temp[x:min(x+dx, simulation_dim),
                 y:min(y+dy, simulation_dim),
                 z:min(z+dz, simulation_dim), 1] = r2_val

        # ----- Add Spheres -----
        for idx in range(spheres_total):
            shrink_factor = 1 / ((idx / spheres_total + 1))
            shape_min = np.floor(simulation_dim * shrink_factor * shape_size_min_factor)
            shape_max = np.floor(simulation_dim * shrink_factor * shape_size_max_factor)

            roi = np.random.choice(list(self.tissue_properties.keys()))
            sus_val = self.tissue_properties[roi]["susceptibility"]
            r2_val = self.tissue_properties[roi]["R2*"]

            d = np.random.randint(shape_min, shape_max)  # sphere diameter
            x, y, z = np.random.randint(simulation_dim, size=3)

            # Bound sphere within volume
            d = min(d, simulation_dim - max(x, y, z))
            destination = temp[x:x+d, y:y+d, z:z+d, :]

            # Generate binary sphere mask
            sphere = rg.sphere(d, d / 2.)
            destination[sphere] = [sus_val, r2_val]

        return temp

    # ===================================================
    # Dipole Kernel Generator
    # ===================================================
    # Computes the k-space dipole kernel aligned with the
    # main magnetic field direction (here: z-axis).
    # ===================================================
    def generate_3d_dipole_kernel(self, data_shape, voxel_size, b_vec):
        fov = np.array(data_shape) * np.array(voxel_size)
        ry, rx, rz = np.meshgrid(
            np.arange(-data_shape[1] // 2, data_shape[1] // 2),
            np.arange(-data_shape[0] // 2, data_shape[0] // 2),
            np.arange(-data_shape[2] // 2, data_shape[2] // 2),
        )
        rx, ry, rz = rx / fov[0], ry / fov[1], rz / fov[2]
        sq_dist = rx**2 + ry**2 + rz**2
        sq_dist[sq_dist == 0] = 1e-6
        d2 = ((b_vec[0]*rx + b_vec[1]*ry + b_vec[2]*rz)**2) / sq_dist
        return (1/3 - d2)

    # ===================================================
    # Forward Convolution
    # ===================================================
    # Simulates MR signal formation:
    # - χ map → Fourier dipole convolution → Phase
    # - R2* map → exponential decay → Magnitude
    # ===================================================
    def forward_convolution(self, sample):
        chi_sample = sample[..., 0]
        r2_sample = sample[..., 1]

        scaling = np.sqrt(chi_sample.size)

        # χ → Fourier domain
        chi_fft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(chi_sample))) / scaling

        # Dipole convolution
        kernel = self.generate_3d_dipole_kernel(chi_sample.shape, voxel_size=1, b_vec=[0, 0, 1])
        chi_fft_t_kernel = chi_fft * kernel
        tissue_phase = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(chi_fft_t_kernel)))
        tissue_phase = np.real(tissue_phase * scaling)

        # Magnitude from exponential signal decay
        tissue_magnitude = self.M0 * np.exp(-r2_sample * self.TE)

        return np.stack([tissue_phase, tissue_magnitude], axis=-1)

    # ===================================================
    # Patch Extraction
    # ===================================================
    # Random sub-volumes are cropped from the full simulation.
    # Forward patches are converted into complex-valued tensors
    # (magnitude + phase → real + imag).
    # Ground truth patches are also converted to complex format.
    # ===================================================
    def cut_patch(self, dim, batch_index, gt_full, fw_full):
        x_max, y_max, z_max, _ = gt_full.shape

        # Random crop location
        rx = np.random.randint(0, x_max - self.training_dim)
        ry = np.random.randint(0, y_max - self.training_dim)
        rz = np.random.randint(0, z_max - self.training_dim)

        # Forward patch
        fw_patch = fw_full[rx:rx+self.training_dim, ry:ry+self.training_dim, rz:rz+self.training_dim, :]
        fw_mag, fw_phase = fw_patch[..., 1], fw_patch[..., 0]
        fw_real = fw_mag * np.cos(fw_phase)
        fw_imag = fw_mag * np.sin(fw_phase)
        fw_complex = torch.complex(torch.tensor(fw_real), torch.tensor(fw_imag))

        # Ground truth patch
        gt_patch = gt_full[rx:rx+self.training_dim, ry:ry+self.training_dim, rz:rz+self.training_dim, :]
        gt_mag, gt_phase = gt_patch[..., 1], gt_patch[..., 0]
        gt_real = gt_mag * np.cos(gt_phase)
        gt_imag = gt_mag * np.sin(gt_phase)
        gt_complex = torch.complex(torch.tensor(gt_real), torch.tensor(gt_imag))

        return gt_complex, fw_complex

    # ===================================================
    # Generator
    # ===================================================
    # Yields infinite batches of complex-valued training
    # patches in the format:
    #   fw_patches: [batch, 1, D, H, W]
    #   gt_patches: [batch, 1, D, H, W]
    # ===================================================
    def generate(self):
        while True:
            sim_gt_patches = torch.zeros((self.batch_size, self.training_dim,
                                          self.training_dim, self.training_dim),
                                         dtype=torch.complex64)
            sim_fw_patches = torch.zeros_like(sim_gt_patches)

            for i in range(self.batch_size):
                gt_patch, fw_patch = self.cut_patch(self.training_dim, i,
                                                    self.sim_gt_full, self.sim_fw_full)
                sim_gt_patches[i] = gt_patch
                sim_fw_patches[i] = fw_patch

            # Add channel dimension
            sim_fw_patches = sim_fw_patches.unsqueeze(1)
            sim_gt_patches = sim_gt_patches.unsqueeze(1)

            yield sim_fw_patches.to(self.device), sim_gt_patches.to(self.device)

    def __iter__(self):
        return iter(self.generate())
