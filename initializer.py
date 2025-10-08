import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =======================================================
# Complex Weight Initialization
# =======================================================
# This function applies a Glorot (Xavier) uniform-style
# initialization adapted for complex-valued tensors.
#
# Key ideas:
# - Real Xavier initialization balances fan-in and fan-out
#   to keep variance stable across layers.
# - For complex numbers, we treat the weights as polar
#   coordinates: radius (modulus) and angle (phase).
# - The modulus is drawn from a uniform distribution
#   scaled according to Xavier theory.
# - The phase is drawn uniformly from [0, 2π), ensuring
#   isotropic distribution in the complex plane.
# =======================================================
def init_weights(self):
    def _complex_glorot_uniform_(tensor):
        """
        In-place initialization of a complex parameter tensor
        using a Glorot-style distribution in polar form.

        Args:
            tensor (torch.Tensor): complex tensor to initialize
        """
        if not tensor.is_complex():
            raise ValueError("Complex Glorot init requires a complex tensor.")

        # Compute fan-in and fan-out from the real part (shape only)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor.real)
        scale = 1.0 / math.sqrt(fan_in + fan_out)

        # Random radius: sqrt ensures proper distribution of magnitude
        modulus = torch.sqrt(torch.empty_like(tensor.real).uniform_(0, scale ** 2))

        # Random phase: uniform sampling ensures isotropy
        phase = torch.empty_like(tensor.real).uniform_(0, 2 * math.pi)

        # Convert to complex weights
        weight = modulus * torch.exp(1j * phase)

        # Copy into the parameter storage
        with torch.no_grad():
            tensor.copy_(weight)

    # ---------------------------------------------------
    # Apply initialization to all modules in the model
    # ---------------------------------------------------
    for m in self.modules():
        # Only apply to complex Conv3d layers
        if isinstance(m, nn.Conv3d) and m.weight.is_complex():
            _complex_glorot_uniform_(m.weight)

            # Bias initialized to zero (common Xavier practice)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
