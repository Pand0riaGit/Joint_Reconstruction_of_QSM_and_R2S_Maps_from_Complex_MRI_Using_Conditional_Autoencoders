import os

import torch
import torch.nn as nn
import torchcomplex.nn as complexnn
import torch.nn.functional as F


# =======================================================
# Complex Channel Attention Block
# =======================================================
# This module implements a lightweight channel-wise attention
# mechanism for 3D complex data. Instead of treating channels
# equally, it learns a set of weights that highlight the most
# informative ones. The weights are estimated from the global
# magnitude statistics of the feature maps.
# =======================================================
class ComplexChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        """
        Args:
            channels (int): number of feature channels
            reduction (int): reduction factor for bottleneck dimension
        """
        super().__init__()
        # Reduce channel dimension -> apply nonlinearity -> expand back
        self.conv1 = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv3d(channels // reduction, channels, kernel_size=1)
        self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        print("Complex Channel Attention added")

    def forward(self, x):
        # Split complex tensor into real and imaginary parts
        real, imag = x.real, x.imag

        # Compute magnitude and apply global average pooling
        mag = torch.sqrt(real ** 2 + imag ** 2)
        gap = F.adaptive_avg_pool3d(mag, 1)  # output shape [B, C, 1, 1, 1]

        # Pass through bottleneck (1x1 convolutions)
        weights = self.conv1(gap)
        weights = self.act(weights)
        weights = self.conv2(weights)
        weights = self.sigmoid(weights)

        # Expand dimensions to match input
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Rescale both real and imaginary parts
        return torch.complex(real * weights, imag * weights)


# =======================================================
# Complex U-Net-like Model
# =======================================================
# This architecture follows an encoder–bottleneck–decoder
# pattern with skip connections. Each encoder block downsamples
# the input and captures increasingly abstract features, while
# the decoder reconstructs spatial detail with the help of
# skip connections. Residual paths are included to stabilize
# training and reduce vanishing gradients.
# =======================================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Nonlinear activation for complex inputs
        self.nonlinearity = complexnn.CReLU()

        # Attention blocks (not explicitly used in forward pass yet)
        self.e_b1_attention = ComplexChannelAttention(16)
        self.e_b2_attention = ComplexChannelAttention(32)
        self.e_b3_attention = ComplexChannelAttention(64)
        self.bn_attention = ComplexChannelAttention(128)

        # ---------------- Encoder ----------------
        # Block 1 (input → 16 channels)
        self.e_b1_c1 = complexnn.Conv3d(1, 16, 3, stride=1, padding=1, bias=True)
        self.e_b1_c2 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)
        self.e_b1_c3 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)
        self.e_b1_dc = complexnn.Conv3d(16, 16, 3, stride=2, padding=1, bias=True)  # downsample

        # Block 2 (16 → 32 channels)
        self.e_b2_c1 = complexnn.Conv3d(16, 32, 3, stride=1, padding=1, bias=True)
        self.e_b2_c2 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)
        self.e_b2_c3 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)
        self.e_b2_dc = complexnn.Conv3d(32, 32, 3, stride=2, padding=1, bias=True)

        # Block 3 (32 → 64 channels)
        self.e_b3_c1 = complexnn.Conv3d(32, 64, 3, stride=1, padding=1, bias=True)
        self.e_b3_c2 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)
        self.e_b3_c3 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)
        self.e_b3_dc = complexnn.Conv3d(64, 64, 3, stride=2, padding=1, bias=True)

        # ---------------- Bottleneck ----------------
        self.bn_c1 = complexnn.Conv3d(64, 128, 3, stride=1, padding=1, bias=True)
        self.bn_c2 = complexnn.Conv3d(128, 128, 3, stride=1, padding=1, bias=True)
        self.bn_c3 = complexnn.Conv3d(128, 128, 3, stride=1, padding=1, bias=True)

        # ---------------- Decoder ----------------
        # Decoder Block 3 (128 → 64)
        self.d_b3_us = complexnn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d_b3_uc = complexnn.Conv3d(128, 64, 3, stride=1, padding=1, bias=True)
        self.d_b3_c1 = complexnn.Conv3d(128, 64, 3, stride=1, padding=1, bias=True)
        self.d_b3_c2 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)
        self.d_b3_c3 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)

        # Decoder Block 2 (64 → 32)
        self.d_b2_us = complexnn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d_b2_uc = complexnn.Conv3d(64, 32, 3, stride=1, padding=1, bias=True)
        self.d_b2_c1 = complexnn.Conv3d(64, 32, 3, stride=1, padding=1, bias=True)
        self.d_b2_c2 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)
        self.d_b2_c3 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)

        # Decoder Block 1 (32 → 16)
        self.d_b1_us = complexnn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d_b1_uc = complexnn.Conv3d(32, 16, 3, stride=1, padding=1, bias=True)
        self.d_b1_c1 = complexnn.Conv3d(32, 16, 3, stride=1, padding=1, bias=True)
        self.d_b1_c2 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)
        self.d_b1_c3 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)

        # Final output layer (restore channel count to 1)
        self.last = complexnn.Conv3d(16, 1, 3, stride=1, padding=1, bias=True)

    # ===================================================
    # Forward Pass
    # ===================================================
    def forward(self, x):
        # ----- Encoder -----
        # Block 1
        e_b1_c1 = self.nonlinearity(self.e_b1_c1(x))
        e_b1_c2 = self.nonlinearity(self.e_b1_c2(e_b1_c1))
        e_b1_c3 = self.nonlinearity(self.e_b1_c3(e_b1_c2))
        e_b1_rc_mean = 0.5 * (e_b1_c1 + e_b1_c3)  # residual shortcut
        e_b1_dc = self.nonlinearity(self.e_b1_dc(e_b1_rc_mean))

        # Block 2
        e_b2_c1 = self.nonlinearity(self.e_b2_c1(e_b1_dc))
        e_b2_c2 = self.nonlinearity(self.e_b2_c2(e_b2_c1))
        e_b2_c3 = self.nonlinearity(self.e_b2_c3(e_b2_c2))
        e_b2_rc_mean = 0.5 * (e_b2_c1 + e_b2_c3)
        e_b2_dc = self.nonlinearity(self.e_b2_dc(e_b2_rc_mean))

        # Block 3
        e_b3_c1 = self.nonlinearity(self.e_b3_c1(e_b2_dc))
        e_b3_c2 = self.nonlinearity(self.e_b3_c2(e_b3_c1))
        e_b3_c3 = self.nonlinearity(self.e_b3_c3(e_b3_c2))
        e_b3_rc_mean = 0.5 * (e_b3_c1 + e_b3_c3)
        e_b3_dc = self.nonlinearity(self.e_b3_dc(e_b3_rc_mean))

        # ----- Bottleneck -----
        bn_c1 = self.nonlinearity(self.bn_c1(e_b3_dc))
        bn_c2 = self.nonlinearity(self.bn_c2(bn_c1))
        bn_c3 = self.nonlinearity(self.bn_c3(bn_c2))
        bn_rc_mean = 0.5 * (bn_c1 + bn_c3)

        # ----- Decoder -----
        # Block 3
        d_b3_us = self.d_b3_us(bn_rc_mean)
        d_b3_uc = self.nonlinearity(self.d_b3_uc(d_b3_us))
        e_b3_rc_mean_cropped = e_b3_rc_mean[..., :d_b3_uc.shape[2], :d_b3_uc.shape[3], :d_b3_uc.shape[4]]
        d_b3_cat = torch.cat([d_b3_uc, e_b3_rc_mean_cropped], dim=1)
        d_b3_c1 = self.nonlinearity(self.d_b3_c1(d_b3_cat))
        d_b3_c2 = self.nonlinearity(self.d_b3_c2(d_b3_c1))
        d_b3_c3 = self.nonlinearity(self.d_b3_c3(d_b3_c2))
        d_b3_rc_mean = 0.5 * (d_b3_c1 + d_b3_c3)

        # Block 2
        d_b2_us = self.d_b2_us(d_b3_rc_mean)
        d_b2_uc = self.nonlinearity(self.d_b2_uc(d_b2_us))
        e_b2_rc_mean_cropped = e_b2_rc_mean[..., :d_b2_uc.shape[2], :d_b2_uc.shape[3], :d_b2_uc.shape[4]]
        d_b2_cat = torch.cat([d_b2_uc, e_b2_rc_mean_cropped], dim=1)
        d_b2_c1 = self.nonlinearity(self.d_b2_c1(d_b2_cat))
        d_b2_c2 = self.nonlinearity(self.d_b2_c2(d_b2_c1))
        d_b2_c3 = self.nonlinearity(self.d_b2_c3(d_b2_c2))
        d_b2_rc_mean = 0.5 * (d_b2_c1 + d_b2_c3)

        # Block 1
        d_b1_us = self.d_b1_us(d_b2_rc_mean)
        d_b1_uc = self.nonlinearity(self.d_b1_uc(d_b1_us))
        e_b1_rc_mean_cropped = e_b1_rc_mean[..., :d_b1_uc.shape[2], :d_b1_uc.shape[3], :d_b1_uc.shape[4]]
        d_b1_cat = torch.cat([d_b1_uc, e_b1_rc_mean_cropped], dim=1)
        d_b1_c1 = self.nonlinearity(self.d_b1_c1(d_b1_cat))
        d_b1_c2 = self.nonlinearity(self.d_b1_c2(d_b1_c1))
        d_b1_c3 = self.nonlinearity(self.d_b1_c3(d_b1_c2))
        d_b1_rc_mean = 0.5 * (d_b1_c1 + d_b1_c3)

        # ----- Final Prediction -----
        last = self.last(d_b1_rc_mean)

        # Residual connection: model predicts correction to input
        yhat = x + last

        return yhat
