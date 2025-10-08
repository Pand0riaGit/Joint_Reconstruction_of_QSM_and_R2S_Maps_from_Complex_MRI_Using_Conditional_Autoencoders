import os

import torch
import torch.nn as nn
import torchcomplex.nn as complexnn
import torch.nn.functional as F


# =======================================================
# Complex Channel Attention
# =======================================================
# This block implements a simple channel reweighting strategy
# for 3D complex-valued feature maps. The average magnitude
# across each channel is compressed through a small bottleneck
# (Conv1x1 → activation → Conv1x1), and the result acts as a
# gating factor between 0 and 1. Both the real and imaginary
# parts of the input are scaled by this weight.
# =======================================================
class ComplexChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv3d(channels // reduction, channels, kernel_size=1)
        self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        print("no_add")  # marker to distinguish this version

    def forward(self, x):
        real, imag = x.real, x.imag

        # Global average pooling over spatial dimensions
        mag = torch.sqrt(real ** 2 + imag ** 2)
        gap = F.adaptive_avg_pool3d(mag, 1)  # shape [B, C, 1, 1, 1]

        # Two-layer channel attention mapping
        weights = self.conv1(gap)
        weights = self.act(weights)
        weights = self.conv2(weights)
        weights = self.sigmoid(weights).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Apply channel weights to both real and imaginary parts
        return torch.complex(real * weights, imag * weights)


# =======================================================
# Complex U-Net Variant
# =======================================================
# This network follows the encoder–bottleneck–decoder design,
# adapted for 3D complex inputs. Each encoder stage downsamples
# and refines the features, while the decoder upsamples and
# combines them with skip connections. Residual shortcuts are
# inserted within each block for stability.
#
# Differences to the other variant:
# - final output is taken directly from the last layer,
#   without adding the input back (no residual at the output).
# =======================================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Nonlinear activation for complex values
        self.nonlinearity = complexnn.CReLU()

        # Attention modules (currently defined but not applied)
        self.e_b1_attention = ComplexChannelAttention(16)
        self.e_b2_attention = ComplexChannelAttention(32)
        self.e_b3_attention = ComplexChannelAttention(64)
        self.bn_attention = ComplexChannelAttention(128)

        # ---------------- Encoder ----------------
        # Block 1 (input → 16 channels)
        self.e_b1_c1 = complexnn.Conv3d(1, 16, 3, stride=1, padding=1, bias=True)
        self.e_b1_c2 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)
        self.e_b1_c3 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)
        self.e_b1_c4 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)
        self.e_b1_dc = complexnn.Conv3d(16, 16, 3, stride=2, padding=1, bias=True)

        # Block 2 (16 → 32 channels)
        self.e_b2_c1 = complexnn.Conv3d(16, 32, 3, stride=1, padding=1, bias=True)
        self.e_b2_c2 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)
        self.e_b2_c3 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)
        self.e_b2_c4 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)
        self.e_b2_dc = complexnn.Conv3d(32, 32, 3, stride=2, padding=1, bias=True)

        # Block 3 (32 → 64 channels)
        self.e_b3_c1 = complexnn.Conv3d(32, 64, 3, stride=1, padding=1, bias=True)
        self.e_b3_c2 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)
        self.e_b3_c3 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)
        self.e_b3_c4 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)
        self.e_b3_dc = complexnn.Conv3d(64, 64, 3, stride=2, padding=1, bias=True)

        # ---------------- Bottleneck ----------------
        self.bn_c1 = complexnn.Conv3d(64, 128, 3, stride=1, padding=1, bias=True)
        self.bn_c2 = complexnn.Conv3d(128, 128, 3, stride=1, padding=1, bias=True)
        self.bn_c3 = complexnn.Conv3d(128, 128, 3, stride=1, padding=1, bias=True)
        self.bn_c4 = complexnn.Conv3d(128, 128, 3, stride=1, padding=1, bias=True)

        # ---------------- Decoder ----------------
        # Decoder Block 3 (128 → 64)
        self.d_b3_us = complexnn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d_b3_uc = complexnn.Conv3d(128, 64, 3, stride=1, padding=1, bias=True)
        self.d_b3_c1 = complexnn.Conv3d(128, 64, 3, stride=1, padding=1, bias=True)
        self.d_b3_c2 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)
        self.d_b3_c3 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)
        self.d_b3_c4 = complexnn.Conv3d(64, 64, 3, stride=1, padding=1, bias=True)

        # Decoder Block 2 (64 → 32)
        self.d_b2_us = complexnn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d_b2_uc = complexnn.Conv3d(64, 32, 3, stride=1, padding=1, bias=True)
        self.d_b2_c1 = complexnn.Conv3d(64, 32, 3, stride=1, padding=1, bias=True)
        self.d_b2_c2 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)
        self.d_b2_c3 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)
        self.d_b2_c4 = complexnn.Conv3d(32, 32, 3, stride=1, padding=1, bias=True)

        # Decoder Block 1 (32 → 16)
        self.d_b1_us = complexnn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d_b1_uc = complexnn.Conv3d(32, 16, 3, stride=1, padding=1, bias=True)
        self.d_b1_c1 = complexnn.Conv3d(32, 16, 3, stride=1, padding=1, bias=True)
        self.d_b1_c2 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)
        self.d_b1_c3 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)
        self.d_b1_c4 = complexnn.Conv3d(16, 16, 3, stride=1, padding=1, bias=True)

        # Final layer (reduce to single channel)
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
        e_b1_rc_mean = 0.5 * (e_b1_c1 + e_b1_c3)
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
        # Block 3 (upsample + skip connection)
        d_b3_us = self.d_b3_us(bn_rc_mean)
        d_b3_uc = self.nonlinearity(self.d_b3_uc(d_b3_us))
        e_b3_crop = e_b3_rc_mean[..., :d_b3_uc.shape[2], :d_b3_uc.shape[3], :d_b3_uc.shape[4]]
        d_b3_cat = torch.cat([d_b3_uc, e_b3_crop], dim=1)
        d_b3_c1 = self.nonlinearity(self.d_b3_c1(d_b3_cat))
        d_b3_c2 = self.nonlinearity(self.d_b3_c2(d_b3_c1))
        d_b3_c3 = self.nonlinearity(self.d_b3_c3(d_b3_c2))
        d_b3_rc_mean = 0.5 * (d_b3_c1 + d_b3_c3)

        # Block 2
        d_b2_us = self.d_b2_us(d_b3_rc_mean)
        d_b2_uc = self.nonlinearity(self.d_b2_uc(d_b2_us))
        e_b2_crop = e_b2_rc_mean[..., :d_b2_uc.shape[2], :d_b2_uc.shape[3], :d_b2_uc.shape[4]]
        d_b2_cat = torch.cat([d_b2_uc, e_b2_crop], dim=1)
        d_b2_c1 = self.nonlinearity(self.d_b2_c1(d_b2_cat))
        d_b2_c2 = self.nonlinearity(self.d_b2_c2(d_b2_c1))
        d_b2_c3 = self.nonlinearity(self.d_b2_c3(d_b2_c2))
        d_b2_rc_mean = 0.5 * (d_b2_c1 + d_b2_c3)

        # Block 1
        d_b1_us = self.d_b1_us(d_b2_rc_mean)
        d_b1_uc = self.nonlinearity(self.d_b1_uc(d_b1_us))
        e_b1_crop = e_b1_rc_mean[..., :d_b1_uc.shape[2], :d_b1_uc.shape[3], :d_b1_uc.shape[4]]
        d_b1_cat = torch.cat([d_b1_uc, e_b1_crop], dim=1)
        d_b1_c1 = self.nonlinearity(self.d_b1_c1(d_b1_cat))
        d_b1_c2 = self.nonlinearity(self.d_b1_c2(d_b1_c1))
        d_b1_c3 = self.nonlinearity(self.d_b1_c3(d_b1_c2))
        d_b1_rc_mean = 0.5 * (d_b1_c1 + d_b1_c3)

        # Final prediction (no residual link to input here)
        yhat = self.last(d_b1_rc_mean)

        return yhat
