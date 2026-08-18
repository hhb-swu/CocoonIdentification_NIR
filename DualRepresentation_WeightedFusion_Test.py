"""Test-set evaluation for the dual-representation weighted-fusion network.

The script reproduces prediction, classification metrics, confusion matrix, ROC,
and UMAP analyses for the finalized model. Machine-specific absolute paths have
been replaced with command-line arguments and repository-relative defaults.

Path configuration
------------------
The script assumes that it is placed in the repository root directory.

Default 1D spectral-sequence path:
    data/Spectral_Sequences/TestSet

Default 2D recurrence-plot path:
    data/Recurrence_Plots/TestSet

Default model-weight path:
    weights

Default result-output path:
    results

Expected repository structure:

    <repository>/
    ├── DualRepresentation_WeightedFusion_Test.py
    ├── data/
    │   ├── Spectral_Sequences/
    │   │   └── TestSet/
    │   │       ├── Day1/
    │   │       │   ├── 0/
    │   │       │   ├── 1/
    │   │       │   └── 2/
    │   │       ├── ...
    │   │       └── Day10/
    │   └── Recurrence_Plots/
    │       └── TestSet/
    │           ├── Day1/
    │           │   ├── 0/
    │           │   ├── 1/
    │           │   └── 2/
    │           ├── ...
    │           └── Day10/
    ├── weights/
    │   ├── Optimal_1D-CNN_Weights.pth
    │   ├── Optimal_2D-CNN_Weights.pth
    │   └── DualRepresentation_Weighted-Fusion_Network_Weights.pth
    └── results/

Class encoding:
    0 = Female cocoon
    1 = Male cocoon
    2 = Imprinted-dead cocoon

The 1D and 2D files are paired by identical file stem within the same Day and
class folder. Therefore, corresponding files must have the same file name.

Important
---------
This script evaluates predictive performance. It does not reproduce the separate
network-inference timing protocol reported in the manuscript.
"""

from __future__ import annotations

import argparse
import math
import os
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import rcParams
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset

# ----------------------------- Checkpoint loading -----------------------------
def load_state_dict_file(path, map_location="cpu"):
    """Load a released PyTorch state dictionary with strict checkpoint semantics."""
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")

    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}

    return checkpoint


# ----------------------------- Attention modules -----------------------------
class AdaptiveCoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=16, alpha=0.9):
        super(AdaptiveCoordAtt, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)
        self.alpha = alpha

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.shared_conv(y)

        x_h_out, x_w_out = torch.split(y, [h, w], dim=2)
        x_w_out = x_w_out.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h_out * self.alpha).sigmoid()
        a_w = self.conv_w(x_w_out * self.alpha).sigmoid()

        out = x * (a_h + a_w)
        return out


class sa_layer(nn.Module):
    """Shuffle-attention layer combining grouped channel and spatial attention."""

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class AdaptiveCoordAtt1D(nn.Module):
    """One-dimensional adaptive coordinate attention with local-position and global-context branches."""
    def __init__(self, in_channels, reduction=16, alpha=0.9):
        super(AdaptiveCoordAtt1D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)
        self.alpha = alpha

        self.shared_conv = nn.Sequential(
            nn.Conv1d(in_channels, self.mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_l = nn.Conv1d(self.mid_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.conv_g = nn.Conv1d(self.mid_channels, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        """Apply attention to an input tensor with shape ``(B, C, L)``."""
        b, c, l = x.shape

        x_l = x

        x_g = F.adaptive_avg_pool1d(x, 1)

        y = torch.cat([x_l, x_g], dim=2)

        y = self.shared_conv(y)

        a_l, a_g = torch.split(y, [l, 1], dim=2)

        a_l = self.conv_l(a_l * self.alpha).sigmoid()
        a_g = self.conv_g(a_g * self.alpha).sigmoid()

        out = x * (a_l + a_g)

        return out


# -------------------------- 1D backbone components --------------------------
class LayerNormFunction1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, L = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, L = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(dim=0), None

class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction1D.apply(x, self.weight, self.bias, self.eps)


class SimpleGate1D(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Branch1D(nn.Module):
    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.branch = nn.Sequential(
            nn.Conv1d(in_channels=self.dw_channel, out_channels=self.dw_channel,
                      kernel_size=3, padding=dilation, stride=1,
                      groups=self.dw_channel, bias=True, dilation=dilation)
        )

    def forward(self, input):
        return self.branch(input)


class DiSpAM1D(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv1d(in_channels=c, out_channels=self.dw_channel,
                               kernel_size=1, padding=0, stride=1, bias=True)

        self.extra_conv = nn.Conv1d(self.dw_channel, self.dw_channel,
                                    kernel_size=3, padding=1, stride=1,
                                    groups=c, bias=True) if extra_depth_wise else nn.Identity()

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch1D(self.dw_channel, DW_Expand=1, dilation=dilation))

        assert len(dilations) == len(self.branches)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2,
                      kernel_size=1, padding=0, stride=1, bias=True)
        )
        self.sg1 = SimpleGate1D()
        self.sg2 = SimpleGate1D()
        self.conv3 = nn.Conv1d(in_channels=self.dw_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, bias=True)

        ffn_channel = FFN_Expand * c
        # Retained unchanged for exact compatibility with the released checkpoint.
        self.conv4 = nn.Conv1d(in_channels=c, out_channels=ffn_channel,
                               kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=ffn_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, bias=True)

        self.norm1 = LayerNorm1d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1)), requires_grad=True)

    def forward(self, inp, adapter=None):
        y = inp
        x = self.norm1(inp)

        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)

        y = inp + self.beta * x

        return y


class GRN1D(nn.Module):
    """Global Response Normalization adapted to one-dimensional feature maps."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def pick_gn_groups(channels: int, gn_groups: int = 8) -> int:
    """Return the largest divisor of ``channels`` that does not exceed ``gn_groups``."""
    if channels % gn_groups == 0:
        return gn_groups
    for g in range(min(gn_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvNeXtV2Block1D(nn.Module):
    """ConvNeXt V2-style one-dimensional residual block using channels-first tensors."""
    def __init__(self, dim, kernel_size=7, stride=1, dilation=1,
                 drop_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 gn_groups: int = 8, keep_shape: bool = True):
        super().__init__()
        self.k, self.s, self.d, self.keep_shape = kernel_size, stride, dilation, keep_shape

        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                stride=stride, padding=0, dilation=dilation,
                                groups=dim, bias=False)
        self.gn  = nn.GroupNorm(pick_gn_groups(dim, gn_groups), dim)
        self.act1 = ScaledTanh()

        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1, bias=True)
        self.act2 = ScaledTanh()
        self.grn = GRN1D(4 * dim)
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1, bias=True)

        self.shortcut = nn.Identity() if stride == 1 else nn.Conv1d(dim, dim, 1, stride=stride, bias=False)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def _same_pad_1d(self, x: torch.Tensor) -> torch.Tensor:
        if not self.keep_shape:
            return x
        L = x.size(-1)
        out_len = math.ceil(L / self.s)
        needed = max(0, (out_len - 1) * self.s + (self.d * (self.k - 1) + 1) - L)
        if needed:
            left = needed // 2
            right = needed - left
            x = F.pad(x, (left, right))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self._same_pad_1d(x)
        x = self.dwconv(x)
        x = self.gn(x)
        x = self.act1(x)

        x = self.pwconv1(x)
        x = self.act2(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1)

        identity = self.shortcut(identity)

        if x.size(-1) != identity.size(-1):
            diff = identity.size(-1) - x.size(-1)
            x = F.pad(x, (0, diff)) if diff > 0 else x[..., :identity.size(-1)]

        return identity + self.drop_path(x)

def _is_npy(p): return p.lower().endswith('.npy')


def _base_stem(p): return os.path.splitext(os.path.basename(p))[0]


# ------------------------------- Data loading --------------------------------
class PairedSpectraImageDataset(Dataset):
    """Load paired 1D spectral sequences and 2D recurrence plots by matching file stems.

    Expected directory layout::

        root_1d / Day1 / 0 / sample.npy
        root_1d / Day1 / 1 / sample.npy
        root_1d / Day1 / 2 / sample.npy
        ...
        root_1d / Day10 / {0,1,2} / sample.npy

        root_2d / Day1 / 0 / sample.npy
        root_2d / Day1 / 1 / sample.npy
        root_2d / Day1 / 2 / sample.npy
        ...
        root_2d / Day10 / {0,1,2} / sample.npy

    Class encoding:
        0 = Female cocoon
        1 = Male cocoon
        2 = Imprinted-dead cocoon

    The released final test set contains female and male samples from Day1-Day9
    and imprinted-dead samples from Day1-Day10.

    Returns a 1D tensor ``[1, L]``, a 2D tensor ``[1, H, W]``, the class label,
    both source paths, and the developmental-day folder name.
    """

    def __init__(self, root_1d, root_2d, days=None, label_names=('0', '1', '2'),
                 transform_2d=None):
        super().__init__()
        self.root_1d = os.fspath(root_1d)
        self.root_2d = os.fspath(root_2d)
        self.label_names = list(label_names)
        self.transform_2d = transform_2d

        if not os.path.isdir(self.root_1d):
            raise RuntimeError(
                f"[PairedDataset] 1D test-set directory does not exist: {self.root_1d}"
            )
        if not os.path.isdir(self.root_2d):
            raise RuntimeError(
                f"[PairedDataset] 2D test-set directory does not exist: {self.root_2d}"
            )

        if days is None:
            day_dirs = [
                d for d in os.listdir(self.root_1d)
                if os.path.isdir(os.path.join(self.root_1d, d)) and d.startswith("Day")
            ]
            days = sorted(
                day_dirs,
                key=lambda name: int(name[3:]) if name[3:].isdigit() else float("inf"),
            )
        self.days = list(days)

        pairs = []
        miss = 0

        for day in self.days:
            for y, label_name in enumerate(self.label_names):
                dir_1d = os.path.join(self.root_1d, day, label_name)
                dir_2d = os.path.join(self.root_2d, day, label_name)

                if not (os.path.isdir(dir_1d) and os.path.isdir(dir_2d)):
                    continue

                files_1d = sorted(
                    [f for f in os.listdir(dir_1d) if _is_npy(f)],
                    key=_base_stem,
                )
                files_2d = sorted(
                    [f for f in os.listdir(dir_2d) if _is_npy(f)],
                    key=_base_stem,
                )

                map_1d = {
                    _base_stem(f): os.path.join(dir_1d, f)
                    for f in files_1d
                }
                map_2d = {
                    _base_stem(f): os.path.join(dir_2d, f)
                    for f in files_2d
                }

                matched_stems = sorted(
                    set(map_1d.keys()) & set(map_2d.keys()),
                    key=lambda stem: (len(stem), stem),
                )

                miss += len(map_1d) + len(map_2d) - 2 * len(matched_stems)

                for stem in matched_stems:
                    pairs.append((map_1d[stem], map_2d[stem], y, day))

        if not pairs:
            raise RuntimeError(
                "[PairedDataset] No paired samples were found. "
                "Verify that matching .npy file stems exist in the 1D and 2D test-set directories."
            )

        if miss != 0:
            raise RuntimeError(
                f"[PairedDataset] Found {miss} unmatched 1D/2D file(s). "
                "Each spectral sequence must have an RP with the same file stem, "
                "Day folder, and class folder."
            )

        self.samples = pairs
        print(
            f"[PairedDataset] days={self.days} | "
            f"pairs={len(self.samples)} | unmatched={miss}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1d, p2d, y, day = self.samples[idx]

        x1d = np.load(p1d, mmap_mode='r').astype(np.float32, copy=False)
        if x1d.ndim == 1:
            x1d = x1d[None, :]
        elif not (x1d.ndim == 2 and x1d.shape[0] == 1):
            raise ValueError(
                f"Unexpected 1D spectral shape {x1d.shape} for {p1d}; "
                "expected (922,) or (1, 922)."
            )
        if x1d.shape[-1] != EXPECTED_1D_LENGTH:
            raise ValueError(
                f"Unexpected 1D spectral length {x1d.shape[-1]} for {p1d}; "
                f"expected {EXPECTED_1D_LENGTH}."
            )

        x1d = np.array(x1d, dtype=np.float32, copy=True)
        x1d = torch.from_numpy(x1d)

        x2d = np.load(p2d, mmap_mode='r').astype(np.float32, copy=False)
        if x2d.ndim == 2:
            x2d = x2d[None, :, :]
        elif not (x2d.ndim == 3 and x2d.shape[0] == 1):
            raise ValueError(
                f"Unexpected RP shape {x2d.shape} for {p2d}; "
                "expected (458, 458) or (1, 458, 458)."
            )
        if tuple(x2d.shape[-2:]) != EXPECTED_RP_SHAPE:
            raise ValueError(
                f"Unexpected RP spatial shape {tuple(x2d.shape[-2:])} for {p2d}; "
                f"expected {EXPECTED_RP_SHAPE}."
            )

        x2d = np.array(x2d, dtype=np.float32, copy=True)
        x2d = torch.from_numpy(x2d)
        if self.transform_2d:
            x2d = self.transform_2d(x2d)

        y = torch.tensor(y, dtype=torch.long)
        return x1d, x2d, y, p1d, p2d, day


class ScaledTanh(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh(0.6666667 * x)


# ------------------------------ 1D representation -----------------------------
class Net1D(nn.Module):
    def __init__(self):
        super(Net1D, self).__init__()
        self.dims = [8, 16, 16, 32, 32, 64, 64]
        self.kernels = [11, 11, 9, 9, 7, 7, 5]
        self.strides = [1, 1, 1, 1, 1, 1, 1]

        self.feature_extractor_1 = nn.Sequential(
            nn.Conv1d(1, self.dims[0], kernel_size=self.kernels[0],stride=self.strides[0], padding=self.kernels[0] // 2),
            nn.GroupNorm(8,self.dims[0]),
            ScaledTanh(),
            DiSpAM1D(c=self.dims[0], DW_Expand=2, dilations=[1, 4, 9], extra_depth_wise=True),
            ConvNeXtV2Block1D(self.dims[0],self.dims[0]),
            nn.AvgPool1d(2),
        )
        self.feature_extractor_2=nn.Sequential(
            nn.Conv1d(self.dims[0], self.dims[1], kernel_size=self.kernels[1], stride=self.strides[1],padding=self.kernels[1] // 2),
            nn.GroupNorm(8, self.dims[1]),
            ScaledTanh(),
            DiSpAM1D(c=self.dims[1], DW_Expand=2, dilations=[1, 4, 9], extra_depth_wise=True),
            ConvNeXtV2Block1D(self.dims[1], self.dims[1]),
            nn.AvgPool1d(2),
        )
        self.feature_extractor_3=nn.Sequential(
            nn.Conv1d(self.dims[1], self.dims[2], kernel_size=self.kernels[2], stride=self.strides[2], padding=self.kernels[2]//2),
            nn.GroupNorm(8, self.dims[2]),
            ScaledTanh(),
            DiSpAM1D(c=self.dims[2], DW_Expand=2, dilations=[1, 4, 9], extra_depth_wise=True),
            ConvNeXtV2Block1D(self.dims[2], self.dims[2]),
            nn.AvgPool1d(2),
        )
        self.feature_extractor_4=nn.Sequential(
            nn.Conv1d(self.dims[2], self.dims[3], kernel_size=self.kernels[3], stride=self.strides[3], padding=self.kernels[3]//2),
            nn.GroupNorm(8, self.dims[3]),
            ScaledTanh(),
            AdaptiveCoordAtt1D(in_channels=self.dims[3], reduction=16, alpha=0.9),
            ConvNeXtV2Block1D(self.dims[3], self.dims[3]),
            nn.AvgPool1d(2),
        )
        self.feature_extractor_5=nn.Sequential(
            nn.Conv1d(self.dims[3], self.dims[4], kernel_size=self.kernels[4], stride=self.strides[4], padding=self.kernels[4]//2),
            nn.GroupNorm(8, self.dims[4]),
            ScaledTanh(),
            AdaptiveCoordAtt1D(in_channels=self.dims[4], reduction=16, alpha=0.9),
            ConvNeXtV2Block1D(self.dims[4], self.dims[4]),
            nn.AvgPool1d(2),
        )
        self.feature_extractor_6=nn.Sequential(
            nn.Conv1d(self.dims[4], self.dims[5], kernel_size=self.kernels[5], stride=self.strides[5], padding=self.kernels[5]//2),
            nn.GroupNorm(8, self.dims[5]),
            ScaledTanh(),
            AdaptiveCoordAtt1D(in_channels=self.dims[5], reduction=16, alpha=0.9),
            ConvNeXtV2Block1D(self.dims[5], self.dims[5]),
            nn.AvgPool1d(2),
        )
        self.feature_extractor_7=nn.Sequential(
            nn.Conv1d(self.dims[5], self.dims[6], kernel_size=self.kernels[6], stride=self.strides[6], padding=self.kernels[6]//2),
            nn.GroupNorm(8, self.dims[6]),
            ScaledTanh(),
            AdaptiveCoordAtt1D(in_channels=self.dims[6], reduction=16, alpha=0.9),
            ConvNeXtV2Block1D(self.dims[6], self.dims[6]),
            nn.AdaptiveAvgPool1d(output_size=3)
        )

        # Retained for strict compatibility with the standalone 1D checkpoint.
        self.classifier = nn.Sequential(
            nn.Linear(self.dims[6] * 3, self.dims[6]//2),
            ScaledTanh(),
            nn.Dropout(p=0.5),

            nn.Linear(self.dims[6]//2, 3)
        )

    def forward(self, x):
        x=self.feature_extractor_1(x)

        x=self.feature_extractor_2(x)

        x=self.feature_extractor_3(x)

        x=self.feature_extractor_4(x)

        x=self.feature_extractor_5(x)

        x=self.feature_extractor_6(x)

        x=self.feature_extractor_7(x)

        return x

    def get_features(self, x):
        """Return branch features before the standalone classification head."""
        return self.forward(x)

# ------------------------------ 2D representation -----------------------------
class ResNet2D(nn.Module):
    """Residual 2D block without projection; input and output channels must match."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = ScaledTanh()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        self.dims = [20, 40, 60, 80, 100]
        self.kernels = [7, 5, 5, 3, 3]
        self.strides = [2, 1, 1, 1, 1]

        self.act = ScaledTanh()

        self.feature_extractor_1 = nn.Sequential(
            nn.Conv2d(1, self.dims[0], kernel_size=self.kernels[0], stride=self.strides[0],
                      padding=self.kernels[0] // 2),
            nn.BatchNorm2d(self.dims[0]),
            self.act,
            sa_layer(self.dims[0], groups=10),
            ResNet2D(self.dims[0], self.dims[0]),
            nn.AvgPool2d(2),
        )
        self.feature_extractor_2 = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[1], kernel_size=self.kernels[1], stride=self.strides[1],
                      padding=self.kernels[1] // 2),
            nn.BatchNorm2d(self.dims[1]),
            self.act,
            AdaptiveCoordAtt(in_channels=self.dims[1], reduction=16),
            ResNet2D(self.dims[1], self.dims[1]),
            nn.AvgPool2d(2),
        )
        self.feature_extractor_3 = nn.Sequential(
            nn.Conv2d(self.dims[1], self.dims[2], kernel_size=self.kernels[2], stride=self.strides[2],
                      padding=self.kernels[2] // 2),
            nn.BatchNorm2d(self.dims[2]),
            self.act,
            AdaptiveCoordAtt(in_channels=self.dims[2], reduction=16),
            ResNet2D(self.dims[2], self.dims[2]),
            nn.AvgPool2d(2),
        )
        self.feature_extractor_4 = nn.Sequential(
            nn.Conv2d(self.dims[2], self.dims[3], kernel_size=self.kernels[3], stride=self.strides[3],
                      padding=self.kernels[3] // 2),
            nn.BatchNorm2d(self.dims[3]),
            self.act,
            AdaptiveCoordAtt(in_channels=self.dims[3], reduction=16),
            ResNet2D(self.dims[3], self.dims[3]),
            nn.AvgPool2d(2),
        )
        self.feature_extractor_5 = nn.Sequential(
            nn.Conv2d(self.dims[3], self.dims[4], kernel_size=self.kernels[4], stride=self.strides[4],
                      padding=self.kernels[4] // 2),
            nn.BatchNorm2d(self.dims[4]),
            self.act,
            AdaptiveCoordAtt(in_channels=self.dims[4], reduction=16),

            ResNet2D(self.dims[4], self.dims[4]),
            nn.AdaptiveAvgPool2d(output_size=3),
            nn.Flatten(),
        )

        # Retained for strict compatibility with the standalone 2D checkpoint.
        self.classifier = nn.Sequential(
            nn.Linear(self.dims[4] * 9, self.dims[4] // 2),
            self.act,
            nn.Dropout(p=0.5),
            nn.Linear(self.dims[4] // 2, 3)
        )

    def forward(self, x):
        x = self.feature_extractor_1(x)
        x = self.feature_extractor_2(x)
        x = self.feature_extractor_3(x)
        x = self.feature_extractor_4(x)
        x = self.feature_extractor_5(x)
        return x

    def get_features(self, x):
        """Return branch features before the standalone classification head."""
        return self.forward(x)


# -------------------------- Dual-representation fusion ------------------------
class WeightedFusionNet(nn.Module):
    """Dual-representation weighted-fusion network using pretrained 1D and 2D branches."""
    def __init__(self, net1d_path, net2d_path, num_classes=3, dropout=0.5,
                 freeze_backbone=True, proj_dim=128):
        super().__init__()

        self.net1d = Net1D()
        self.net1d.load_state_dict(load_state_dict_file(net1d_path, map_location='cpu'), strict=True)
        print(f"Loaded pretrained 1D model: {net1d_path}")

        self.net2d = Net2D()
        self.net2d.load_state_dict(load_state_dict_file(net2d_path, map_location='cpu'), strict=True)
        print(f"Loaded pretrained 2D model: {net2d_path}")

        if freeze_backbone:
            for param in self.net1d.parameters():
                param.requires_grad = False
            for param in self.net2d.parameters():
                param.requires_grad = False
            print("Backbone parameters are frozen.")

        self.feat1d_dim = 64 * 3
        self.feat2d_dim = 100 * 3 * 3
        self.proj_dim = proj_dim

        print(f"Original feature dimensions: 1D={self.feat1d_dim}, 2D={self.feat2d_dim}")
        print(f"Projection dimension: {self.proj_dim}")

        self.norm1d = nn.BatchNorm1d(self.feat1d_dim)
        self.norm2d = nn.BatchNorm1d(self.feat2d_dim)

        self.proj1d = nn.Sequential(
            nn.Linear(self.feat1d_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            ScaledTanh()
        )
        self.proj2d = nn.Sequential(
            nn.Linear(self.feat2d_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            ScaledTanh()
        )

        self.weight_1d = nn.Parameter(torch.ones(1) * 0.5)
        self.weight_2d = nn.Parameter(torch.ones(1) * 0.5)

        fusion_dim = self.proj_dim

        act = ScaledTanh()
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            act,
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )

    def forward(self, x1d, x2d):

        feat1d = self.net1d.get_features(x1d)
        feat2d = self.net2d.get_features(x2d)

        feat1d_flat = feat1d.reshape(feat1d.size(0), -1)
        feat2d_flat = feat2d.reshape(feat2d.size(0), -1)

        feat1d_norm = self.norm1d(feat1d_flat)
        feat2d_norm = self.norm2d(feat2d_flat)

        feat1d_proj = self.proj1d(feat1d_norm)
        feat2d_proj = self.proj2d(feat2d_norm)

        weights = torch.softmax(torch.stack([self.weight_1d, self.weight_2d]), dim=0)
        w1d, w2d = weights[0], weights[1]

        fused_feat = w1d * feat1d_proj + w2d * feat2d_proj

        output = self.classifier(fused_feat)

        return output, (w1d.item(), w2d.item())

    def get_features(self, x1d, x2d):
        """Return the weighted concatenation of normalized branch features for UMAP.

        Classification itself uses the projected weighted-sum feature computed in ``forward``.
        This method preserves the representation used for the manuscript UMAP analysis.
        """

        feat1d = self.net1d.get_features(x1d)
        feat2d = self.net2d.get_features(x2d)

        feat1d_flat = feat1d.reshape(feat1d.size(0), -1)
        feat2d_flat = feat2d.reshape(feat2d.size(0), -1)

        feat1d_norm = self.norm1d(feat1d_flat)
        feat2d_norm = self.norm2d(feat2d_flat)

        weights = torch.softmax(torch.stack([self.weight_1d, self.weight_2d]), dim=0)
        w1d, w2d = weights[0], weights[1]

        weighted_feat1d = w1d * feat1d_norm
        weighted_feat2d = w2d * feat2d_norm

        fused_feat = torch.cat([weighted_feat1d, weighted_feat2d], dim=1)

        return fused_feat


# ---------------------------- Evaluation utilities ----------------------------
def plot_roc_curve(y_true, y_prob, class_names, save_path=None):
    """Plot one-vs-rest ROC curves for the three-class prediction task."""
    n_classes = len(class_names)

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(6.5, 6))
    lw = 1.6
    colors = cycle(['red', 'blue', 'turquoise', 'orange', 'purple'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC of {class_names[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.close()

    return roc_auc


def plot_umap(features_array, labels_array, class_names, save_path=None):
    """Visualize extracted representation features using UMAP."""

    try:
        import umap.umap_ as umap
    except ImportError as exc:
        raise ImportError("UMAP visualization requires the 'umap-learn' package.") from exc

    reducer = umap.UMAP(random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(features_array)

    plt.figure(figsize=(6.5, 6))

    colors = ['red', 'blue', 'turquoise']
    markers = ['*', '^', 'x']
    size = 6

    for i, class_name in enumerate(class_names):
        indices = labels_array == i
        plt.scatter(embedding[indices, 0], embedding[indices, 1],
                    color=colors[i], marker=markers[i], label=class_name, s=size)

    plt.legend(loc="best", fontsize=12)

    plt.xlabel('UMAP 1', fontsize=14)
    plt.ylabel('UMAP 2', fontsize=14)
    plt.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.close()

    return embedding

def plot_cm(cm, class_names, save_path=None):
    cm = cm.astype(np.int64)
    proportions = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    disp_text = np.vectorize(lambda v: f"{v:.2%}")(proportions)

    rcParams.update({"font.family": "serif"})
    plt.figure(figsize=(7, 7))
    plt.imshow(proportions, interpolation='nearest', cmap='RdPu')
    plt.colorbar(fraction=0.045, pad=0.04)

    ticks = np.arange(len(class_names))

    plt.xticks(ticks, class_names, fontsize=14, rotation=0, weight=8)
    plt.yticks(ticks, class_names, fontsize=14, weight=8)

    thresh = proportions.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if proportions[i, j] > thresh else "black"
            plt.text(j, i, f"{cm[i, j]}\n{disp_text[i, j]}",
                     ha='center', va='center', fontsize=12, color=color)

    plt.ylabel('True label', fontsize=15, weight=5)
    plt.xlabel('Predicted label', fontsize=15, weight=5)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

# ------------------------------ Test-set runner -------------------------------

CLASSES = ["Female", "Male", "Imprinted-dead"]
CLASSES_CM = ["Female", "Male", "Imprinted-\ndead"]
CLASSES_ROC = ["female cocoon", "male cocoon", "imprinted-dead cocoon"]
CLASSES_UMAP = ["Female cocoon", "Male cocoon", "Imprinted-dead cocoon"]
DEFAULT_DAYS = [f"Day{day}" for day in range(1, 11)]
EXPECTED_1D_LENGTH = 922
EXPECTED_RP_SHAPE = (458, 458)
EXPECTED_DAY_CLASS_COUNTS = {
    **{f"Day{day}": {0: 36, 1: 36, 2: 22} for day in range(1, 10)},
    "Day10": {0: 0, 1: 0, 2: 22},
}


def validate_test_set_distribution(day_class_counts):
    """Verify that the released test set contains the expected 868 paired samples."""
    errors = []
    for day, expected_counts in EXPECTED_DAY_CLASS_COUNTS.items():
        observed_counts = day_class_counts.get(day, {0: 0, 1: 0, 2: 0})
        for label, expected in expected_counts.items():
            observed = observed_counts.get(label, 0)
            if observed != expected:
                errors.append(
                    f"{day}, class {label}: observed {observed}, expected {expected}"
                )

    if errors:
        raise RuntimeError(
            "The paired test-set distribution does not match the released dataset:\n"
            + "\n".join(errors)
        )


def resolve_device(device_name: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device_name.lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def evaluate(model, data_loader, device):
    """Run test-set inference with the finalized checkpoint."""
    y_true, y_prob, all_features = [], [], []
    final_weights = None

    model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            x1d, x2d, y, *_ = batch
            x1d = x1d.to(device, non_blocking=device.type == "cuda")
            x2d = x2d.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")

            logits, weights = model(x1d=x1d, x2d=x2d)
            probabilities = F.softmax(logits, dim=1)
            features = model.get_features(x1d, x2d)

            y_true.append(y.cpu().numpy())
            y_prob.append(probabilities.cpu().numpy())
            all_features.append(features.cpu().numpy())

            if final_weights is None:
                final_weights = weights

    return (
        np.concatenate(y_true, axis=0),
        np.concatenate(y_prob, axis=0),
        np.concatenate(all_features, axis=0),
        final_weights,
    )


# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# The following paths are repository-relative defaults.
#
# 1D spectral sequences:
#   <repository>/data/Spectral_Sequences/TestSet
#
# 2D recurrence plots:
#   <repository>/data/Recurrence_Plots/TestSet
#
# Released model weights:
#   <repository>/weights
#
# Evaluation outputs:
#   <repository>/results
#
# If the repository follows this structure, no path editing is required.
# Users with a different directory layout can override the defaults using:
#   --spectral-root
#   --recurrence-root
#   --weights-dir
#   --output-dir
# =============================================================================

def parse_args():
    """Parse repository paths and evaluation options."""
    repo_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Evaluate the finalized dual-representation weighted-fusion model."
    )
    parser.add_argument(
        "--spectral-root",
        type=Path,
        default=repo_dir / "data" / "Spectral_Sequences" / "TestSet",
        help="Path to the 1D spectral-sequence TestSet directory containing Day1-Day10.",
    )
    parser.add_argument(
        "--recurrence-root",
        type=Path,
        default=repo_dir / "data" / "Recurrence_Plots" / "TestSet",
        help="Path to the 2D recurrence-plot TestSet directory containing Day1-Day10.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=repo_dir / "weights",
        help="Directory containing the released .pth weight files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_dir / "results",
        help="Directory used to save evaluation figures.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, e.g. 'auto', 'cpu', or 'cuda:0'.",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP visualization when umap-learn is unavailable or not required.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Classes: {CLASSES}")

    test_dataset = PairedSpectraImageDataset(
        args.spectral_root,
        args.recurrence_root,
        days=DEFAULT_DAYS,
    )
    class_counts = {0: 0, 1: 0, 2: 0}
    day_class_counts = {
        day: {0: 0, 1: 0, 2: 0}
        for day in DEFAULT_DAYS
    }
    for _, _, label, day in test_dataset.samples:
        class_counts[label] += 1
        day_class_counts[day][label] += 1

    validate_test_set_distribution(day_class_counts)

    print(
        "Class distribution: "
        f"Female={class_counts[0]}, "
        f"Male={class_counts[1]}, "
        f"Imprinted-dead={class_counts[2]}"
    )

    for day in DEFAULT_DAYS:
        counts = day_class_counts[day]
        print(
            f"{day}: "
            f"Female={counts[0]}, "
            f"Male={counts[1]}, "
            f"Imprinted-dead={counts[2]}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Test samples: {len(test_dataset)}")

    pretrained_1d_path = args.weights_dir / "Optimal_1D-CNN_Weights.pth"
    pretrained_2d_path = args.weights_dir / "Optimal_2D-CNN_Weights.pth"
    fusion_checkpoint_path = (
        args.weights_dir / "DualRepresentation_Weighted-Fusion_Network_Weights.pth"
    )

    for required_path in (pretrained_1d_path, pretrained_2d_path, fusion_checkpoint_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required weight file not found: {required_path}")

    model = WeightedFusionNet(
        net1d_path=str(pretrained_1d_path),
        net2d_path=str(pretrained_2d_path),
        num_classes=len(CLASSES),
        dropout=0.5,
        freeze_backbone=True,
    ).to(device)

    state = load_state_dict_file(fusion_checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded dual-representation fusion checkpoint: {fusion_checkpoint_path}")

    y_true, y_prob, all_features, fusion_weights = evaluate(model, test_loader, device)
    y_pred = y_prob.argmax(axis=1)

    if fusion_weights is not None:
        print(
            "Final fusion weights: "
            f"1D={fusion_weights[0]:.4f}, 2D={fusion_weights[1]:.4f}"
        )

    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

    print(f"Number of test samples: {len(test_dataset)}")
    print("Confusion matrix:\n", matrix)
    print(f"Overall accuracy: {accuracy:.4f}")

    row_totals = matrix.sum(axis=1)
    class_accuracy = np.divide(
        np.diag(matrix),
        row_totals,
        out=np.zeros(len(CLASSES), dtype=float),
        where=row_totals != 0,
    )
    for class_name, value in zip(CLASSES, class_accuracy):
        print(f"{class_name} accuracy: {value:.4f}")

    plot_cm(
        matrix,
        CLASSES_CM,
        save_path=args.output_dir / "DualRepresentationWeightedFusion_ConfusionMatrix.png",
    )

    roc_auc = plot_roc_curve(
        y_true,
        y_prob,
        CLASSES_ROC,
        save_path=args.output_dir / "DualRepresentationWeightedFusion_ROC.png",
    )
    if not args.skip_umap:
        embedding = plot_umap(
            all_features,
            y_true,
            CLASSES_UMAP,
            save_path=args.output_dir / "DualRepresentationWeightedFusion_UMAP.png",
        )

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
