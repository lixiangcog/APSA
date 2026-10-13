"""
Shape-Prompt-Learning: The proposed SP-TTA for MRI-Prostate dataset in online steup.
"""
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from utils.init import init_random_and_cudnn, Recorder
from time import time
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from utils.file_utils import *
import argparse
from test_time_training_online.base_tta import BaseAdapter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import scipy.ndimage as ndi
# from batchgenerators.utilities.file_and_folder_operations import *
from models import *
from utils.metrics.dice import  get_dice_assd_hd, get_dice_threshold
from utils.tools import AverageMeter, save_np2nii, save_tensor2nii
from utils.metrics.dice import get_hard_dice
from datasets.utils.normalize import normalize_image
import torch.nn.functional as F
import models.moment_tta.losses as moment_tta_losses
from models.moment_tta.bounds import *
from loss_functions.bn_loss import bn_loss
from models.PromptTTA.ours_ema import *

def _keep_largest_connected_component(x: np.ndarray, min_size: int = 0) -> np.ndarray:
    """
    Keep the largest connected component of a 2D/3D binary mask.
    This reduces tiny distant false positives that can dominate HD95.
    """
    arr = (np.asarray(x) > 0).astype(np.uint8)
    if arr.sum() == 0:
        return arr
    if arr.ndim == 2:
        structure = np.ones((3, 3), dtype=np.uint8)
    elif arr.ndim == 3:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
    else:
        # fallback: treat everything as foreground (shouldn't happen here)
        return arr

    lab, n = ndi.label(arr, structure=structure)
    if n <= 1:
        return arr
    counts = np.bincount(lab.reshape(-1))
    counts[0] = 0  # background
    k = int(np.argmax(counts))
    if min_size and int(min_size) > 0:
        if int(counts[k]) < int(min_size):
            return np.zeros_like(arr, dtype=np.uint8)
    return (lab == k).astype(np.uint8)


def _keep_connected_component_by_prior(
    mask_zyx: np.ndarray,
    prior_hw: Optional[np.ndarray],
    prior_slice_idx: Optional[int] = None,
    slice_sigma: float = 0.0,
    slice_radius: int = 0,
    min_size: int = 0,
    min_score: float = 0.0,
) -> np.ndarray:
    """
    Keep the 3D connected component that best aligns with a 2D prior (broadcast across slices).

    More robust than "largest CC" when a large false-positive component appears far away, which can
    inflate HD95 dramatically.
    """
    arr = (np.asarray(mask_zyx) > 0).astype(np.uint8)
    if arr.sum() == 0:
        return arr
    if prior_hw is None:
        return _keep_largest_connected_component(arr, min_size=min_size)

    prior = np.asarray(prior_hw, dtype=np.float32)
    if prior.ndim != 2:
        return _keep_largest_connected_component(arr, min_size=min_size)
    prior = np.clip(prior, 0.0, 1.0)
    if float(prior.max()) > 1e-6:
        prior = prior / float(prior.max())

    z = int(arr.shape[0])
    gate = np.ones((z,), dtype=np.float32)
    if prior_slice_idx is not None:
        mu = float(prior_slice_idx)
        sig = float(slice_sigma)
        if sig and sig > 0:
            zz = np.arange(z, dtype=np.float32)
            gate = np.exp(-((zz - mu) ** 2) / (2.0 * sig * sig)).astype(np.float32)
        rad = int(slice_radius or 0)
        if rad and rad > 0:
            lo = max(0, int(round(mu)) - rad)
            hi = min(z, int(round(mu)) + rad + 1)
            hard = np.zeros((z,), dtype=np.float32)
            hard[lo:hi] = 1.0
            gate *= hard
        if float(gate.max()) > 1e-6:
            gate = gate / float(gate.max())

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lab, n = ndi.label(arr, structure=structure)
    if n <= 1:
        return arr

    best_k = None
    best_score = -1e18
    best_size = -1
    for k in range(1, int(n) + 1):
        comp = (lab == k)
        size = int(comp.sum())
        if min_size and size < int(min_size):
            continue
        sc = 0.0
        for zi in range(z):
            g = float(gate[zi])
            if g <= 0:
                continue
            sl = comp[zi]
            if not sl.any():
                continue
            sc += g * float((prior * sl.astype(np.float32)).sum())
        if sc > best_score or (sc == best_score and size > best_size):
            best_score = sc
            best_size = size
            best_k = k

    if best_k is None or float(best_score) < float(min_score):
        return _keep_largest_connected_component(arr, min_size=min_size)
    return (lab == int(best_k)).astype(np.uint8)
import pandas as pd
import time as time_module


def entropy_loss_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    # Repo's original entropy implementation (used by prior PASS runs here).
    p = torch.sigmoid(logits).max(dim=1)[0]
    y1 = -1.0 * torch.sum(p * torch.log(p + 1e-6), dim=0)
    return torch.mean(y1)


def entropy_loss_sigmoid_per_sample(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample version of `entropy_loss_sigmoid`, returns [B]."""
    p = torch.sigmoid(logits).max(dim=1)[0]  # [B,H,W]
    ent = -(p * torch.log(p + 1e-6))  # [B,H,W]
    return ent.mean(dim=(1, 2))


def entropy_loss_sigmoid_weighted(
    logits: torch.Tensor,
    weight_hw: torch.Tensor,
    weight_gamma: float = 1.0,
    weight_min: float = 0.0,
) -> torch.Tensor:
    """
    Weighted entropy loss in the same style as `entropy_loss_sigmoid`, using a CLIP prior
    (or any spatial prior) as per-pixel weights.

    logits: [B, 1, H, W]
    weight_hw: [H, W] or [1, 1, H, W]
    """
    # Match repo's original entropy (which uses only -p*log(p) over the max-class prob).
    p = torch.sigmoid(logits).max(dim=1)[0]  # [B,H,W]
    ent = -(p * torch.log(p + 1e-6))  # [B,H,W]
    w = weight_hw
    if w.dim() == 2:
        w = w.view(1, w.shape[0], w.shape[1])
    elif w.dim() == 4:
        # [1,1,H,W] -> [1,H,W]
        w = w[:, 0, :, :]
    w = w.to(device=logits.device, dtype=ent.dtype)
    w = torch.clamp(w, 0.0, 1.0)
    g = float(weight_gamma)
    if g <= 0:
        g = 1.0
    w = w.pow(g)
    wm = float(weight_min)
    if wm > 0:
        w = w * (1.0 - wm) + wm
    # Match original reduction: sum over batch, then mean over spatial.
    # This keeps the loss scale comparable to `entropy_loss_sigmoid` (which sums over batch).
    if w.shape[0] == 1 and ent.shape[0] > 1:
        w = w.repeat(ent.shape[0], 1, 1)
    weighted = (ent * w).sum(dim=0)  # [H,W]
    return weighted.mean()


def entropy_loss_sigmoid_weighted_per_sample(
    logits: torch.Tensor,
    weight_hw: torch.Tensor,
    weight_gamma: float = 1.0,
    weight_min: float = 0.0,
) -> torch.Tensor:
    """Per-sample version of `entropy_loss_sigmoid_weighted`, returns [B]."""
    p = torch.sigmoid(logits).max(dim=1)[0]  # [B,H,W]
    ent = -(p * torch.log(p + 1e-6))  # [B,H,W]
    w = weight_hw
    if w.dim() == 2:
        w = w.view(1, w.shape[0], w.shape[1])
    elif w.dim() == 4:
        w = w[:, 0, :, :]
    w = w.to(device=logits.device, dtype=ent.dtype)
    w = torch.clamp(w, 0.0, 1.0)
    g = float(weight_gamma)
    if g <= 0:
        g = 1.0
    w = w.pow(g)
    wm = float(weight_min)
    if wm > 0:
        w = w * (1.0 - wm) + wm
    if w.shape[0] == 1 and ent.shape[0] > 1:
        w = w.repeat(ent.shape[0], 1, 1)
    wsum = w.sum(dim=(1, 2)).clamp_min(1e-6)
    return (ent * w).sum(dim=(1, 2)) / wsum


def bce_with_logits_weighted(
    logits: torch.Tensor,
    target_hw: torch.Tensor,
    gamma: float = 1.0,
    weight_min: float = 0.0,
) -> torch.Tensor:
    """
    Weighted BCE with logits against a soft target map.
    logits: [B,1,H,W]
    target_hw: [H,W] or [1,1,H,W] in [0,1]
    """
    t = target_hw
    if t.dim() == 2:
        t = t.view(1, 1, t.shape[0], t.shape[1])
    t = t.to(device=logits.device, dtype=logits.dtype)
    t = torch.clamp(t, 0.0, 1.0)
    if t.shape[0] == 1 and logits.shape[0] > 1:
        t = t.repeat(logits.shape[0], 1, 1, 1)
    loss = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    w = torch.clamp(t, 0.0, 1.0)
    g = float(gamma)
    if g <= 0:
        g = 1.0
    w = w.pow(g)
    wm = float(weight_min)
    if wm > 0:
        w = w * (1.0 - wm) + wm
    return (loss * w).sum() / w.sum().clamp_min(1e-6)


def _rand_like(x: torch.Tensor) -> torch.Tensor:
    # TorchScript-friendly rand_like equivalent that supports older torch versions.
    return torch.rand(x.shape, device=x.device, dtype=x.dtype)


def _argmax_2d_xy(t_hw: torch.Tensor) -> Tuple[float, float]:
    """Return (x, y) of argmax in a [H,W] tensor (float coords)."""
    h, w = int(t_hw.shape[-2]), int(t_hw.shape[-1])
    idx = int(torch.argmax(t_hw).item())
    y = float(idx // w)
    x = float(idx % w)
    return x, y


def _centroid_xy_from_mask(mask_hw: torch.Tensor) -> Optional[Tuple[float, float]]:
    """Centroid (x,y) for a 2D binary mask tensor; returns None if empty."""
    m = (mask_hw > 0).to(dtype=torch.float32)
    s = float(m.sum().item())
    if s <= 1e-6:
        return None
    ys, xs = torch.where(m > 0)
    if ys.numel() == 0:
        return None
    return float(xs.to(dtype=torch.float32).mean().item()), float(ys.to(dtype=torch.float32).mean().item())


def _centroid_xy_from_logits_3d(logits_zyx: torch.Tensor, thr: float = 0.5) -> Optional[Tuple[float, float]]:
    """Centroid (x,y) on a 3D stack of logits [Z,1,H,W] or [Z,H,W] after sigmoid+thr."""
    if logits_zyx.dim() == 4:
        x = torch.sigmoid(logits_zyx)[:, 0, :, :]
    elif logits_zyx.dim() == 3:
        x = torch.sigmoid(logits_zyx)
    else:
        return None
    m = (x >= float(thr)).to(dtype=torch.float32)  # [Z,H,W]
    if float(m.sum().item()) <= 1e-6:
        return None
    # collapse z for x/y centroid
    m2 = m.sum(dim=0)  # [H,W]
    ys, xs = torch.where(m2 > 0)
    if ys.numel() == 0:
        return None
    return float(xs.to(dtype=torch.float32).mean().item()), float(ys.to(dtype=torch.float32).mean().item())


def photometric_augment_2d(
    x: torch.Tensor,
    noise_std: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 0.0,
    gamma: float = 0.0,
    p: float = 1.0,
) -> torch.Tensor:
    """
    Photometric-only augmentation for 2D slices.
    Keeps geometry unchanged, so no inverse warp is needed for teacher-student consistency.
    x: [B,C,H,W]
    """
    if float(p) < 1.0:
        if float(_rand_like(x[:1, :1, :1, :1]).item()) > float(p):
            return x
    y = x
    b = float(brightness)
    if b > 0:
        shift = (2.0 * _rand_like(y[:, :1, :1, :1]) - 1.0) * b
        y = y + shift
    c = float(contrast)
    if c > 0:
        scale = 1.0 + (2.0 * _rand_like(y[:, :1, :1, :1]) - 1.0) * c
        mean = y.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * scale + mean
    g = float(gamma)
    if g > 0:
        exp = 1.0 + (2.0 * _rand_like(y[:, :1, :1, :1]) - 1.0) * g
        y01 = torch.clamp(y, 0.0, 1.0)
        y = torch.pow(y01, exp)
    n = float(noise_std)
    if n > 0:
        y = y + torch.randn_like(y) * n
    return y


def mse_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weight_hw: torch.Tensor = None,
    weight_gamma: float = 1.0,
    weight_min: float = 0.0,
) -> torch.Tensor:
    """
    MSE consistency on sigmoid probabilities with optional spatial weights.
    student_logits/teacher_logits: [B,1,H,W]
    weight_hw: [B,1,H,W] or [H,W]
    """
    ps = torch.sigmoid(student_logits)
    pt = torch.sigmoid(teacher_logits)
    diff2 = (ps - pt).pow(2.0)
    if weight_hw is None:
        return diff2.mean()
    w = weight_hw
    if w.dim() == 2:
        w = w.view(1, 1, w.shape[0], w.shape[1])
    elif w.dim() == 4 and w.shape[1] != 1:
        w = w[:, :1, :, :]
    w = w.to(device=diff2.device, dtype=diff2.dtype)
    w = torch.clamp(w, 0.0, 1.0)
    g = float(weight_gamma)
    if g <= 0:
        g = 1.0
    w = w.pow(g)
    wm = float(weight_min)
    if wm > 0:
        w = w * (1.0 - wm) + wm
    wsum = w.sum().clamp_min(1e-6)
    return (diff2 * w).sum() / wsum


def teacher_confidence_from_logits(teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Confidence proxy in [0,1], per-sample.
    Use mean distance from 0.5 of sigmoid probs (higher => more confident).
    """
    p = torch.sigmoid(teacher_logits)
    conf = (p - 0.5).abs() * 2.0
    return conf.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)


def sigmoid_gate(x: torch.Tensor, thr: float = 0.0, temp: float = 0.1, floor: float = 0.0) -> torch.Tensor:
    """
    Map a scalar tensor to [0,1] using a sigmoid gate around `thr`.
    floor in [0,1): output = floor + (1-floor)*gate
    """
    t = float(temp)
    if t <= 1e-8:
        g = (x > float(thr)).to(dtype=x.dtype)
    else:
        g = torch.sigmoid((x - float(thr)) / t).to(dtype=x.dtype)
    f = float(floor)
    f = max(0.0, min(0.95, f))
    return g * (1.0 - f) + f


def load_case_clip_prior(
    clip_prior_root: str,
    domain: str,
    case_id: str,
    kind: str = "clip",
) -> Tuple[torch.Tensor, Optional[int]]:
    """
    Load precomputed CLIP prior map for a case as:
      - torch.Tensor [H, W] in [0, 1]
      - optional slice_idx (int) if present in the npz

    If requested kind is missing and kind is 'ellip', fall back to the corresponding 'clip' prior.
    """
    import numpy as np

    fname = f"{case_id}_{kind}.npz"
    path = os.path.join(clip_prior_root, domain, fname)
    if not os.path.exists(path):
        # fallback: some runs store as "Segmentation_clip"
        alt = os.path.join(clip_prior_root, domain, f"{case_id}_Segmentation_{kind}.npz")
        if os.path.exists(alt):
            path = alt
        else:
            # Unified robustness: ellip prior can be derived from clip prior at runtime.
            if str(kind).lower().strip() in ("ellip", "ellipse", "blob", "pointblob"):
                fname2 = f"{case_id}_clip.npz"
                path2 = os.path.join(clip_prior_root, domain, fname2)
                if not os.path.exists(path2):
                    alt2 = os.path.join(clip_prior_root, domain, f"{case_id}_Segmentation_clip.npz")
                    if os.path.exists(alt2):
                        path2 = alt2
                if os.path.exists(path2):
                    path = path2
                else:
                    raise FileNotFoundError(f"Missing CLIP prior: {path} (and fallback clip prior also missing)")
            else:
                raise FileNotFoundError(f"Missing CLIP prior: {path}")

    z = np.load(path)
    if "soft_map" not in z:
        raise ValueError(f"Bad prior file (missing soft_map): {path}, keys={list(z.keys())}")
    m = z["soft_map"].astype(np.float32)
    m = np.clip(m, 0.0, 1.0)
    slice_idx = None
    if "slice_idx" in z:
        try:
            slice_idx = int(z["slice_idx"].item())
        except Exception:
            slice_idx = None
    return torch.from_numpy(m), slice_idx


def clip_alignment_score_sigmoid(logits: torch.Tensor, weight_hw: torch.Tensor) -> torch.Tensor:
    """
    Score how well a prediction aligns with a spatial prior.
    Higher is better: encourages probability mass inside prior and discourages outside.
    Returns a scalar tensor.
    """
    p = torch.sigmoid(logits)
    w = weight_hw
    if w.dim() == 2:
        w = w.view(1, 1, w.shape[0], w.shape[1])
    w = w.to(device=logits.device, dtype=p.dtype)
    w = torch.clamp(w, 0.0, 1.0)
    mass_in = (p * w).mean()
    mass_out = (p * (1.0 - w)).mean()
    return mass_in - mass_out


def clip_alignment_score_sigmoid_per_sample(logits: torch.Tensor, weight_hw: torch.Tensor) -> torch.Tensor:
    """Per-sample alignment score, returns [B]."""
    p = torch.sigmoid(logits)  # [B,1,H,W]
    w = weight_hw
    if w.dim() == 2:
        w = w.view(1, 1, w.shape[0], w.shape[1])
    elif w.dim() == 3:
        w = w.view(w.shape[0], 1, w.shape[1], w.shape[2])
    w = w.to(device=logits.device, dtype=p.dtype)
    w = torch.clamp(w, 0.0, 1.0)
    mass_in = (p * w).mean(dim=(1, 2, 3))
    mass_out = (p * (1.0 - w)).mean(dim=(1, 2, 3))
    return mass_in - mass_out


def _gaussian_blur_2d_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Gaussian blur for 2D maps in torch.
    x: [B,1,H,W] float
    """
    sigma = float(sigma)
    if sigma <= 0:
        return x
    # kernel size ~ 6*sigma, odd
    k = int(max(3, round(6.0 * sigma)))
    if k % 2 == 0:
        k += 1
    half = k // 2
    device = x.device
    dtype = x.dtype
    t = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(t * t) / (2.0 * sigma * sigma))
    g = g / g.sum().clamp_min(1e-8)
    g1 = g.view(1, 1, 1, k)
    g2 = g.view(1, 1, k, 1)
    x = F.conv2d(x, g1, padding=(0, half))
    x = F.conv2d(x, g2, padding=(half, 0))
    return x


def _normalize01_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn).clamp_min(eps)


def _saliency_confidence(sal_b1hw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Confidence score in [0,1] for a soft saliency map based on peakiness.
    Higher when the map is concentrated (low entropy + high max mass).

    sal_b1hw: [B,1,H,W] or [B,H,W] (values in [0,1] recommended).
    Returns: [B] float tensor in [0,1].
    """
    s = sal_b1hw
    if s is None:
        return None
    if s.dim() == 2:
        s = s.view(1, 1, s.shape[0], s.shape[1])
    elif s.dim() == 3:
        s = s.unsqueeze(1)
    s = torch.clamp(s.to(dtype=torch.float32), 0.0, 1.0)
    b, _, h, w = s.shape
    # Normalize to a probability distribution over pixels.
    p = s / (s.sum(dim=(-2, -1), keepdim=True) + float(eps))
    # Normalized entropy in [0,1].
    ent = -(p * (p + float(eps)).log()).sum(dim=(-2, -1)).view(b) / float(np.log(max(2, h * w)))
    peak_ent = (1.0 - ent).clamp(0.0, 1.0)
    # Max probability mass relative to uniform baseline.
    pmax = p.amax(dim=(-2, -1)).view(b)
    sharp = ((pmax * float(h * w)) - 1.0) / max(1.0, float(h * w - 1))
    sharp = sharp.clamp(0.0, 1.0)
    return (0.5 * peak_ent + 0.5 * sharp).clamp(0.0, 1.0)


def _make_generic_prior_map(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    cx_n: float = 0.50,
    cy_n: float = 0.72,
    sx_n: float = 0.16,
    sy_n: float = 0.14,
) -> torch.Tensor:
    """A broad anatomical-ish prior blob in [0,1], shape [1,1,H,W]."""
    cx = float(cx_n) * float(w)
    cy = float(cy_n) * float(h)
    sx = max(1.0, float(sx_n) * float(w))
    sy = max(1.0, float(sy_n) * float(h))
    yy = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1)
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w)
    blob = torch.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy))).view(1, 1, h, w)
    blob = _normalize01_torch(blob.to(dtype=torch.float32))
    return blob.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _blend_prior_with_generic(
    prior_b1hw: torch.Tensor,
    q_b: Optional[torch.Tensor],
    generic_11hw: torch.Tensor,
) -> torch.Tensor:
    """
    Blend a per-sample prior with a generic blob based on confidence q in [0,1].
    Returns a normalized map in [0,1], same shape as prior_b1hw.
    """
    if prior_b1hw is None:
        return None
    if q_b is None:
        return prior_b1hw
    q = q_b.view(-1, 1, 1, 1).to(device=prior_b1hw.device, dtype=prior_b1hw.dtype).clamp(0.0, 1.0)
    g = generic_11hw
    if g.shape[0] == 1 and prior_b1hw.shape[0] > 1:
        g = g.repeat(prior_b1hw.shape[0], 1, 1, 1)
    out = q * prior_b1hw + (1.0 - q) * g.to(device=prior_b1hw.device, dtype=prior_b1hw.dtype)
    return _normalize01_torch(out)


def saliency_to_elliptic_blob(
    sal: torch.Tensor,
    sigma_y: float,
    sigma_x: float,
    power: float = 1.0,
    sym: bool = False,
    smooth_sigma: float = 0.0,
) -> torch.Tensor:
    """
    Convert a saliency map to a coarse, smooth elliptical blob centered at its peak.

    This is meant as a robust *localization cue* (\"prostate is around here\") rather than a segmentation target.
    sal: [B,1,H,W] or [B,H,W] in [0,1]
    Returns: [B,1,H,W] in [0,1]
    """
    if sal.dim() == 3:
        sal = sal.unsqueeze(1)
    sal = sal.to(dtype=torch.float32)
    sal = torch.clamp(sal, 0.0, 1.0)
    if sym:
        sal = 0.5 * (sal + torch.flip(sal, dims=[-1]))
    if smooth_sigma and float(smooth_sigma) > 0:
        sal = _gaussian_blur_2d_torch(sal, sigma=float(smooth_sigma))

    b, _, h, w = sal.shape
    idx = sal.view(b, -1).argmax(dim=1)
    cy = torch.div(idx, w, rounding_mode="trunc").to(dtype=torch.float32)
    cx = (idx - (cy.to(dtype=idx.dtype) * w)).to(dtype=torch.float32)

    yy = torch.arange(h, device=sal.device, dtype=torch.float32).view(1, h, 1)
    xx = torch.arange(w, device=sal.device, dtype=torch.float32).view(1, 1, w)
    cy = cy.view(b, 1, 1)
    cx = cx.view(b, 1, 1)

    sy = max(1.0, float(sigma_y))
    sx = max(1.0, float(sigma_x))
    blob = torch.exp(-(((yy - cy) ** 2) / (2.0 * sy * sy) + ((xx - cx) ** 2) / (2.0 * sx * sx)))
    blob = blob.view(b, 1, h, w)
    blob = _normalize01_torch(blob)

    p = float(power)
    if p and p != 1.0:
        blob = blob.pow(max(0.1, p))
        blob = _normalize01_torch(blob)
    return blob.clamp(0.0, 1.0)


def _pred_area_ratio_from_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Foreground area ratio proxy in [0,1], per-sample. logits: [B,1,H,W]."""
    p = torch.sigmoid(logits)
    m = (p >= float(thr)).to(dtype=torch.float32)
    return m.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)


def _compute_case_reliability_score(
    logits: torch.Tensor,
    clip_weight: Optional[torch.Tensor] = None,
    clip_q: Optional[torch.Tensor] = None,
    area_min: float = 0.002,
    area_max: float = 0.10,
    center_max: float = 0.30,
) -> dict:
    """
    Compute per-slice reliability signals for a candidate logit set.

    Returns a dict with per-slice scores and case-level aggregates.
    All signals are computed WITHOUT ground truth labels.
    """
    info = {}
    n = int(logits.shape[0])

    # 1. Area ratio (foreground fraction)
    area = _pred_area_ratio_from_logits(logits, thr=0.5)  # [N]
    info["area"] = area.detach().cpu()
    info["area_mean"] = float(area.mean().item())
    # Use mean area check: per-slice min/max is too strict (edge slices naturally have no prostate).
    info["area_ok"] = bool(area_min <= info["area_mean"] <= area_max)

    # 2. Entropy (lower = more confident)
    if clip_weight is not None:
        w = clip_weight
        if w.shape[0] != n:
            if w.shape[0] == 1:
                w = w.repeat(n, 1, 1, 1)
        ent = entropy_loss_sigmoid_weighted_per_sample(logits, w)  # [N]
    else:
        ent = entropy_loss_sigmoid_per_sample(logits)  # [N]
    info["entropy"] = ent.detach().cpu()
    info["entropy_mean"] = float(ent.mean().item())

    # 3. CLIP alignment (higher = better agreement with prior)
    if clip_weight is not None:
        w = clip_weight
        if w.shape[0] != n:
            if w.shape[0] == 1:
                w = w.repeat(n, 1, 1, 1)
        align = clip_alignment_score_sigmoid_per_sample(logits, w)  # [N]
        info["clip_align"] = align.detach().cpu()
        info["clip_align_mean"] = float(align.mean().item())

        # 4. Center distance from prior peak
        pred_c = _centroid_xy_from_logits_3d(logits, thr=0.5)
        if pred_c is not None:
            pm = w.detach().mean(dim=0)[0]
            qx, qy = _argmax_2d_xy(pm)
            hh = float(pm.shape[-2])
            ww = float(pm.shape[-1])
            dx = (pred_c[0] - qx) / max(1.0, ww)
            dy = (pred_c[1] - qy) / max(1.0, hh)
            dist = float((dx * dx + dy * dy) ** 0.5)
            info["center_dist"] = dist
            info["center_ok"] = bool(dist <= center_max)
        else:
            info["center_dist"] = None
            info["center_ok"] = True  # no prediction, can't penalize
    else:
        info["clip_align"] = None
        info["clip_align_mean"] = None
        info["center_dist"] = None
        info["center_ok"] = True

    # 5. CLIP prior confidence
    if clip_q is not None:
        info["prior_q_mean"] = float(clip_q.mean().detach().cpu().item())
    else:
        info["prior_q_mean"] = None

    # Composite safety score (higher = better)
    score = 0.0
    if info["clip_align_mean"] is not None:
        score += info["clip_align_mean"] * 2.0  # alignment weight
    score -= info["entropy_mean"] * 5.0  # entropy penalty (typically 0.01-0.1)
    if not info["area_ok"]:
        score -= 3.0
    if info["center_dist"] is not None:
        score -= info["center_dist"] * 2.0  # center distance penalty
    if info["prior_q_mean"] is not None and info["prior_q_mean"] < 0.3:
        score -= 1.0 * (0.3 - info["prior_q_mean"])  # low prior confidence penalty
    info["score"] = score

    return info


def _prior_center_distance_from_logits(
    logits: torch.Tensor,
    clip_weight: Optional[torch.Tensor],
    clip_q: Optional[torch.Tensor] = None,
) -> Optional[float]:
    """Normalized 3D foreground centroid distance to the anatomical prior peak."""
    if logits is None or clip_weight is None:
        return None
    pred_c = _centroid_xy_from_logits_3d(logits, thr=0.5)
    if pred_c is None:
        return None
    w_case = clip_weight.detach()
    if w_case.dim() == 2:
        prior_map = w_case
    elif w_case.dim() == 4:
        if clip_q is not None and clip_q.numel() == w_case.shape[0]:
            q = clip_q.detach().clamp(0.0, 1.0).view(-1, 1, 1, 1)
            prior_map = (w_case * q).sum(dim=0)[0] / q.sum().clamp_min(1e-6)
        else:
            prior_map = w_case.mean(dim=0)[0]
    else:
        return None
    qx, qy = _argmax_2d_xy(prior_map)
    h = float(prior_map.shape[-2])
    w = float(prior_map.shape[-1])
    dx = (pred_c[0] - qx) / max(1.0, w)
    dy = (pred_c[1] - qy) / max(1.0, h)
    return float((dx * dx + dy * dy) ** 0.5)


def _safe_branch_select(
    info_src: dict,
    info_ada: dict,
    info_fin: dict,
    conservative: bool = False,
    adaptive: bool = False,
    final_min_gain: float = 0.02,
    final_bias: float = 0.03,
) -> Tuple[str, str]:
    """
    Hierarchical, domain-agnostic branch selector.

    Decision logic (no ground truth, no domain labels):
    1. Compute composite reliability score for each candidate.
    2. Guard against final degradation (entropy spike / area collapse vs adapted).
    3. When source is clearly broken, allow adapted/final takeover.
    4. Preserve adapted when it's good and final degrades.

    Returns (best_branch, reason_string).
    """
    candidates = {
        "source": info_src,
        "adapted": info_ada,
        "final": info_fin,
    }

    # --- Step 1: compute reliability score for each candidate ---
    def _rel_score(info: dict) -> float:
        s = 0.0
        if info.get("clip_align_mean") is not None:
            s += info["clip_align_mean"] * 2.0
        s -= info["entropy_mean"] * 5.0
        if not info["area_ok"]:
            s -= 3.0
        cd = info.get("center_dist")
        if cd is not None:
            s -= cd * 2.0
        if info.get("prior_q_mean") is not None and info["prior_q_mean"] < 0.3:
            s -= 1.0 * (0.3 - info["prior_q_mean"])
        return s

    scores = {k: _rel_score(v) for k, v in candidates.items()}

    # --- Step 2: final quality guard ---
    # Reject final if it clearly degrades relative to adapted
    final_blocked = False
    ada_ent = info_ada["entropy_mean"]
    fin_ent = info_fin["entropy_mean"]
    ada_align = info_ada.get("clip_align_mean")
    fin_align = info_fin.get("clip_align_mean")
    ada_area = info_ada.get("area_mean", 0.0)
    fin_area = info_fin.get("area_mean", 0.0)

    # Final entropy significantly higher than adapted -> postprocess may have hurt.
    # Use a conservative threshold: in prostate, useful LCC/post prior can slightly
    # increase entropy while still removing distant false positives.
    if fin_ent > ada_ent * 1.8 and fin_ent > ada_ent + 0.006:
        final_blocked = True
    # Final area collapsed relative to adapted (and adapted had reasonable area)
    if ada_area > 0.003 and fin_area < ada_area * 0.5:
        final_blocked = True
    # Final alignment clearly worse than adapted
    if ada_align is not None and fin_align is not None:
        if fin_align < ada_align - 0.02:
            final_blocked = True

    # --- Step 3: source viability check ---
    # If source is clearly broken, adapted/final should be allowed to take over
    src_score = scores.get("source", -999)
    ada_score = scores.get("adapted", -999)
    fin_score = scores.get("final", -999)
    src_broken = (not info_src.get("area_ok", True)) or src_score < -1.2  # source has poor reliability

    # --- Step 4: pick best ---
    # Among viable candidates, pick highest reliability score. Prefer final when
    # it is close to adapted/source because it usually encodes the safer
    # postprocessed anatomical output and reduces HD95 outliers.
    viable = {}
    if not final_blocked:
        # Default mode keeps the historical mild preference for final outputs,
        # which often reduce HD95. Conservative mode removes that bias and
        # requires a measurable reliability gain before postprocessed/final
        # predictions can override a good adapted branch.
        viable["final"] = fin_score if conservative else fin_score + float(final_bias)
    viable["adapted"] = ada_score
    # Source is a fallback, not a default winner. Give it a small penalty unless
    # both adaptation branches are geometrically unsafe.
    source_penalty = 0.04
    if not info_ada.get("area_ok", True) and not info_fin.get("area_ok", True):
        source_penalty = 0.0
    viable["source"] = src_score - source_penalty

    # If source is broken AND adapted/final are at least somewhat better, exclude source
    if src_broken:
        max_other = max(ada_score, fin_score if not final_blocked else ada_score)
        if max_other > src_score + 0.05:  # others meaningfully better
            viable.pop("source", None)

    best = max(viable, key=viable.get)

    adaptive_conservative = False
    if adaptive and not final_blocked:
        # Conservative only when the non-final branch already looks reliable and
        # final mainly changes the mask by inflating entropy/area. If source or
        # adapted is unreliable, final remains allowed to rescue the case.
        adapted_plausible = info_ada.get("area_ok", True) and ada_ent < 0.006
        source_plausible = info_src.get("area_ok", True) and info_src["entropy_mean"] < 0.006
        final_entropy_inflated = fin_ent > ada_ent + 0.0025 and fin_ent > ada_ent * 1.35
        final_area_inflated = ada_area > 0.003 and fin_area > ada_area * 1.35
        final_align_gain = 0.0
        if ada_align is not None and fin_align is not None:
            final_align_gain = fin_align - ada_align
        small_align_gain = final_align_gain < 0.004
        adaptive_conservative = (adapted_plausible or source_plausible) and (
            (final_entropy_inflated and small_align_gain) or final_area_inflated
        )

    use_conservative = conservative or adaptive_conservative

    if use_conservative and best == "final":
        non_final = {"adapted": ada_score, "source": src_score - source_penalty}
        if "source" not in viable:
            non_final.pop("source", None)
        best_non_final = max(non_final, key=non_final.get)
        best_non_final_score = non_final[best_non_final]
        # If final is only marginally better, prefer the non-final branch to
        # avoid LCC/postprocess-induced Dice drops on domains where raw/adapted
        # predictions are already anatomically plausible.
        if fin_score < best_non_final_score + float(final_min_gain):
            best = best_non_final
            reason_parts_extra = "conservative_final_rejected"
        else:
            reason_parts_extra = "conservative_final_accepted"
    else:
        reason_parts_extra = None

    reason_parts = [f"scores={ {k: round(v,3) for k,v in scores.items()} }"]
    if final_blocked:
        reason_parts.append("final_blocked")
    if src_broken:
        reason_parts.append("src_broken")
    if adaptive:
        reason_parts.append("adaptive_conservative" if adaptive_conservative else "adaptive_allow_final")
    if reason_parts_extra is not None:
        reason_parts.append(reason_parts_extra)
    reason_parts.append(f"best={best}")

    return best, ",".join(reason_parts)


def _snapshot_ema_wrapper_state(model) -> dict:
    """
    Snapshot OnlineTargetWrapper state for rollback.
    Keeps tensors on CPU to reduce GPU memory pressure.
    """
    st = {}
    st["online"] = {k: v.detach().cpu().clone() for k, v in model.online_network.state_dict().items()}
    st["target"] = {k: v.detach().cpu().clone() for k, v in model.target_network.state_dict().items()}
    st["mom_pre"] = float(getattr(model, "mom_pre", 0.1))
    return st


def _restore_ema_wrapper_state(model, snap: dict) -> None:
    """Restore OnlineTargetWrapper state from `_snapshot_ema_wrapper_state`."""
    if not snap:
        return
    model.online_network.load_state_dict(snap["online"], strict=True)
    model.target_network.load_state_dict(snap["target"], strict=True)
    try:
        model.mom_pre = float(snap.get("mom_pre", 0.1))
    except Exception:
        pass


def _simple_mri_to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Best-effort per-slice normalization to [0,1] without expensive percentiles.
    Handles both [0,1] and roughly z-scored inputs.
    x: [B,1,H,W] or [B,3,H,W]
    """
    if x.dim() == 4 and x.shape[1] == 3:
        x1 = x[:, :1]
    elif x.dim() == 4 and x.shape[1] == 1:
        x1 = x
    else:
        raise ValueError(f"Unexpected shape for _simple_mri_to_01: {tuple(x.shape)}")

    mn = x1.amin(dim=(-2, -1), keepdim=True)
    mx = x1.amax(dim=(-2, -1), keepdim=True)
    # If already looks like [0,1], keep. Else assume z-score-ish and map [-3,3] -> [0,1].
    looks_01 = (mn >= -0.05).all() and (mx <= 1.05).all()
    if looks_01:
        return x1.clamp(0.0, 1.0)
    x1 = x1.clamp(-3.0, 3.0)
    return ((x1 + 3.0) / 6.0).clamp(0.0, 1.0)


class BioMedCLIPSaliency:
    """
    On-the-fly BioMedCLIP patch-token similarity saliency (per slice).
    Designed to provide a coarse localization cue, not a segmentation mask.
    """

    def __init__(
        self,
        model_root: str,
        device: torch.device,
        prompts: list,
        softmax_temp: float = 0.06,
        patch_sigma: float = 3.5,
        blur_sigma: float = 2.0,
        roi: tuple = None,
        roi_feather: float = 0.10,
    ):
        import json
        from open_clip.factory import load_checkpoint
        from open_clip.model import CustomTextCLIP
        from open_clip.tokenizer import HFTokenizer

        self.device = device
        self.softmax_temp = float(softmax_temp)
        self.patch_sigma = float(patch_sigma)
        self.blur_sigma = float(blur_sigma)
        self.roi = roi
        self.roi_feather = float(roi_feather)

        cfg_path = os.path.join(model_root, "open_clip_config.json")
        ckpt_path = os.path.join(model_root, "open_clip_pytorch_model.bin")
        bert_dir = os.path.join(model_root, "biomedbert")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing BioMedCLIP config: {cfg_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing BioMedCLIP ckpt: {ckpt_path}")
        if not os.path.isdir(bert_dir):
            raise FileNotFoundError(f"Missing biomedbert dir: {bert_dir}")

        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        model_cfg = cfg["model_cfg"]
        preprocess_cfg = cfg.get("preprocess_cfg", {})
        self.mean = torch.tensor(preprocess_cfg.get("mean", [0.48145466, 0.4578275, 0.40821073]), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(preprocess_cfg.get("std", [0.26862954, 0.26130258, 0.27577711]), device=device).view(1, 3, 1, 1)
        self.image_size = int(model_cfg["vision_cfg"].get("image_size", 224))
        self.context_length = int(model_cfg["text_cfg"].get("context_length", 256))

        # Force local HF model/tokenizer paths.
        model_cfg["text_cfg"]["hf_model_name"] = bert_dir
        model_cfg["text_cfg"]["hf_tokenizer_name"] = bert_dir

        model = CustomTextCLIP(**model_cfg).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            load_checkpoint(model, ckpt_path, strict=True)
        except Exception:
            load_checkpoint(model, ckpt_path, strict=False)

        tokenizer = HFTokenizer(tokenizer_name=bert_dir, context_length=self.context_length)
        self.model = model
        self.tokenizer = tokenizer
        self.text_features = self._encode_text_mean(prompts)

    def _encode_text_mean(self, prompts: list) -> torch.Tensor:
        text = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            t = self.model.encode_text(text)
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
            t = t.mean(dim=0, keepdim=True)
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        return t

    def _roi_crop(self, x01_1ch: torch.Tensor):
        if self.roi is None:
            return x01_1ch, None
        y1, y2, x1, x2 = [float(v) for v in self.roi]
        b, _, h, w = x01_1ch.shape
        yy1 = int(round(h * y1))
        yy2 = int(round(h * y2))
        xx1 = int(round(w * x1))
        xx2 = int(round(w * x2))
        yy1 = max(0, min(h, yy1))
        yy2 = max(0, min(h, yy2))
        xx1 = max(0, min(w, xx1))
        xx2 = max(0, min(w, xx2))
        if yy2 <= yy1 or xx2 <= xx1:
            return x01_1ch, None
        crop = x01_1ch[:, :, yy1:yy2, xx1:xx2]
        meta = (yy1, xx1, yy2 - yy1, xx2 - xx1, h, w)
        return crop, meta

    def _roi_paste_back(self, sal_crop: torch.Tensor, meta):
        yy1, xx1, hh, ww, h, w = meta
        out = torch.zeros((sal_crop.shape[0], 1, h, w), device=sal_crop.device, dtype=sal_crop.dtype)
        out[:, :, yy1:yy1 + hh, xx1:xx1 + ww] = sal_crop
        if self.roi_feather > 1e-6:
            # Feather ROI edges with a separable cosine window.
            a = max(0.0, min(1.0, float(self.roi_feather)))
            wy = torch.ones((hh,), device=sal_crop.device, dtype=sal_crop.dtype)
            wx = torch.ones((ww,), device=sal_crop.device, dtype=sal_crop.dtype)
            edge_y = int(max(1, round(a * hh * 0.5)))
            edge_x = int(max(1, round(a * ww * 0.5)))
            if edge_y > 1:
                t = torch.linspace(0, 1, edge_y, device=sal_crop.device, dtype=sal_crop.dtype)
                ramp = 0.5 * (1.0 - torch.cos(np.pi * t))
                wy[:edge_y] = ramp
                wy[-edge_y:] = torch.flip(ramp, dims=[0])
            if edge_x > 1:
                t = torch.linspace(0, 1, edge_x, device=sal_crop.device, dtype=sal_crop.dtype)
                ramp = 0.5 * (1.0 - torch.cos(np.pi * t))
                wx[:edge_x] = ramp
                wx[-edge_x:] = torch.flip(ramp, dims=[0])
            win = (wy.view(1, 1, hh, 1) * wx.view(1, 1, 1, ww)).clamp(0.0, 1.0)
            out[:, :, yy1:yy1 + hh, xx1:xx1 + ww] *= win
        return out

    @torch.no_grad()
    def saliency(self, x_3ch: torch.Tensor) -> torch.Tensor:
        """
        x_3ch: [B,3,H,W] (any range; will be normalized to [0,1] internally)
        Returns: [B,1,H,W] in [0,1]
        """
        b, c, h, w = x_3ch.shape
        x01 = _simple_mri_to_01(x_3ch)  # [B,1,H,W]
        x01, meta = self._roi_crop(x01)
        hh, ww = x01.shape[-2:]
        x3 = x01.repeat(1, 3, 1, 1)
        x3 = F.interpolate(x3, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x_norm = (x3 - self.mean) / self.std

        feat = self.model.visual.trunk.forward_features(x_norm)  # [B, 1+N, C]
        if feat.dim() != 3 or feat.shape[1] <= 2:
            out = torch.zeros((b, 1, h, w), device=self.device, dtype=torch.float32)
            return out
        tokens = feat[:, 1:, :]  # [B, N, C]
        proj = self.model.visual.head.proj(tokens)  # [B, N, D]
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
        tv = self.text_features / (self.text_features.norm(dim=-1, keepdim=True) + 1e-8)  # [1,D]
        sim = (proj * tv.unsqueeze(1)).sum(dim=-1)  # [B,N]
        sim = torch.clamp(sim, min=0.0)

        n = int(sim.shape[1])
        side = int(round(np.sqrt(n)))
        if side * side != n:
            # fallback: make a 1xN strip (rare)
            patch = sim.view(b, 1, 1, n)
        else:
            patch = sim.view(b, 1, side, side)

        # Softmax over patches makes a compact blob; temperature controls sharpness.
        t = self.softmax_temp
        if t and t > 0:
            z = patch / t
            z = z - z.amax(dim=(-2, -1), keepdim=True)
            p = torch.exp(z)
            p = p / p.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            patch = _normalize01_torch(p)
        else:
            patch = _normalize01_torch(patch)

        # Upsample to image space and lightly blur to avoid dotty look.
        up = F.interpolate(patch, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if self.patch_sigma > 0:
            up = _gaussian_blur_2d_torch(up, sigma=self.patch_sigma)
        up = F.interpolate(up, size=(hh, ww), mode="bilinear", align_corners=False)
        if self.blur_sigma > 0:
            up = _gaussian_blur_2d_torch(up, sigma=self.blur_sigma)
        up = _normalize01_torch(up).clamp(0.0, 1.0)
        if meta is not None:
            up = self._roi_paste_back(up, meta)
        return up


class Outer_Adapter(BaseAdapter):
    def __init__(self, args):
        super(Outer_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.all_steps = 0
        self._clip_engine = None
        self._clip_slice_sigma_eff_last = None
        self._clip_prior_slice_idx_last = None

    def _get_clip_engine(self):
        if self._clip_engine is not None:
            return self._clip_engine
        prompts = getattr(self.args, "clip_prompts", None)
        if not prompts:
            prompts = [
                "Prostate gland.",
                "Anatomical location: prostate gland below the bladder.",
                "Dark hypointense round structure.",
            ]
        roi = getattr(self.args, "clip_roi", None)
        if roi is not None and len(roi) == 4:
            roi = tuple(float(v) for v in roi)
        else:
            roi = None
        self._clip_engine = BioMedCLIPSaliency(
            model_root=getattr(self.args, "clip_model_root", "."),
            device=self.device,
            prompts=list(prompts),
            softmax_temp=float(getattr(self.args, "clip_softmax_temp", 0.06)),
            patch_sigma=float(getattr(self.args, "clip_patch_sigma", 3.5)),
            blur_sigma=float(getattr(self.args, "clip_blur_sigma", 2.0)),
            roi=roi,
            roi_feather=float(getattr(self.args, "clip_roi_feather", 0.10)),
        )
        return self._clip_engine

    def do_prostate_seg_tta_per_case(self, data, clip_prior_hw=None, clip_prior_slice_idx: Optional[int] = None):
        """
        tta for one case:BATCH 2d slices
        """
        num_samples = data.shape[0]
        num_batches = num_samples // self.args.batch_size
        self._clip_slice_sigma_eff_last = None
        self._clip_prior_slice_idx_last = clip_prior_slice_idx if clip_prior_slice_idx is not None else None

        # Optional: case-wise rollback. Snapshot current student/teacher weights and measure
        # a pre-adaptation proxy on the *current* online network (important for non-episodic mode).
        rollback_enabled = bool(getattr(self.args, "case_rollback", False))
        rollback_snap = None
        pre_case_logits = None
        rollback_replaced_preds = False
        if rollback_enabled:
            try:
                rollback_snap = _snapshot_ema_wrapper_state(self.model)
            except Exception as e:
                self.logger.info(f"[WARN] case_rollback snapshot failed: {e}")
                rollback_snap = None
            try:
                pre_case_logits = []
                self.model.online_network.eval()
                with torch.no_grad():
                    for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
                        batch_start = i * self.args.batch_size
                        batch_end = min((i + 1) * self.args.batch_size, num_samples)
                        batch_data = data[batch_start:batch_end, :, :, :]
                        pre_case_logits.append(self.model.online_network(batch_data))
            except Exception as e:
                self.logger.info(f"[WARN] case_rollback pre-case logits failed: {e}")
                pre_case_logits = None

        # Baseline logits before adaptation (for case-level fallback / protection and slice_idx sanity checks).
        baseline_logits = []
        with torch.no_grad():
            self.source_model.eval()
            for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
                batch_start = i * self.args.batch_size
                batch_end = min((i + 1) * self.args.batch_size, num_samples)
                batch_data = data[batch_start:batch_end, :, :, :]
                baseline_logits.append(self.source_model(batch_data))

        clip_weight_case = None
        clip_weight_case_raw = None
        clip_q_case = None
        if getattr(self.args, "clip_guided", False):
            kind = str(getattr(self.args, "clip_prior_kind", "clip")).lower().strip()
            # Prefer per-slice (per test image) CLIP guidance if enabled.
            if getattr(self.args, "clip_otf", False):
                try:
                    engine = self._get_clip_engine()
                    self.model.online_network.eval()
                    clip_weight_case = engine.saliency(data)  # [N,1,H,W]
                    clip_weight_case = _normalize01_torch(clip_weight_case.to(device=self.device, dtype=torch.float32))
                    clip_weight_case_raw = clip_weight_case.detach()
                except Exception as e:
                    self.logger.info(f"[WARN] on-the-fly CLIP saliency failed: {e}")
                    clip_weight_case = None
            # Otherwise, use offline case prior and broadcast to all slices, optionally gating by slice distance.
            if clip_weight_case is None and clip_prior_hw is not None:
                try:
                    w = clip_prior_hw
                    if w.dim() == 2:
                        w = w.view(1, 1, w.shape[0], w.shape[1])  # [1,1,H,W]
                    elif w.dim() == 3:
                        w = w.view(w.shape[0], 1, w.shape[1], w.shape[2])
                    # Broadcast to all slices [N,1,H,W]
                    n = int(data.shape[0])
                    if w.shape[0] == 1 and n > 1:
                        w = w.repeat(n, 1, 1, 1)
                    clip_weight_case = w.to(device=self.device, dtype=torch.float32)

                    # Slice-distance gating if the prior was computed on a specific slice.
                    if clip_prior_slice_idx is not None and n > 1:
                        si = int(clip_prior_slice_idx)
                        si = max(0, min(n - 1, si))
                        sigma = float(getattr(self.args, "clip_slice_sigma", 0.0))
                        radius = int(getattr(self.args, "clip_slice_radius", 0) or 0)

                        # Unified slice_idx reliability gate:
                        # If slice_idx looks mismatched vs a coarse source prediction proxy, auto-disable
                        # slice gating by inflating sigma (prevents domain collapse), but still allows
                        # slice gating to help domains where slice_idx is meaningful (e.g. I2CVB).
                        sigma_eff = sigma
                        if bool(getattr(self.args, "clip_slice_auto_gate", False)):
                            try:
                                src_logits_all = torch.cat(baseline_logits, dim=0).detach()
                                area = _pred_area_ratio_from_logits(src_logits_all, thr=0.5)  # [N]
                                if int(area.numel()) == int(n):
                                    z_pred = int(area.argmax().detach().cpu().item())
                                    dz = abs(int(z_pred) - int(si))
                                    max_delta = int(getattr(self.args, "clip_slice_auto_max_delta", 8) or 8)
                                    tau = float(getattr(self.args, "clip_slice_auto_tau", 6.0) or 6.0)
                                    tau = max(1e-3, tau)
                                    trust = float(np.exp(-float(dz) / tau))
                                    if dz > max_delta:
                                        trust = 0.0
                                    min_trust = float(getattr(self.args, "clip_slice_auto_min_trust", 0.35) or 0.35)
                                    untrusted_sigma = float(getattr(self.args, "clip_slice_sigma_untrusted", 999.0) or 999.0)
                                    untrusted_sigma = max(10.0, untrusted_sigma)

                                    if sigma_eff <= 0:
                                        sigma_eff = 3.0
                                    # Smoothly inflate sigma when trust is low; cap at untrusted_sigma.
                                    sigma_eff = min(untrusted_sigma, float(sigma_eff) / max(float(trust), 1e-3))
                                    if trust < min_trust:
                                        sigma_eff = float(untrusted_sigma)

                                    self._clip_slice_sigma_eff_last = float(sigma_eff)
                                    self._clip_prior_slice_idx_last = int(si)
                                    self.logger.info(
                                        f"[clip_slice_auto_gate] si={si} z_pred={z_pred} dz={dz} trust={trust:.3f} sigma_eff={sigma_eff:.2f}"
                                    )
                            except Exception as e:
                                self.logger.info(f"[WARN] clip_slice_auto_gate failed: {e}")

                        if sigma_eff is None or float(sigma_eff) <= 0:
                            sigma_eff = 3.0
                        dist = (torch.arange(n, device=self.device, dtype=torch.float32) - float(si)).abs()
                        gate = torch.exp(-(dist * dist) / (2.0 * float(sigma_eff) * float(sigma_eff)))
                        if radius > 0:
                            gate = gate * (dist <= float(radius)).to(dtype=gate.dtype)
                        gate = gate.view(n, 1, 1, 1)
                        clip_weight_case = clip_weight_case * gate

                    # Normalize to [0,1] per-slice for stability.
                    clip_weight_case = _normalize01_torch(clip_weight_case)
                    clip_weight_case_raw = clip_weight_case.detach()
                except Exception as e:
                    self.logger.info(f"[WARN] offline CLIP prior broadcast failed: {e}")
                    clip_weight_case = None

            # Convert to robust ellip cue if requested (works for both otf and offline prior).
            if clip_weight_case is not None and kind in ("ellip", "ellipse", "blob", "pointblob"):
                h, w = int(clip_weight_case.shape[-2]), int(clip_weight_case.shape[-1])
                sx = float(getattr(self.args, "clip_ellip_sigma_x", 0.0))
                sy = float(getattr(self.args, "clip_ellip_sigma_y", 0.0))
                if sx <= 0:
                    sx = 0.16 * w
                if sy <= 0:
                    sy = 0.12 * h
                p = float(getattr(self.args, "clip_ellip_power", 1.0))
                sym = bool(getattr(self.args, "clip_ellip_sym", True))
                sm = float(getattr(self.args, "clip_ellip_smooth_sigma", 2.0))
                clip_weight_case = saliency_to_elliptic_blob(
                    clip_weight_case,
                    sigma_y=sy,
                    sigma_x=sx,
                    power=p,
                    sym=sym,
                    smooth_sigma=sm,
                ).to(device=self.device, dtype=clip_weight_case.dtype)

        # Unified, test-time-only adaptation: estimate confidence of CLIP guidance on this *case* and
        # automatically down-weight/soften it when it appears unreliable (no domain-specific tuning).
        if getattr(self.args, "clip_guided", False) and clip_weight_case is not None:
            try:
                w_rel = clip_weight_case_raw if clip_weight_case_raw is not None else clip_weight_case
                # 1) Intrinsic peakiness (saliency concentrated?) per slice.
                conf_peak = _saliency_confidence(w_rel)  # [N]
                src_logits_all = torch.cat(baseline_logits, dim=0).detach()
                src_area = float(_pred_area_ratio_from_logits(src_logits_all, thr=0.5).mean().item())
                # 2) Agreement with source prediction (avoid trusting a sharp but off-target prior),
                # but when the source collapses to (near) all-background, *do not* use it to down-weight CLIP.
                if src_area < float(getattr(self.args, "clip_rel_src_area_min", 0.002)):
                    conf_align = torch.ones_like(conf_peak)
                    conf_center = torch.ones_like(conf_peak)
                else:
                    align = clip_alignment_score_sigmoid_per_sample(src_logits_all, w_rel).detach()  # [N]
                    rel_thr = float(getattr(self.args, "clip_rel_thr", 0.0))
                    rel_temp = float(getattr(self.args, "clip_rel_temp", 0.1))
                    conf_align = sigmoid_gate(align, thr=rel_thr, temp=rel_temp, floor=0.0).detach().clamp(0.0, 1.0)
                    # 3) Center agreement between source centroid and prior peak (penalize gross mis-localization).
                    src_c = _centroid_xy_from_logits_3d(src_logits_all, thr=0.5)
                    if src_c is not None:
                        px, py = src_c
                        pm = w_rel.detach().mean(dim=0)[0]
                        qx, qy = _argmax_2d_xy(pm)
                        hh = float(pm.shape[-2])
                        ww = float(pm.shape[-1])
                        dx = (px - qx) / max(1.0, ww)
                        dy = (py - qy) / max(1.0, hh)
                        dist0 = float((dx * dx + dy * dy) ** 0.5)
                        sig0 = float(getattr(self.args, "clip_rel_center_sigma", 0.25))
                        sig0 = max(1e-3, sig0)
                        conf_center = float(np.exp(-(dist0 * dist0) / (2.0 * sig0 * sig0)))
                        conf_center = torch.full_like(conf_peak, float(conf_center))
                    else:
                        conf_center = torch.ones_like(conf_peak)

                clip_q_case = (conf_peak * conf_align * conf_center).detach().clamp(0.0, 1.0)  # [N]

                # Optionally blend with a generic anatomical blob; this keeps a unified config fair across domains.
                if bool(getattr(self.args, "clip_auto_blend", True)):
                    h0 = int(clip_weight_case.shape[-2])
                    w0 = int(clip_weight_case.shape[-1])
                    generic = _make_generic_prior_map(h0, w0, device=self.device, dtype=clip_weight_case.dtype)
                    clip_weight_case = _blend_prior_with_generic(clip_weight_case, clip_q_case, generic)
            except Exception as e:
                self.logger.info(f"[WARN] clip_auto_blend failed: {e}")
                clip_q_case = None

        # Case-wise update gate: estimate if this case is safe/useful to adapt.
        case_gate_val = 1.0
        if getattr(self.args, "case_gate", False):
            try:
                src_logits_all = torch.cat(baseline_logits, dim=0).detach()
                w_all = clip_weight_case if clip_weight_case is not None else clip_prior_hw
                metric = str(getattr(self.args, "case_gate_metric", "clip_align"))

                s = None
                if metric == "clip_align" and w_all is not None:
                    s = clip_alignment_score_sigmoid_per_sample(src_logits_all, w_all).mean()
                elif metric == "teacher_conf":
                    s = teacher_confidence_from_logits(src_logits_all).mean()
                else:
                    # entropy (lower is better) -> negate so larger is better for sigmoid_gate
                    if getattr(self.args, "clip_guided", False) and w_all is not None and getattr(self.args, "clip_entropy_weighted", True):
                        s = -entropy_loss_sigmoid_weighted_per_sample(
                            src_logits_all,
                            weight_hw=w_all,
                            weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                            weight_min=getattr(self.args, "clip_weight_min", 0.0),
                        ).mean()
                    else:
                        s = -entropy_loss_sigmoid_per_sample(src_logits_all).mean()

                if s is not None:
                    thr = float(getattr(self.args, "case_gate_thr", 0.0))
                    temp = float(getattr(self.args, "case_gate_temp", 0.1))
                    floor = float(getattr(self.args, "case_gate_floor", 0.0))
                    if temp <= 1e-6:
                        g = (s >= thr).to(dtype=s.dtype)
                        if floor > 0.0:
                            g = torch.clamp(g, min=floor, max=1.0)
                    else:
                        g = sigmoid_gate(s, thr=thr, temp=temp, floor=floor)
                    case_gate_val = float(g.detach().cpu().item())
                    s_raw = float(s.detach().cpu().item())
                else:
                    s_raw = None

                skip_thr = float(getattr(self.args, "case_gate_skip_thr", 0.0))
                if skip_thr > 0.0 and case_gate_val < skip_thr:
                    self.logger.info(
                        f"[case_gate] skip adaptation for this case (gate={case_gate_val:.3f} < {skip_thr:.3f}, s={s_raw})"
                    )
                    preds = list(baseline_logits)
                    preds_after_one_case = list(baseline_logits)
                    full_pred = torch.cat(preds, dim=0)
                    full_pred = torch.sigmoid(full_pred)
                    full_pred = (full_pred > 0.5).float()
                    full_pred_final = torch.cat(preds_after_one_case, dim=0)
                    full_pred_final = torch.sigmoid(full_pred_final)
                    full_pred_final = (full_pred_final > 0.5).float()
                    return full_pred, full_pred_final

                self.logger.info(f"[case_gate] metric={metric} s={s_raw} gate={case_gate_val:.3f}")
            except Exception as e:
                self.logger.info(f"[WARN] case_gate failed: {e}")
                case_gate_val = 1.0

        if not self.args.model_episodic:
            reset_mode = getattr(self.args, "reset_mode", "from_target")
            self.model.reset_online_network(mode=reset_mode)

        prev_step_loss = None
        for step in range(0, self.args.optim_steps):
            self.model.online_network.train()
            train_loss_list = list()
            preds = []
            for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
                batch_start = i * self.args.batch_size
                batch_end = min((i + 1) * self.args.batch_size, num_samples)
                batch_data = data[batch_start:batch_end, :, :, :]

                self.model.online_network.train()
                output = self.model.online_network(batch_data)

                # Choose per-batch weight map:
                # - clip_otf: per-slice weights (preferred for "each test image has guidance")
                # - else: a single case-level prior map loaded from disk (older behavior)
                w_batch = None
                q_batch = None
                if clip_weight_case is not None:
                    w_batch = clip_weight_case[batch_start:batch_end]
                    if clip_q_case is not None:
                        q_batch = clip_q_case[batch_start:batch_end]
                elif clip_prior_hw is not None:
                    w_batch = clip_prior_hw

                if getattr(self.args, "clip_guided", False) and w_batch is not None and getattr(self.args, "clip_entropy_weighted", True):
                    loss_entropy_before = entropy_loss_sigmoid_weighted(
                        output,
                        weight_hw=w_batch,
                        weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                        weight_min=getattr(self.args, "clip_weight_min", 0.0),
                    )
                else:
                    loss_entropy_before = entropy_loss_sigmoid(output)
                all_loss = loss_entropy_before

                # Teacher-student consistency (EMA teacher): stabilize online updates and reduce drift.
                if float(getattr(self.args, "ts_lambda", 0.0)) > 0:
                    try:
                        self.model.target_network.eval()
                        x_w = photometric_augment_2d(
                            batch_data,
                            noise_std=float(getattr(self.args, "ts_weak_noise", 0.01)),
                            brightness=float(getattr(self.args, "ts_weak_brightness", 0.05)),
                            contrast=float(getattr(self.args, "ts_weak_contrast", 0.05)),
                            gamma=0.0,
                            p=1.0,
                        )
                        x_s = photometric_augment_2d(
                            batch_data,
                            noise_std=float(getattr(self.args, "ts_strong_noise", 0.03)),
                            brightness=float(getattr(self.args, "ts_strong_brightness", 0.10)),
                            contrast=float(getattr(self.args, "ts_strong_contrast", 0.10)),
                            gamma=float(getattr(self.args, "ts_strong_gamma", 0.0)),
                            p=1.0,
                        )
                        with torch.no_grad():
                            t_logits = self.model(x_w, training=False)
                            if isinstance(t_logits, (tuple, list)):
                                t_logits = t_logits[0]
                        s_logits = self.model.online_network(x_s)
                        if isinstance(s_logits, (tuple, list)):
                            s_logits = s_logits[0]
                        conf = teacher_confidence_from_logits(t_logits)  # [B]
                        thr = float(getattr(self.args, "ts_conf_thr", 0.12))
                        temp = float(getattr(self.args, "ts_conf_temp", 0.08))
                        if temp <= 1e-6:
                            gate = (conf > thr).to(dtype=s_logits.dtype)
                        else:
                            gate = torch.sigmoid((conf - thr) / temp).to(dtype=s_logits.dtype)
                        w_cons = w_batch if getattr(self.args, "ts_weight_with_clip", False) else None
                        cons = mse_consistency_loss(
                            student_logits=s_logits,
                            teacher_logits=t_logits.detach(),
                            weight_hw=w_cons,
                            weight_gamma=float(getattr(self.args, "ts_weight_gamma", 1.0)),
                            weight_min=float(getattr(self.args, "ts_weight_min", 0.0)),
                        )
                        ts_lambda = float(getattr(self.args, "ts_lambda", 0.0))
                        if getattr(self.args, "case_gate_scale_clip", False):
                            ts_lambda *= float(case_gate_val)
                        all_loss = all_loss + ts_lambda * cons * gate.mean()
                    except Exception as e:
                        self.logger.info(f"[WARN] teacher-student loss failed: {e}")

                # Localization constraint (your request): encourage probability mass inside the CLIP saliency blob.
                if getattr(self.args, "clip_guided", False) and w_batch is not None and float(getattr(self.args, "clip_loc_lambda", 0.0)) > 0:
                    w = w_batch
                    if w.dim() == 2:
                        w = w.view(1, 1, w.shape[0], w.shape[1])
                    elif w.dim() == 3:
                        w = w.view(w.shape[0], 1, w.shape[1], w.shape[2])
                    w = w.to(device=output.device, dtype=output.dtype)
                    w = torch.clamp(w, 0.0, 1.0)
                    wp = float(getattr(self.args, "clip_loc_power", 1.0))
                    if wp and wp != 1.0:
                        w = w.pow(max(0.1, wp))
                    p = torch.sigmoid(output)
                    mass_in = (p * w).mean()
                    mass_out = (p * (1.0 - w)).mean()
                    clip_loc_lambda = float(self.args.clip_loc_lambda)
                    if getattr(self.args, "case_gate_scale_clip", False):
                        clip_loc_lambda *= float(case_gate_val)
                    if q_batch is not None and bool(getattr(self.args, "clip_auto_weight", True)):
                        qv = float(q_batch.mean().detach().cpu().item())
                        qf = float(getattr(self.args, "clip_auto_weight_floor", 0.35))
                        qf = max(0.0, min(1.0, qf))
                        clip_loc_lambda *= (qf + (1.0 - qf) * qv)
                    all_loss = all_loss + clip_loc_lambda * (mass_out - mass_in)

                # ViDA/CLIP paper-style: entropy + soft prior BCE (soft target).
                if getattr(self.args, "clip_guided", False) and w_batch is not None and float(getattr(self.args, "clip_bce_lambda", 0.0)) > 0:
                    loss_bce = bce_with_logits_weighted(
                        logits=output,
                        target_hw=w_batch,
                        gamma=getattr(self.args, "clip_bce_gamma", 1.0),
                        weight_min=getattr(self.args, "clip_bce_weight_min", 0.0),
                    )
                    clip_bce_lambda = float(self.args.clip_bce_lambda)
                    if getattr(self.args, "case_gate_scale_clip", False):
                        clip_bce_lambda *= float(case_gate_val)
                    if q_batch is not None and bool(getattr(self.args, "clip_auto_weight", True)):
                        qv = float(q_batch.mean().detach().cpu().item())
                        qf = float(getattr(self.args, "clip_auto_weight_floor", 0.35))
                        qf = max(0.0, min(1.0, qf))
                        clip_bce_lambda *= (qf + (1.0 - qf) * qv)
                    all_loss = all_loss + clip_bce_lambda * loss_bce

                if getattr(self.args, "clip_align_lambda", 0.0) and w_batch is not None:
                    w = w_batch
                    if w.dim() == 2:
                        w = w.view(1, 1, w.shape[0], w.shape[1])
                    elif w.dim() == 3:
                        w = w.view(w.shape[0], 1, w.shape[1], w.shape[2])
                    w = w.to(device=output.device, dtype=output.dtype)
                    w = torch.clamp(w, 0.0, 1.0)
                    p = torch.sigmoid(output)
                    mass_in = (p * w).mean()
                    mass_out = (p * (1.0 - w)).mean()
                    all_loss = all_loss + float(self.args.clip_align_lambda) * (mass_out - mass_in)

                train_loss_list.append(all_loss.item())
                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()

                # CLIP-gated AMU/AUM: update teacher more on "good/localized" samples, less on bad ones.
                gate_val = None
                if getattr(self.args, "aum_gate", False) and getattr(self.args, "clip_guided", False) and w_batch is not None:
                    metric = str(getattr(self.args, "aum_gate_metric", "clip_align"))
                    try:
                        if metric == "clip_align":
                            s = clip_alignment_score_sigmoid_per_sample(output.detach(), w_batch).mean()
                        elif metric == "teacher_conf":
                            with torch.no_grad():
                                t_logits = self.model(batch_data, training=False)
                                if isinstance(t_logits, (tuple, list)):
                                    t_logits = t_logits[0]
                            s = teacher_confidence_from_logits(t_logits).mean()
                        elif metric == "entropy":
                            if getattr(self.args, "clip_entropy_weighted", True):
                                s = (
                                    -entropy_loss_sigmoid_weighted_per_sample(
                                        output.detach(),
                                        weight_hw=w_batch,
                                        weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                                        weight_min=getattr(self.args, "clip_weight_min", 0.0),
                                    ).mean()
                                )
                            else:
                                s = -entropy_loss_sigmoid_per_sample(output.detach()).mean()
                        else:
                            s = clip_alignment_score_sigmoid_per_sample(output.detach(), w_batch).mean()
                        g = sigmoid_gate(
                            s,
                            thr=float(getattr(self.args, "aum_gate_thr", 0.0)),
                            temp=float(getattr(self.args, "aum_gate_temp", 0.1)),
                            floor=float(getattr(self.args, "aum_gate_floor", 0.0)),
                        )
                        gate_val = float(g.detach().cpu().item())
                    except Exception as e:
                        self.logger.info(f"[WARN] aum_gate failed: {e}")
                        gate_val = None

                if gate_val is None:
                    self.model.update_target_network()
                else:
                    self.model.update_target_network(gate=gate_val)

                self.source_model.eval()
                output_wo_ada = self.source_model(batch_data)

                self.model.online_network.eval()
                output = self.model.online_network(batch_data)
                use_align_select = (
                    getattr(self.args, "clip_guided", False) and w_batch is not None and getattr(self.args, "clip_select", "entropy") == "align"
                )
                if use_align_select:
                    s_ada = clip_alignment_score_sigmoid_per_sample(output, w_batch)
                    s_src = clip_alignment_score_sigmoid_per_sample(output_wo_ada, w_batch)
                    keep = (s_ada >= s_src).view(-1, 1, 1, 1)
                    preds.append(torch.where(keep, output, output_wo_ada))
                else:
                    if getattr(self.args, "clip_guided", False) and w_batch is not None and getattr(self.args, "clip_entropy_weighted", True):
                        e_ada = entropy_loss_sigmoid_weighted_per_sample(
                            output,
                            weight_hw=w_batch,
                            weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                            weight_min=getattr(self.args, "clip_weight_min", 0.0),
                        )
                        e_src = entropy_loss_sigmoid_weighted_per_sample(
                            output_wo_ada,
                            weight_hw=w_batch,
                            weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                            weight_min=getattr(self.args, "clip_weight_min", 0.0),
                        )
                    else:
                        e_ada = entropy_loss_sigmoid_per_sample(output)
                        e_src = entropy_loss_sigmoid_per_sample(output_wo_ada)
                    keep = (e_ada <= e_src).view(-1, 1, 1, 1)
                    preds.append(torch.where(keep, output, output_wo_ada))

                train_loss_list.append(all_loss.detach().cpu().numpy())

            self.all_steps += 1
            self.logger.info('Step {}:'.format(step))
            self.logger.info('  loss:{}'.format(np.mean(train_loss_list)))
            self.writer.add_scalar("Train Scalars/Train Loss", np.mean(train_loss_list), self.all_steps)
            if getattr(self.args, "ts_early_stop", False):
                cur = float(np.mean(train_loss_list)) if len(train_loss_list) else None
                if cur is not None and prev_step_loss is not None:
                    delta = cur - prev_step_loss
                    tol = float(getattr(self.args, "ts_early_stop_tol", 0.0))
                    if delta > tol:
                        self.logger.info(f"Early-stop at step {step} (loss increased by {delta:.4f} > {tol:.4f}).")
                        break
                prev_step_loss = cur

        self.model.online_network.eval()
        preds_after_one_case = []
        final_select = bool(getattr(self.args, "final_select", False))
        with torch.no_grad():
            for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
                batch_start = i * self.args.batch_size
                batch_end = min((i + 1) * self.args.batch_size, num_samples)
                batch_data = data[batch_start:batch_end, :, :, :]
                out_ada = self.model.online_network(batch_data)
                if not final_select:
                    preds_after_one_case.append(out_ada)
                    continue
                w_final = None
                if clip_weight_case is not None:
                    w_final = clip_weight_case[batch_start:batch_end]
                elif clip_prior_hw is not None:
                    w_final = clip_prior_hw

                if getattr(self.args, "clip_guided", False) and w_final is not None and getattr(self.args, "clip_select", "entropy") == "align":
                    out_src = self.source_model(batch_data)
                    s_ada = clip_alignment_score_sigmoid_per_sample(out_ada, w_final)
                    s_src = clip_alignment_score_sigmoid_per_sample(out_src, w_final)
                    keep = (s_ada >= s_src).view(-1, 1, 1, 1)
                    preds_after_one_case.append(torch.where(keep, out_ada, out_src))
                elif getattr(self.args, "clip_guided", False) and w_final is not None:
                    out_src = self.source_model(batch_data)
                    if getattr(self.args, "clip_entropy_weighted", True):
                        e_ada = entropy_loss_sigmoid_weighted_per_sample(
                            out_ada,
                            weight_hw=w_final,
                            weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                            weight_min=getattr(self.args, "clip_weight_min", 0.0),
                        )
                        e_src = entropy_loss_sigmoid_weighted_per_sample(
                            out_src,
                            weight_hw=w_final,
                            weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                            weight_min=getattr(self.args, "clip_weight_min", 0.0),
                        )
                    else:
                        e_ada = entropy_loss_sigmoid_per_sample(out_ada)
                        e_src = entropy_loss_sigmoid_per_sample(out_src)
                    keep = (e_ada <= e_src).view(-1, 1, 1, 1)
                    preds_after_one_case.append(torch.where(keep, out_ada, out_src))
                else:
                    preds_after_one_case.append(out_ada)

        # Case-level fallback (protect hard cases): if adaptation looks worse by a proxy, revert Dice_final to source.
        keep_case_by_proxy = None
        keep_case_dbg = None
        if getattr(self.args, "case_fallback", False):
            try:
                ada_logits = torch.cat(preds_after_one_case, dim=0)
                src_logits = torch.cat(baseline_logits, dim=0)
                w_all = clip_weight_case if clip_weight_case is not None else clip_prior_hw

                metric = str(getattr(self.args, "case_fallback_metric", "entropy"))
                margin = max(0.0, float(getattr(self.args, "case_fallback_margin", 0.0)))

                if metric == "clip_align" and w_all is not None:
                    s_ada = clip_alignment_score_sigmoid_per_sample(ada_logits, w_all).mean().item()
                    s_src = clip_alignment_score_sigmoid_per_sample(src_logits, w_all).mean().item()
                    keep_case = (s_ada >= (s_src - margin))
                    dbg = f"clip_align ada={s_ada:.4f} src={s_src:.4f} m={margin:.4f}"
                elif metric == "teacher_conf":
                    s_ada = teacher_confidence_from_logits(ada_logits).mean().item()
                    s_src = teacher_confidence_from_logits(src_logits).mean().item()
                    keep_case = (s_ada >= (s_src - margin))
                    dbg = f"conf ada={s_ada:.4f} src={s_src:.4f} m={margin:.4f}"
                else:
                    if getattr(self.args, "clip_guided", False) and w_all is not None and getattr(self.args, "clip_entropy_weighted", True):
                        e_ada = entropy_loss_sigmoid_weighted_per_sample(
                            ada_logits,
                            weight_hw=w_all,
                            weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                            weight_min=getattr(self.args, "clip_weight_min", 0.0),
                        ).mean().item()
                        e_src = entropy_loss_sigmoid_weighted_per_sample(
                            src_logits,
                            weight_hw=w_all,
                            weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                            weight_min=getattr(self.args, "clip_weight_min", 0.0),
                        ).mean().item()
                    else:
                        e_ada = entropy_loss_sigmoid_per_sample(ada_logits).mean().item()
                        e_src = entropy_loss_sigmoid_per_sample(src_logits).mean().item()
                    keep_case = (e_ada <= (e_src + margin))
                    dbg = f"entropy ada={e_ada:.4f} src={e_src:.4f} m={margin:.4f}"

                keep_case_by_proxy = bool(keep_case)
                keep_case_dbg = dbg
                if not keep_case:
                    preds_after_one_case = list(baseline_logits)
                    self.logger.info(f"[case_fallback] revert Dice_final to source ({dbg})")
                else:
                    self.logger.info(f"[case_fallback] keep adapted Dice_final ({dbg})")
            except Exception as e:
                self.logger.info(f"[WARN] case_fallback failed: {e}")

        # Case-level rollback: if a proxy says this case update is harmful, restore pre-case weights.
        if rollback_enabled and rollback_snap is not None:
            try:
                reject = None
                if keep_case_by_proxy is not None:
                    reject = not bool(keep_case_by_proxy)
                    dbg = keep_case_dbg or "case_fallback"
                else:
                    post_logits = torch.cat(preds_after_one_case, dim=0).detach()
                    pre_logits = torch.cat(pre_case_logits, dim=0).detach() if pre_case_logits is not None else post_logits
                    w_all = clip_weight_case if clip_weight_case is not None else clip_prior_hw
                    metric = str(getattr(self.args, "case_rollback_metric", "entropy"))
                    margin = float(getattr(self.args, "case_rollback_margin", 0.0))
                    if metric == "clip_align" and w_all is not None:
                        s_post = float(clip_alignment_score_sigmoid_per_sample(post_logits, w_all).mean().item())
                        s_pre = float(clip_alignment_score_sigmoid_per_sample(pre_logits, w_all).mean().item())
                        reject = not (s_post >= (s_pre - margin))
                        dbg = f"clip_align post={s_post:.4f} pre={s_pre:.4f} m={margin:.4f}"
                    elif metric == "teacher_conf":
                        s_post = float(teacher_confidence_from_logits(post_logits).mean().item())
                        s_pre = float(teacher_confidence_from_logits(pre_logits).mean().item())
                        reject = not (s_post >= (s_pre - margin))
                        dbg = f"conf post={s_post:.4f} pre={s_pre:.4f} m={margin:.4f}"
                    else:
                        if getattr(self.args, "clip_guided", False) and w_all is not None and getattr(self.args, "clip_entropy_weighted", True):
                            e_post = float(
                                entropy_loss_sigmoid_weighted_per_sample(
                                    post_logits,
                                    weight_hw=w_all,
                                    weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                                    weight_min=getattr(self.args, "clip_weight_min", 0.0),
                                ).mean().item()
                            )
                            e_pre = float(
                                entropy_loss_sigmoid_weighted_per_sample(
                                    pre_logits,
                                    weight_hw=w_all,
                                    weight_gamma=getattr(self.args, "clip_weight_gamma", 1.0),
                                    weight_min=getattr(self.args, "clip_weight_min", 0.0),
                                ).mean().item()
                            )
                        else:
                            e_post = float(entropy_loss_sigmoid_per_sample(post_logits).mean().item())
                            e_pre = float(entropy_loss_sigmoid_per_sample(pre_logits).mean().item())
                        reject = not (e_post <= (e_pre + margin))
                        dbg = f"entropy post={e_post:.4f} pre={e_pre:.4f} m={margin:.4f}"

                    a_min = getattr(self.args, "case_rollback_area_min", None)
                    a_max = getattr(self.args, "case_rollback_area_max", None)
                    if a_min is not None or a_max is not None:
                        a_post = float(_pred_area_ratio_from_logits(post_logits, thr=0.5).mean().item())
                        ok = True
                        if a_min is not None:
                            ok = ok and (a_post >= float(a_min))
                        if a_max is not None:
                            ok = ok and (a_post <= float(a_max))
                        if not ok:
                            reject = True
                            dbg = dbg + f" area_post={a_post:.4f} (range {a_min},{a_max})"

                    c_max = getattr(self.args, "case_rollback_center_max", None)
                    c_margin = getattr(self.args, "case_rollback_center_margin", None)
                    if (c_max is not None or c_margin is not None) and w_all is not None:
                        c_post = _prior_center_distance_from_logits(post_logits, w_all, clip_q_case)
                        c_pre = _prior_center_distance_from_logits(pre_logits, w_all, clip_q_case)
                        center_bad = False
                        center_dbg = []
                        if c_post is not None:
                            if c_max is not None and c_post > float(c_max):
                                center_bad = True
                                center_dbg.append(f"center_post={c_post:.3f}>{float(c_max):.3f}")
                            if c_margin is not None and c_pre is not None and c_post > c_pre + float(c_margin):
                                center_bad = True
                                center_dbg.append(f"center_post={c_post:.3f}>pre={c_pre:.3f}+{float(c_margin):.3f}")
                        if center_bad:
                            reject = True
                            dbg = dbg + " " + " ".join(center_dbg)

                if reject:
                    self.logger.info(f"[case_rollback] rollback weights ({dbg})")
                    _restore_ema_wrapper_state(self.model, rollback_snap)
                    # Keep predictions consistent with rolled-back weights: use pre-adaptation logits.
                    preds = list(pre_case_logits)
                    preds_after_one_case = list(pre_case_logits)
                    rollback_replaced_preds = True
                else:
                    self.logger.info(f"[case_rollback] keep weights ({dbg})")
            except Exception as e:
                self.logger.info(f"[WARN] case_rollback failed: {e}")

        # Hard safety guard: if the final predictions for this case collapse to an implausible
        # foreground area ratio, replace them with source predictions (and rollback weights).
        #
        # This catches confident all-background failures that entropy/confidence proxies may miss.
        try:
            g_min = getattr(self.args, "case_guard_area_min", None)
            g_max = getattr(self.args, "case_guard_area_max", None)
            if g_min is not None or g_max is not None:
                post_logits_all = torch.cat(preds_after_one_case, dim=0).detach()
                a_post = float(_pred_area_ratio_from_logits(post_logits_all, thr=0.5).mean().item())
                ok = True
                if g_min is not None:
                    ok = ok and (a_post >= float(g_min))
                if g_max is not None:
                    ok = ok and (a_post <= float(g_max))
                if not ok:
                    self.logger.info(f"[case_guard] replace preds with source (area_post={a_post:.4f}, range {g_min},{g_max})")
                    preds = list(baseline_logits)
                    preds_after_one_case = list(baseline_logits)
                    if rollback_enabled and rollback_snap is not None and not rollback_replaced_preds:
                        self.logger.info("[case_guard] rollback weights to pre-case snapshot")
                        _restore_ema_wrapper_state(self.model, rollback_snap)

            # Extra safety: if prediction centroid is far from the CLIP prior peak, treat as mis-localized
            # and fall back to source (also rollback weights). This helps reduce catastrophic HD95 outliers.
            c_max = getattr(self.args, "case_guard_center_max", None)
            if c_max is not None and getattr(self.args, "clip_guided", False):
                post_logits_all = torch.cat(preds_after_one_case, dim=0).detach()
                pred_c = _centroid_xy_from_logits_3d(post_logits_all, thr=0.5)
                prior_map = None
                if clip_weight_case is not None:
                    # [N,1,H,W] -> [H,W], optionally weight slices by confidence.
                    w_case = clip_weight_case.detach()
                    if clip_q_case is not None:
                        q = clip_q_case.detach().clamp(0.0, 1.0).view(-1, 1, 1, 1)
                        prior_map = (w_case * q).sum(dim=0)[0] / (q.sum().clamp_min(1e-6))
                    else:
                        prior_map = w_case.mean(dim=0)[0]
                elif clip_prior_hw is not None:
                    prior_map = clip_prior_hw.detach()
                if pred_c is not None and prior_map is not None:
                    px, py = pred_c
                    qx, qy = _argmax_2d_xy(prior_map)
                    h = float(prior_map.shape[-2])
                    w = float(prior_map.shape[-1])
                    dx = (px - qx) / max(1.0, w)
                    dy = (py - qy) / max(1.0, h)
                    dist = float((dx * dx + dy * dy) ** 0.5)
                    q_mean = float(clip_q_case.mean().detach().cpu().item()) if clip_q_case is not None else 1.0
                    q_min = float(getattr(self.args, "case_guard_center_qmin", 0.15))
                    relax = float(getattr(self.args, "case_guard_center_relax", 0.25))
                    c_eff = float(c_max) + max(0.0, relax) * max(0.0, 1.0 - q_mean)
                    # Only enforce when CLIP guidance appears reliable enough.
                    if q_mean >= q_min and dist > float(c_eff):
                        self.logger.info(
                            f"[case_guard] replace preds with source (center_dist={dist:.3f} > {float(c_eff):.3f}, q={q_mean:.3f})"
                        )
                        preds = list(baseline_logits)
                        preds_after_one_case = list(baseline_logits)
                        if rollback_enabled and rollback_snap is not None and not rollback_replaced_preds:
                            self.logger.info("[case_guard] rollback weights to pre-case snapshot")
                            _restore_ema_wrapper_state(self.model, rollback_snap)
        except Exception as e:
            self.logger.info(f"[WARN] case_guard failed: {e}")

        # Safe Branch Selector: domain-agnostic per-case selection between candidate outputs.
        # Uses test-time reliability signals only (no ground truth, no domain labels).
        if getattr(self.args, "safe_branch_selector", False):
            try:
                src_logits_all = torch.cat(baseline_logits, dim=0).detach()
                ada_logits_all = torch.cat(preds, dim=0).detach()
                final_logits_all = torch.cat(preds_after_one_case, dim=0).detach()
                w_sel = clip_weight_case
                q_sel = clip_q_case
                a_min = float(getattr(self.args, "safe_branch_selector_area_min", 0.002))
                a_max = float(getattr(self.args, "safe_branch_selector_area_max", 0.10))
                c_max = float(getattr(self.args, "safe_branch_selector_center_max", 0.30))

                info_src = _compute_case_reliability_score(src_logits_all, w_sel, q_sel, a_min, a_max, c_max)
                info_ada = _compute_case_reliability_score(ada_logits_all, w_sel, q_sel, a_min, a_max, c_max)
                info_fin = _compute_case_reliability_score(final_logits_all, w_sel, q_sel, a_min, a_max, c_max)

                best, reason = _safe_branch_select(
                    info_src,
                    info_ada,
                    info_fin,
                    conservative=bool(getattr(self.args, "safe_branch_selector_conservative", False)),
                    adaptive=bool(getattr(self.args, "safe_branch_selector_adaptive", False)),
                    final_min_gain=float(getattr(self.args, "safe_branch_selector_final_min_gain", 0.02)),
                    final_bias=float(getattr(self.args, "safe_branch_selector_final_bias", 0.03)),
                )

                # Case-level selection only. Per-slice source/adapted mixing breaks
                # 3D anatomical continuity and caused HD95 outliers on prostate.
                if best == "source":
                    selected_preds = list(baseline_logits)
                elif best == "adapted":
                    selected_preds = list(preds)
                else:
                    selected_preds = list(preds_after_one_case)

                # Keep raw output as the adapted branch unless it is globally unsafe.
                # Dice_final carries the selector decision; raw remains useful for
                # reporting and avoids postprocess-induced Dice drops.
                if best == "source" and (not info_ada.get("area_ok", True)):
                    preds = list(baseline_logits)
                else:
                    preds = list(preds)
                preds_after_one_case = selected_preds

                n_src = 1 if best == "source" else 0
                n_ada = 1 if best == "adapted" else 0
                n_fin = 1 if best == "final" else 0
                self.logger.info(
                    f"[safe_branch_selector] best={best} reason={reason} "
                    f"slice_sel=src:{n_src}/ada:{n_ada}/fin:{n_fin} "
                    f"score_src={info_src['score']:.3f} score_ada={info_ada['score']:.3f} "
                    f"score_fin={info_fin['score']:.3f} "
                    f"area_src={info_src['area_mean']:.4f} area_ada={info_ada['area_mean']:.4f} "
                    f"area_fin={info_fin['area_mean']:.4f} "
                    f"ent_src={info_src['entropy_mean']:.4f} ent_ada={info_ada['entropy_mean']:.4f} "
                    f"ent_fin={info_fin['entropy_mean']:.4f} "
                    f"align_src={info_src.get('clip_align_mean','?'):.4f} align_ada={info_ada.get('clip_align_mean','?'):.4f} "
                    f"align_fin={info_fin.get('clip_align_mean','?'):.4f}"
                )
            except Exception as e:
                self.logger.info(f"[WARN] safe_branch_selector failed: {e}")

        # Expose CLIP reliability to evaluation-time postprocess (e.g., CC-by-prior).
        # If CLIP is unreliable for this case/domain, downstream steps should avoid using the prior.
        try:
            if clip_q_case is not None:
                self._clip_q_mean_last = float(clip_q_case.mean().detach().cpu().item())
                self._clip_q_min_last = float(clip_q_case.min().detach().cpu().item())
            else:
                self._clip_q_mean_last = None
                self._clip_q_min_last = None
        except Exception:
            self._clip_q_mean_last = None
            self._clip_q_min_last = None

        full_pred = torch.cat(preds, dim=0)
        full_pred = torch.sigmoid(full_pred)
        full_pred = (full_pred > 0.5).float()

        full_pred_final = torch.cat(preds_after_one_case, dim=0)
        full_pred_final = torch.sigmoid(full_pred_final)
        full_pred_final = (full_pred_final > 0.5).float()
        return full_pred, full_pred_final


    def evaluate_prostate_seg(self):
        for epoch in range(1):
            case_name_all = []
            Dice_Metrics = AverageMeter(save_all=True)
            ASSD_Metrics = AverageMeter(save_all=True)
            HD_Metrics = AverageMeter(save_all=True)

            Dice_Metrics_final = AverageMeter(save_all=True)
            ASSD_Metrics_final = AverageMeter(save_all=True)
            HD_Metrics_final = AverageMeter(save_all=True)

            for iter, batch in enumerate(self.ts_dataloader):
                data, seg, case_name = batch
                data = data[0].to(self.device)
                seg = seg[0].to(self.device)

                clip_prior_hw = None
                clip_prior_slice_idx = None
                if getattr(self.args, "clip_guided", False):
                    try:
                        if not getattr(self.args, "clip_otf", False):
                            clip_prior_hw, clip_prior_slice_idx = load_case_clip_prior(
                                clip_prior_root=self.args.clip_prior_root,
                                domain=self.args.domain,
                                case_id=case_name[0],
                                kind=getattr(self.args, "clip_prior_kind", "clip"),
                            )
                            clip_prior_hw = clip_prior_hw.to(self.device)
                    except Exception as e:
                        self.logger.info(f"[WARN] clip prior load failed for {case_name[0]}: {e}")
                        clip_prior_hw = None
                        clip_prior_slice_idx = None

                ##################### perform tta per case ################
                start = time()
                pred_aggregated, pred_aggregated_final = self.do_prostate_seg_tta_per_case(
                    data.unsqueeze(1).repeat(1, 3, 1, 1),
                    clip_prior_hw=clip_prior_hw,
                    clip_prior_slice_idx=clip_prior_slice_idx if getattr(self.args, "clip_guided", False) else None,
                )
                end = time()
                case_name_all.append(case_name[0])
                self.logger.info('  Case {}  TTA time:{}'.format(case_name[0], end-start))
                ########### compute the tta-pred metrics ##################
                pred_array = pred_aggregated.detach().squeeze().cpu().numpy()
                pred_array_final = pred_aggregated_final.detach().squeeze().cpu().numpy()

                target_array = seg.squeeze().cpu().numpy()
                # Postprocess: connected-component filtering to suppress distant FPs that can dominate HD95.
                min_size = int(getattr(self.args, "post_lcc_min_size", 0) or 0)
                if getattr(self.args, "post_cc_by_prior", False) and clip_prior_hw is not None:
                    # Use CLIP prior (offline) to select the most plausible component.
                    # IMPORTANT: only trust the prior when CLIP reliability for this case is high enough,
                    # otherwise fall back to largest-CC to avoid catastrophic failures on some domains.
                    q_mean = getattr(self, "_clip_q_mean_last", None)
                    q_min = float(getattr(self.args, "post_cc_by_prior_qmin", 0.0) or 0.0)
                    if q_min > 0.0 and (q_mean is None or float(q_mean) < q_min):
                        self.logger.info(f"[post_cc_by_prior] skip (q_mean={q_mean}) < qmin={q_min}; fallback to LCC")
                        pred_array = _keep_largest_connected_component(pred_array, min_size=min_size)
                        pred_array_final = _keep_largest_connected_component(pred_array_final, min_size=min_size)
                    else:
                        sigma_eff = getattr(self, "_clip_slice_sigma_eff_last", None)
                        if sigma_eff is None:
                            sigma_eff = float(getattr(self.args, "clip_slice_sigma", 0.0) or 0.0)
                        pred_array = _keep_connected_component_by_prior(
                            pred_array,
                            prior_hw=clip_prior_hw.detach().cpu().numpy(),
                            prior_slice_idx=clip_prior_slice_idx,
                            slice_sigma=float(sigma_eff),
                            slice_radius=int(getattr(self.args, "clip_slice_radius", 0) or 0),
                            min_size=min_size,
                            min_score=float(getattr(self.args, "post_cc_by_prior_min_score", 0.0) or 0.0),
                        )
                        pred_array_final = _keep_connected_component_by_prior(
                            pred_array_final,
                            prior_hw=clip_prior_hw.detach().cpu().numpy(),
                            prior_slice_idx=clip_prior_slice_idx,
                            slice_sigma=float(sigma_eff),
                            slice_radius=int(getattr(self.args, "clip_slice_radius", 0) or 0),
                            min_size=min_size,
                            min_score=float(getattr(self.args, "post_cc_by_prior_min_score", 0.0) or 0.0),
                        )
                elif getattr(self.args, "post_lcc", False):
                    pred_array = _keep_largest_connected_component(pred_array, min_size=min_size)
                    pred_array_final = _keep_largest_connected_component(pred_array_final, min_size=min_size)
                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, self.logger)
                dc_final,assd_final, hd_final = get_dice_assd_hd(target_array, pred_array_final, self.logger)

                # Optional diagnostic/pipeline artifact for candidate-level selection.
                # The selector only uses the prediction masks; the target is stored only
                # to evaluate the selected result offline after selection.
                if self.args.save_pre:
                    try:
                        pred_dir = os.path.join(self.log_folder, 'predictions')
                        os.makedirs(pred_dir, exist_ok=True)
                        np.savez_compressed(
                            os.path.join(pred_dir, f'{case_name[0]}.npz'),
                            pred=pred_array.astype(np.uint8),
                            pred_final=pred_array_final.astype(np.uint8),
                            target=target_array.astype(np.uint8),
                        )
                    except Exception as e:
                        self.logger.info(f"[WARN] save_pre failed for {case_name[0]}: {e}")

                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)

                Dice_Metrics_final.update(dc_final)
                ASSD_Metrics_final.update(assd_final)
                HD_Metrics_final.update(hd_final)


                self.logger.info(
                "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name[0], dc, assd, hd))
                self.logger.info(
                "Cur-patient {}  dice_final:{:.3f} assd_final:{:.3f} hd_final:{:.3f}".format(case_name[0], dc_final, assd_final, hd_final))

            avg_dc = Dice_Metrics.avg
            avg_hd = HD_Metrics.avg
            avg_assd = ASSD_Metrics.avg

            std_dc = np.std(Dice_Metrics.all_data)
            std_hd = np.std(HD_Metrics.all_data)
            std_assd = np.std(ASSD_Metrics.all_data)

            avg_dc_final = Dice_Metrics_final.avg
            avg_hd_final = HD_Metrics_final.avg
            avg_assd_final = ASSD_Metrics_final.avg

            std_dc_final = np.std(Dice_Metrics_final.all_data)
            std_hd_final = np.std(HD_Metrics_final.all_data)
            std_assd_final = np.std(ASSD_Metrics_final.all_data)

            self.logger.info(
                    " Test Avg- dice:{:.4f}+{:.4f} hd:{:.4f}+{:.4f} assd:{:.4f}+{:.4f}".format(avg_dc,std_dc, avg_hd,std_hd, avg_assd, std_assd))
            self.logger.info(
                " Test Avg- dice_final:{:.4f}+{:.4f} hd_final:{:.4f}+{:.4f} assd_final:{:.4f}+{:.4f}".format(avg_dc_final,std_dc_final, avg_hd_final,std_hd_final, avg_assd_final, std_assd_final))

            # save per-case results for table alignment
            time_str = time_module.strftime("%Y%m%d%H%M", time_module.localtime())
            data_frame = pd.DataFrame(
                data={
                    'Case': case_name_all,
                    'Dice': Dice_Metrics.all_data,
                    'HD': HD_Metrics.all_data,
                    'ASSD': ASSD_Metrics.all_data,
                    'Dice_final': Dice_Metrics_final.all_data,
                    'HD_final': HD_Metrics_final.all_data,
                    'ASSD_final': ASSD_Metrics_final.all_data,
                },
                index=range(len(case_name_all)),
            )
            data_frame.to_csv(self.metrics_folder + '/' + time_str + '_results.csv', index_label='Index')


    def evaluate(self):
        if self.dataset_name == 'Prostate':
            self.evaluate_prostate_seg()
        else:
            raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Tent-EMA", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA', 'Tent_EMA'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_sptta", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=1, required=False,
                        help='Num of classes.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--only_bn_updated', default=True, required=False,
                        help='which part to be updated.')
    parser.add_argument('--tag', default="Base2_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='../medical_TTA/data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['../medical_TTA/data/RIGAPlus/MESSIDOR_Base2_test.csv'], required=False, help='test csv file.')
    parser.add_argument('--optim_steps', type=int, default=1, required=False,
                        help='optimization steps.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Model/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')
    parser.add_argument('--save_pre', required=False,default=False,
                        help='Whether to save the predicted mask.')

    ### ema model
    parser.add_argument('--ema_decay', type=float, default=0.6, required=False,
                        help='ema decay.')
    parser.add_argument('--min_momentum_constant', type=float, default=0.005, required=False,help='min momentum constant.')
    parser.add_argument(
        '--reset_mode',
        type=str,
        default='from_target',
        choices=['from_target', 'from_source'],
        help='How to reset the student/teacher at the start of each case.',
    )
    parser.add_argument(
        '--domain',
        type=str,
        default='UCL',
        choices=['BMC', 'BIDMC', 'UCL', 'HK', 'I2CVB'],
        help='Target domain for prostate online TTA.',
    )
    # Overrides after domain config is applied (useful for fast ablations / debugging).
    parser.add_argument(
        '--override_ts_csv',
        nargs='+',
        default=None,
        help='Override ts_csv after domain config update (e.g., data/MRI_prostate/BMC_test.csv).',
    )
    parser.add_argument(
        '--override_batch_size',
        type=int,
        default=None,
        help='Override batch_size after domain config update.',
    )
    parser.add_argument(
        '--override_model_episodic',
        type=int,
        default=None,
        choices=[0, 1],
        help='Override model_episodic after domain config update (0=False, 1=True).',
    )
    parser.add_argument(
        '--override_optim_steps',
        type=int,
        default=None,
        help='Override optim_steps after domain config update.',
    )
    parser.add_argument(
        '--override_initial_lr',
        type=float,
        default=None,
        help='Override initial_lr after domain config update.',
    )
    parser.add_argument(
        '--override_tag',
        action='store_true',
        help='Keep the CLI-provided --tag (do not overwrite with domain config tag).',
    )
    parser.add_argument('--exp_name', type=str, default="", help='Extra name suffix for log folder.')
    parser.add_argument('--use_vida', action='store_true', help='Enable ViDA (HR/LR + HKA) adapter inside UNet_SPTTA.')
    parser.add_argument('--vida_rank', type=int, default=64, help='Low-rank dimension for ViDA adapter.')
    parser.add_argument('--vida_hka_hidden', type=int, default=128, help='Hidden dim for HKA gate.')
    parser.add_argument('--train_prompt', action='store_true', help='Train prompt parameters during adaptation.')
    parser.add_argument('--no-train_prompt', dest='train_prompt', action='store_false')
    parser.set_defaults(train_prompt=True)
    parser.add_argument('--train_data_adaptor', action='store_true', help='Train data_adaptor during adaptation.')
    parser.add_argument('--no-train_data_adaptor', dest='train_data_adaptor', action='store_false')
    parser.set_defaults(train_data_adaptor=True)
    parser.add_argument('--train_vida', action='store_true', help='Train ViDA parameters during adaptation.')
    parser.add_argument('--no-train_vida', dest='train_vida', action='store_false')
    parser.set_defaults(train_vida=True)

    parser.add_argument('--lr_bn', type=float, default=None, help='LR for BN params (defaults to initial_lr).')
    parser.add_argument('--lr_prompt', type=float, default=1e-3, help='LR for prompt params.')
    parser.add_argument('--lr_data', type=float, default=1e-3, help='LR for data_adaptor params.')
    parser.add_argument('--lr_vida', type=float, default=1e-4, help='LR for ViDA params.')
    parser.add_argument('--wd_vida', type=float, default=1e-5, help='Weight decay for ViDA params.')
    parser.add_argument('--wd_other', type=float, default=0.0, help='Weight decay for non-ViDA params.')
    parser.add_argument('--clip_guided', action='store_true', help='Enable CLIP-prior guided online updates.')
    parser.add_argument('--clip_prior_root', type=str, default='data/MRI_prostate/clip_priors', help='Root folder of CLIP priors.')
    parser.add_argument(
        '--clip_prior_kind',
        type=str,
        default='clip',
        help="Prior kind: 'clip' (raw saliency) or 'ellip' (ellipse blob around peak). Also used as on-disk suffix.",
    )
    parser.add_argument('--clip_otf', action='store_true', help='Compute BioMedCLIP saliency on-the-fly per slice (per test image).')
    parser.add_argument('--clip_model_root', type=str, default='.', help='Path containing open_clip_config.json/open_clip_pytorch_model.bin/biomedbert/.')
    parser.add_argument('--clip_prompt', action='append', dest='clip_prompts', default=None, help='BioMedCLIP text prompt (repeatable).')
    parser.add_argument('--clip_softmax_temp', type=float, default=0.06, help='Patch softmax temperature (smaller => sharper).')
    parser.add_argument('--clip_patch_sigma', type=float, default=3.5, help='Blur sigma on patch saliency at 224px (smaller => less spread).')
    parser.add_argument('--clip_blur_sigma', type=float, default=2.0, help='Additional blur sigma on final saliency at HxW.')
    parser.add_argument('--clip_roi', type=float, nargs=4, default=None, metavar=('Y1','Y2','X1','X2'), help='Optional ROI crop ratios for CLIP saliency.')
    parser.add_argument('--clip_roi_feather', type=float, default=0.10, help='Feather ratio for ROI edges (0 disables).')
    # If clip_prior_kind is 'ellip', convert raw saliency into a coarse ellipse cue (more robust than raw heatmaps).
    parser.add_argument('--clip_ellip_sigma_x', type=float, default=0.0, help='Ellipse sigma-x in pixels (0 => 0.16*W).')
    parser.add_argument('--clip_ellip_sigma_y', type=float, default=0.0, help='Ellipse sigma-y in pixels (0 => 0.12*H).')
    parser.add_argument('--clip_ellip_power', type=float, default=1.0, help='Power on ellipse blob (higher => sharper center).')
    parser.add_argument('--clip_ellip_sym', action='store_true', help='Symmetrize saliency before ellipse (reduce side distractions).')
    parser.add_argument('--no-clip_ellip_sym', dest='clip_ellip_sym', action='store_false')
    parser.set_defaults(clip_ellip_sym=True)
    parser.add_argument('--clip_ellip_smooth_sigma', type=float, default=2.0, help='Pre-smooth sigma before finding peak (pixels).')
    parser.add_argument('--clip_weight_gamma', type=float, default=1.0, help='Gamma on CLIP prior weights.')
    parser.add_argument('--clip_weight_min', type=float, default=0.05, help='Min weight floor to avoid zeroing gradients.')
    parser.add_argument(
        '--clip_auto_blend',
        action='store_true',
        help='Auto-blend CLIP prior with a generic anatomical blob based on per-case confidence (unified across domains).',
    )
    parser.add_argument('--no-clip_auto_blend', dest='clip_auto_blend', action='store_false')
    parser.set_defaults(clip_auto_blend=True)
    parser.add_argument(
        '--clip_auto_weight',
        action='store_true',
        help='Auto-scale CLIP-guided losses (loc/bce) by per-case confidence (unified across domains).',
    )
    parser.add_argument('--no-clip_auto_weight', dest='clip_auto_weight', action='store_false')
    parser.set_defaults(clip_auto_weight=True)
    parser.add_argument(
        '--clip_auto_weight_floor',
        type=float,
        default=0.35,
        help='Floor for auto-weight scaling (keeps some CLIP/generic guidance even when confidence is low).',
    )
    parser.add_argument('--clip_rel_thr', type=float, default=0.0, help='Reliability threshold for source-vs-prior agreement (sigmoid gate).')
    parser.add_argument('--clip_rel_temp', type=float, default=0.1, help='Reliability temperature for source-vs-prior agreement (sigmoid gate).')
    parser.add_argument(
        '--clip_rel_center_sigma',
        type=float,
        default=0.25,
        help='Sigma (normalized) for center-agreement penalty between source centroid and prior peak (smaller => stricter).',
    )
    parser.add_argument(
        '--clip_rel_src_area_min',
        type=float,
        default=0.002,
        help='If source prediction foreground area-ratio is below this, skip source-agreement penalties when computing CLIP reliability.',
    )
    parser.add_argument(
        '--clip_slice_sigma',
        type=float,
        default=3.0,
        help='If offline prior provides slice_idx, apply a Gaussian gate over slice distance with this sigma (in slices).',
    )
    parser.add_argument('--clip_slice_auto_gate', action='store_true', help='Auto-disable slice gating when slice_idx appears mismatched (domain-agnostic).')
    parser.add_argument('--clip_slice_auto_tau', type=float, default=6.0, help='Tau for slice_idx trust decay vs |z_pred - slice_idx|.')
    parser.add_argument('--clip_slice_auto_max_delta', type=int, default=8, help='If |z_pred - slice_idx| > this, treat slice_idx as untrusted.')
    parser.add_argument('--clip_slice_auto_min_trust', type=float, default=0.35, help='Below this trust, force untrusted sigma.')
    parser.add_argument('--clip_slice_sigma_untrusted', type=float, default=999.0, help='Sigma used when slice_idx is untrusted (large => effectively disables gating).')
    parser.add_argument(
        '--clip_slice_radius',
        type=int,
        default=0,
        help='Optional hard radius (in slices) for slice-distance gating (0 disables).',
    )
    parser.add_argument('--clip_align_lambda', type=float, default=0.0, help='Extra prior alignment loss weight.')
    parser.add_argument('--clip_loc_lambda', type=float, default=0.0, help='Localization loss weight using saliency (mass_out - mass_in).')
    parser.add_argument('--clip_loc_power', type=float, default=1.0, help='Power on saliency weights for localization loss.')
    parser.add_argument('--clip_select', type=str, default='entropy', choices=['entropy', 'align'], help='How to choose between adapted and source predictions.')
    parser.add_argument('--final_select', action='store_true', help='Also select between adapted and source for the final output (dice_final).')
    parser.add_argument('--clip_entropy_weighted', action='store_true', help='Use weighted entropy when clip_guided.')
    parser.add_argument('--no-clip_entropy_weighted', dest='clip_entropy_weighted', action='store_false')
    parser.set_defaults(clip_entropy_weighted=True)
    parser.add_argument('--clip_bce_lambda', type=float, default=0.0, help='Lambda for soft prior BCE-with-logits loss.')
    parser.add_argument('--clip_bce_gamma', type=float, default=1.0, help='Gamma on prior weights for BCE.')
    parser.add_argument('--clip_bce_weight_min', type=float, default=0.0, help='Min weight floor for BCE weights.')

    # CLIP-gated AMU/AUM (teacher EMA update gate)
    parser.add_argument('--aum_gate', action='store_true', help='Gate AMU/AUM (EMA teacher update) by a quality signal per batch.')
    parser.add_argument('--aum_gate_metric', type=str, default='clip_align', choices=['clip_align', 'teacher_conf', 'entropy'], help='Signal used for AMU/AUM gate.')
    parser.add_argument('--aum_gate_thr', type=float, default=0.0, help='Threshold for AMU/AUM gate signal.')
    parser.add_argument('--aum_gate_temp', type=float, default=0.1, help='Temperature for AMU/AUM gate sigmoid (0 => hard).')
    parser.add_argument('--aum_gate_floor', type=float, default=0.0, help='Min gate value in [0,1) to avoid freezing teacher.')

    # Safe Branch Selector: domain-agnostic auto-selection between source / adapted / final outputs.
    parser.add_argument(
        '--safe_branch_selector',
        action='store_true',
        help='Per-case auto-select the most reliable output branch (source, adapted, or final) using test-time reliability signals only.',
    )
    parser.add_argument(
        '--safe_branch_selector_area_min',
        type=float,
        default=0.002,
        help='Min FG area ratio for safe_branch_selector.',
    )
    parser.add_argument(
        '--safe_branch_selector_area_max',
        type=float,
        default=0.10,
        help='Max FG area ratio for safe_branch_selector.',
    )
    parser.add_argument(
        '--safe_branch_selector_center_max',
        type=float,
        default=0.30,
        help='Max normalized center distance for safe_branch_selector.',
    )
    parser.add_argument(
        '--safe_branch_selector_conservative',
        action='store_true',
        help='Use conservative safe-branch selection: final output must improve reliability over source/adapted by a margin.',
    )
    parser.add_argument(
        '--safe_branch_selector_adaptive',
        action='store_true',
        help='Adaptively use conservative final selection only when source/adapted is plausible and final appears inflated.',
    )
    parser.add_argument(
        '--safe_branch_selector_final_min_gain',
        type=float,
        default=0.02,
        help='Minimum reliability gain required for final branch in conservative safe_branch_selector.',
    )
    parser.add_argument(
        '--safe_branch_selector_final_bias',
        type=float,
        default=0.03,
        help='Reliability bonus for the final/postprocessed branch when conservative mode is disabled.',
    )

    # Case-level fallback / protection (avoid catastrophic drift on hard cases)
    parser.add_argument('--case_fallback', action='store_true', help='Revert Dice_final to source for a case if adaptation looks worse by a proxy.')
    parser.add_argument('--case_fallback_metric', type=str, default='entropy', choices=['entropy', 'clip_align', 'teacher_conf'], help='Proxy metric for case fallback.')
    parser.add_argument('--case_fallback_margin', type=float, default=0.0, help='Allowed degradation margin before fallback (metric-dependent).')

    # Case-level rollback (weight protection): restore pre-case student/teacher weights if a proxy deems the update harmful.
    parser.add_argument('--case_rollback', action='store_true', help='Rollback student/teacher weights for a case if a proxy deems the update harmful.')
    parser.add_argument(
        '--case_rollback_metric',
        type=str,
        default='entropy',
        choices=['entropy', 'clip_align', 'teacher_conf'],
        help='Proxy metric for case rollback.',
    )
    parser.add_argument('--case_rollback_margin', type=float, default=0.0, help='Allowed degradation margin before rollback (metric-dependent).')
    parser.add_argument('--case_rollback_area_min', type=float, default=None, help='If set, reject updates when FG area ratio < this (proxy).')
    parser.add_argument('--case_rollback_area_max', type=float, default=None, help='If set, reject updates when FG area ratio > this (proxy).')
    parser.add_argument('--case_rollback_center_max', type=float, default=None, help='If set, reject updates when prediction centroid is too far from the anatomical prior peak.')
    parser.add_argument('--case_rollback_center_margin', type=float, default=None, help='If set, reject updates when center distance worsens over the pre-case snapshot by this margin.')
    parser.add_argument(
        '--case_guard_area_min',
        type=float,
        default=0.002,
        help='Hard guard: if final FG area ratio < this, replace preds with source and rollback weights.',
    )
    parser.add_argument(
        '--case_guard_area_max',
        type=float,
        default=0.10,
        help='Hard guard: if final FG area ratio > this, replace preds with source and rollback weights.',
    )
    parser.add_argument(
        '--case_guard_center_max',
        type=float,
        default=None,
        help='Hard guard: if prediction centroid is farther than this normalized distance from the CLIP prior peak, replace preds with source and rollback weights.',
    )
    parser.add_argument(
        '--case_guard_center_relax',
        type=float,
        default=0.25,
        help='Relaxation added to case_guard_center_max when CLIP confidence is low (unified across domains).',
    )
    parser.add_argument(
        '--case_guard_center_qmin',
        type=float,
        default=0.15,
        help='Only enforce center guard when CLIP confidence >= this (unified across domains).',
    )

    # Case-wise update gate / suppression (skip or down-weight updates for low-quality cases)
    parser.add_argument('--case_gate', action='store_true', help='Enable case-wise update gate to suppress harmful updates.')
    parser.add_argument(
        '--case_gate_metric',
        type=str,
        default='clip_align',
        choices=['clip_align', 'entropy', 'teacher_conf'],
        help='Quality signal used for case-wise gate.',
    )
    parser.add_argument('--case_gate_thr', type=float, default=0.0, help='Threshold for case-wise gate signal.')
    parser.add_argument('--case_gate_temp', type=float, default=0.1, help='Temperature for case-wise gate sigmoid (0 => hard).')
    parser.add_argument('--case_gate_floor', type=float, default=0.0, help='Min gate value in [0,1) to avoid freezing.')
    parser.add_argument('--case_gate_skip_thr', type=float, default=0.0, help='If gate < this value, skip adaptation for the entire case (0 disables).')
    parser.add_argument('--case_gate_scale_clip', action='store_true', help='Scale CLIP-guided losses (loc/bce/ts) by the case gate value.')

    # Metrics-time postprocess: suppress tiny distant false positives that dominate HD95.
    parser.add_argument('--post_lcc', action='store_true', help='Keep largest connected component in 3D prediction before computing metrics.')
    parser.add_argument('--post_lcc_min_size', type=int, default=0, help='If >0, drop all components when the largest is smaller than this size (voxels).')
    parser.add_argument(
        '--post_cc_by_prior',
        action='store_true',
        help='Keep the 3D connected component that best aligns with the CLIP prior (more robust than largest-CC for HD95).',
    )
    parser.add_argument(
        '--post_cc_by_prior_qmin',
        type=float,
        default=0.0,
        help='Only apply CC-by-CLIP-prior when CLIP case reliability (q_mean) >= this; set 0 to disable this gate.',
    )
    parser.add_argument(
        '--post_cc_by_prior_min_score',
        type=float,
        default=0.0,
        help='If >0, fall back to largest-CC when the best prior-alignment score is below this value.',
    )

    # Teacher-student consistency (EMA teacher) options
    parser.add_argument('--ts_lambda', type=float, default=0.0, help='Teacher-student consistency loss weight (EMA teacher).')
    parser.add_argument('--ts_weight_with_clip', action='store_true', help='Use CLIP saliency as spatial weights for TS consistency.')
    parser.add_argument('--ts_weight_gamma', type=float, default=1.0, help='Gamma on TS spatial weights.')
    parser.add_argument('--ts_weight_min', type=float, default=0.0, help='Min floor for TS spatial weights.')
    parser.add_argument('--ts_conf_thr', type=float, default=0.12, help='Teacher confidence threshold for TS gating.')
    parser.add_argument('--ts_conf_temp', type=float, default=0.08, help='Temperature for smooth TS gate (0 => hard gate).')
    parser.add_argument('--ts_weak_noise', type=float, default=0.01, help='Weak augment noise std.')
    parser.add_argument('--ts_weak_brightness', type=float, default=0.05, help='Weak augment brightness range.')
    parser.add_argument('--ts_weak_contrast', type=float, default=0.05, help='Weak augment contrast range.')
    parser.add_argument('--ts_strong_noise', type=float, default=0.03, help='Strong augment noise std.')
    parser.add_argument('--ts_strong_brightness', type=float, default=0.10, help='Strong augment brightness range.')
    parser.add_argument('--ts_strong_contrast', type=float, default=0.10, help='Strong augment contrast range.')
    parser.add_argument('--ts_strong_gamma', type=float, default=0.0, help='Strong augment gamma range (0 disables).')
    parser.add_argument('--ts_early_stop', action='store_true', help='Early-stop adaptation steps if loss increases.')
    parser.add_argument('--ts_early_stop_tol', type=float, default=0.0, help='Tolerance for early-stop (loss increase).')

    args = parser.parse_args()
    cli_tag = args.tag
    from scripts.tta_configs import (
        Prostate_RUNMC2BMC_config,
        Prostate_RUNMC2BIDMC_config,
        Prostate_RUNMC2HK_config,
        Prostate_RUNMC2I2CVB_config,
        Prostate_RUNMC2UCL_config,
    )
    domain_cfg = {
        'BMC': Prostate_RUNMC2BMC_config,
        'BIDMC': Prostate_RUNMC2BIDMC_config,
        'HK': Prostate_RUNMC2HK_config,
        'I2CVB': Prostate_RUNMC2I2CVB_config,
        'UCL': Prostate_RUNMC2UCL_config,
    }
    args.__dict__.update(domain_cfg[args.domain])
    if getattr(args, "override_tag", False):
        args.tag = cli_tag
    if args.override_ts_csv:
        args.ts_csv = args.override_ts_csv
    if args.override_batch_size is not None:
        args.batch_size = int(args.override_batch_size)
    if args.override_model_episodic is not None:
        args.model_episodic = bool(int(args.override_model_episodic))
    if args.override_optim_steps is not None:
        args.optim_steps = int(args.override_optim_steps)
    if args.override_initial_lr is not None:
        args.initial_lr = float(args.override_initial_lr)
    if args.lr_bn is None:
        args.lr_bn = args.initial_lr
    if args.exp_name:
        args.tag = args.tag + '_' + str(args.exp_name)
    args.tag = args.tag +'_sptta'
    adpater = Outer_Adapter(args)
    adpater.evaluate()
