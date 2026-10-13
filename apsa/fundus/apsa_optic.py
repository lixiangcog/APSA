import os
import sys
import argparse
import csv
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import Logger
from utils.convert import AdaBN
from utils.memory import Memory
from utils.prompt import Prompt
from utils.metrics import calculate_metrics
from networks.ResUnet_TTA import ResUnet
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list


torch.set_num_threads(1)


class ShiftDecomposedAdapter(nn.Module):
    """
    Lightweight ViDA-style adapter for fundus CTTA.
    - SSB: shift-sensitive branch (depthwise 3x3 + pointwise 1x1)
    - SIB: shift-invariant branch (low-rank 1x1 bottleneck)
    - SAF: shift-aware fusion (global-pooled gate -> alpha/beta)
    """

    def __init__(self, in_ch: int = 3, rank: int = 4, init_scale: float = 0.0):
        super().__init__()
        r = int(max(2, rank))
        self.ssb = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
        )
        self.sib_down = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.sib_up = nn.Conv2d(r, in_ch, kernel_size=1, bias=False)
        self.saf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(8, 2 * r), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, 2 * r), 2, kernel_size=1),
        )
        # Learnable residual strength. Start near identity for stability.
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.saf(x)  # [B,2,1,1]
        g = torch.softmax(g, dim=1)
        alpha = g[:, 0:1]
        beta = g[:, 1:2]
        y_ssb = self.ssb(x)
        y_sib = self.sib_up(self.sib_down(x))
        # Residual fusion: stable and parameter-efficient.
        return x + self.scale * (alpha * y_ssb + beta * y_sib)


def _build_target_csvs(targets: List[str]) -> List[str]:
    target_test_csv: List[str] = []
    for t in targets:
        if t != "REFUGE_Valid":
            target_test_csv.append(t + "_train.csv")
            target_test_csv.append(t + "_test.csv")
        else:
            target_test_csv.append(t + ".csv")
    return target_test_csv


def _optic_overlay_uint8(image_chw01: np.ndarray, disc: np.ndarray, cup: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    image = np.transpose(np.clip(image_chw01, 0.0, 1.0), (1, 2, 0))
    out = (image * 255.0).astype(np.float32)
    disc_color = np.array([30, 144, 255], dtype=np.float32)
    cup_color = np.array([255, 80, 80], dtype=np.float32)
    disc = disc.astype(bool)
    cup = cup.astype(bool)
    out[disc] = (1.0 - alpha) * out[disc] + alpha * disc_color
    out[cup] = (1.0 - alpha) * out[cup] + alpha * cup_color
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_optic_prediction_visuals(
    out_dir: str,
    index: int,
    name: str,
    image_chw01: np.ndarray,
    gt_2hw: np.ndarray,
    pred_2hw: np.ndarray,
    disc_thr: float,
    cup_thr: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.splitext(os.path.basename(str(name)))[0])[0]
    base = (np.transpose(np.clip(image_chw01, 0.0, 1.0), (1, 2, 0)) * 255.0).astype(np.uint8)
    gt_overlay = _optic_overlay_uint8(image_chw01, gt_2hw[0] > 0.5, gt_2hw[1] > 0.5)
    pred_disc = pred_2hw[0] > float(disc_thr)
    pred_cup = pred_2hw[1] > float(cup_thr)
    pred_overlay = _optic_overlay_uint8(image_chw01, pred_disc, pred_cup)
    pred_mask = np.zeros_like(base)
    pred_mask[pred_disc] = [30, 144, 255]
    pred_mask[pred_cup] = [255, 80, 80]
    prefix = f"{index:04d}_{stem}"
    Image.fromarray(base).save(os.path.join(out_dir, prefix + "_image.png"))
    Image.fromarray(gt_overlay).save(os.path.join(out_dir, prefix + "_gt_overlay.png"))
    Image.fromarray(pred_overlay).save(os.path.join(out_dir, prefix + "_ragtta_overlay.png"))
    Image.fromarray(pred_mask).save(os.path.join(out_dir, prefix + "_ragtta_mask.png"))


def _safe_case_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)


def _optic_domain_from_name(name: str) -> str:
    parts = str(name).replace("\\", "/").split("/")
    known = {"RIM_ONE_r3", "REFUGE", "ORIGA", "REFUGE_Valid", "Drishti_GS"}
    for p in parts:
        if p in known:
            return p
    return "unknown"


def _optic_violation_metrics(pred_2hw: np.ndarray) -> Dict[str, float]:
    m = _optic_binary_anatomy_metrics(pred_2hw)
    return {
        "pred_disc_area_frac": m["disc_area_frac"],
        "pred_cup_area_frac": m["cup_area_frac"],
        "pred_cdr_area": m["cdr_area"],
        "oc_outside_od_frac": m["oc_outside_od_frac"],
        "oc_outside_od_area": m["oc_outside_od_area"],
        "oc_outside_od_flag": m["oc_outside_od_flag"],
        "oc_area_gt_od_flag": m["oc_area_gt_od_flag"],
        "abnormal_cdr_flag": m["abnormal_cdr_flag"],
        "od_empty_flag": m["od_empty_flag"],
        "oc_empty_flag": m["oc_empty_flag"],
    }


def _optic_binary_anatomy_metrics(mask_2hw: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(mask_2hw)
    disc = pred[0] > 0.5
    cup = pred[1] > 0.5
    disc_area = float(disc.sum())
    cup_area = float(cup.sum())
    image_area = float(max(1, disc.size))
    cup_outside = np.logical_and(cup, np.logical_not(disc))
    cup_outside_px = float(cup_outside.sum())
    cup_outside_frac = cup_outside_px / max(1.0, cup_area)
    cup_outside_area = cup_outside_px / image_area
    cdr_area = cup_area / max(1.0, disc_area)
    return {
        "disc_area_frac": disc_area / image_area,
        "cup_area_frac": cup_area / image_area,
        "cdr_area": cdr_area,
        "oc_outside_od_frac": cup_outside_frac,
        "oc_outside_od_area": cup_outside_area,
        "oc_outside_od_flag": float(cup_outside_frac > 0.01),
        "oc_area_gt_od_flag": float(cup_area > disc_area),
        "abnormal_cdr_flag": float((cdr_area < 0.05) or (cdr_area > 0.90)),
        "od_empty_flag": float(disc_area < 1.0),
        "oc_empty_flag": float(cup_area < 1.0),
    }


def _optic_prob_anatomy_metrics(seg_prob: torch.Tensor) -> Dict[str, float]:
    """Soft and thresholded OD/OC anatomy metrics for diagnostics.

    These are logged before/after online updates so we can detect CDR drift and
    topology violations that final postprocess may hide.
    """
    if seg_prob.dim() == 4:
        prob = seg_prob[0].detach()
    else:
        prob = seg_prob.detach()
    prob = torch.clamp(prob, 0.0, 1.0)
    disc = prob[0]
    cup = prob[1]
    disc_area = float(disc.mean().detach().cpu().item())
    cup_area = float(cup.mean().detach().cpu().item())
    soft_cdr = float(cup_area / max(1e-6, disc_area))
    soft_cup_out = float(torch.relu(cup - disc).sum().detach().cpu().item() / (cup.sum().detach().cpu().item() + 1e-6))
    hard = _optic_binary_anatomy_metrics((prob.detach().cpu().numpy() > 0.5).astype(np.uint8))
    out = {
        "soft_disc_area_frac": disc_area,
        "soft_cup_area_frac": cup_area,
        "soft_cdr_area": soft_cdr,
        "soft_oc_outside_od_frac": soft_cup_out,
    }
    out.update(hard)
    return out


def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {prefix + str(k): float(v) for k, v in metrics.items()}


def _build_target_csvs_mode(targets: List[str], mode: str) -> List[str]:
    mode = (mode or "train_test").lower().strip()
    if mode not in ("train_test", "test"):
        mode = "train_test"
    if mode == "train_test":
        return _build_target_csvs(targets)
    # test-only
    out: List[str] = []
    for t in targets:
        if t != "REFUGE_Valid":
            out.append(t + "_test.csv")
        else:
            out.append(t + ".csv")
    return out


def _soft_entropy_bernoulli(p: torch.Tensor) -> torch.Tensor:
    # p in [0,1]
    p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _soft_centroid_2d(prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # prob: (H,W)
    h, w = prob.shape[-2], prob.shape[-1]
    yy = torch.linspace(0.0, 1.0, h, device=prob.device, dtype=prob.dtype).view(h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=prob.device, dtype=prob.dtype).view(1, w)
    m = torch.clamp(prob, min=0.0)
    s = m.sum() + 1e-6
    cy = (m * yy).sum() / s
    cx = (m * xx).sum() / s
    return cy, cx


def _central_gaussian_prior(h: int, w: int, sigma: float = 0.22) -> torch.Tensor:
    yy = torch.linspace(0.0, 1.0, h).view(h, 1)
    xx = torch.linspace(0.0, 1.0, w).view(1, w)
    cy, cx = 0.5, 0.5
    s2 = float(sigma) * float(sigma)
    prior = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s2))
    prior = prior / (prior.max() + 1e-6)
    return prior.to(dtype=torch.float32)


def _gaussian_prior_at(h: int, w: int, cy_px: float, cx_px: float, sigma: float) -> torch.Tensor:
    """Return a (H,W) Gaussian prior centered at (cy_px,cx_px) in pixels."""
    cy = float(cy_px) / max(1.0, float(h - 1))
    cx = float(cx_px) / max(1.0, float(w - 1))
    yy = torch.linspace(0.0, 1.0, h).view(h, 1)
    xx = torch.linspace(0.0, 1.0, w).view(1, w)
    s2 = float(sigma) * float(sigma) + 1e-12
    prior = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s2))
    prior = prior / (prior.max() + 1e-6)
    return prior.to(dtype=torch.float32)


def _coarse_od_roi_prior_at(
    h: int,
    w: int,
    cy_px: float,
    cx_px: float,
    r_px: float,
    roi_scale: float = 2.50,
    roi_thr: float = 0.20,
    min_sigma: float = 0.035,
    max_sigma: float = 0.36,
) -> torch.Tensor:
    """Gaussian soft ROI whose thresholded support is a coarse OD neighborhood.

    The prior is not a pseudo-label. With a Gaussian prior, the radius of the
    level set p>=roi_thr is sigma * sqrt(2 log(1/roi_thr)). We choose sigma so
    that this support roughly corresponds to roi_scale times the estimated OD
    radius, matching the standalone prior-quality evaluation.
    """
    r_norm = float(max(1.0, r_px)) / max(1.0, float(min(h, w)))
    support_r = float(max(0.03, min(0.45, roi_scale * r_norm)))
    thr = float(max(1e-4, min(0.95, roi_thr)))
    denom = float(np.sqrt(max(1e-6, 2.0 * np.log(1.0 / thr))))
    sigma = support_r / max(1e-6, denom)
    sigma = float(max(min_sigma, min(max_sigma, sigma)))
    return _gaussian_prior_at(h, w, cy_px=cy_px, cx_px=cx_px, sigma=sigma)


def _weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = torch.clamp(w, 0.0, 1.0)
    return (x * w).sum() / (w.sum() + 1e-6)


def _roi_weight_from_prior(prior01_hw: torch.Tensor, w_min: float = 0.20, gamma: float = 1.0) -> torch.Tensor:
    """Map a (H,W) prior in [0,1] to a soft ROI weight map in [w_min,1]."""
    p = torch.clamp(prior01_hw, 0.0, 1.0).to(dtype=torch.float32)
    g = float(max(0.3, min(4.0, gamma)))
    p = p**g
    w_min = float(max(0.0, min(0.95, w_min)))
    return (w_min + (1.0 - w_min) * p).to(dtype=torch.float32)


def _prior_outside_weight(prior01_hw: torch.Tensor, roi_thr: float = 0.20, power: float = 2.0) -> torch.Tensor:
    """Soft outside-ROI weight: 0 inside the prior support, 1 far outside."""
    p = torch.clamp(prior01_hw, 0.0, 1.0).to(dtype=torch.float32)
    thr = float(max(1e-4, min(0.95, roi_thr)))
    outside = torch.relu(thr - p) / thr
    outside = torch.clamp(outside, 0.0, 1.0)
    pow_v = float(max(0.5, min(6.0, power)))
    return outside**pow_v


def _mass_outside_prior_ratio(
    prob: torch.Tensor,
    prior01_hw: torch.Tensor,
    roi_thr: float = 0.20,
    power: float = 2.0,
) -> torch.Tensor:
    """Fraction of soft mask mass outside a coarse anatomical prior ROI.

    Unlike BCE to a Gaussian prior, this does not prescribe the object shape inside the
    ROI. It only discourages confident mass far away from the anatomical support.
    """
    q = torch.clamp(prob, 0.0, 1.0)
    w = _prior_outside_weight(prior01_hw.to(device=q.device, dtype=torch.float32), roi_thr=roi_thr, power=power)
    while w.dim() < q.dim():
        w = w.unsqueeze(0)
    w = w.to(device=q.device, dtype=q.dtype)
    return (q * w).sum() / (q.sum() + 1e-6)


def _fuse_prior_candidates(
    candidates: List[Tuple[torch.Tensor, float, float]],
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted fusion of multiple coarse OD localization priors.

    Each candidate is (prior_map, confidence, base_weight). The output remains a
    normalized soft support map, not a segmentation pseudo-label.
    """
    acc: Optional[torch.Tensor] = None
    total_w = 0.0
    fb = _normalize01_hw(fallback.detach().to(dtype=torch.float32))
    for prior, conf, base_w in candidates:
        try:
            w = float(max(0.0, min(1.0, conf))) * float(max(0.0, base_w))
        except Exception:
            w = 0.0
        if w <= 1e-8:
            continue
        p = _normalize01_hw(prior.detach().to(device=fb.device, dtype=torch.float32))
        if p.shape != fb.shape:
            continue
        acc = p * w if acc is None else acc + p * w
        total_w += w
    if acc is None or total_w <= 1e-8:
        return fb
    return _normalize01_hw(acc / float(total_w))


def _weighted_confidence(votes: List[Tuple[float, float]]) -> float:
    total = 0.0
    weight = 0.0
    for conf, base_w in votes:
        try:
            c = float(max(0.0, min(1.0, conf)))
            w = float(max(0.0, base_w))
        except Exception:
            continue
        if w <= 1e-8:
            continue
        total += c * w
        weight += w
    return float(total / weight) if weight > 1e-8 else 0.0


def _prior_prediction_agreement_conf(
    seg_prob: torch.Tensor,
    prior01_hw: torch.Tensor,
    roi_thr: float = 0.20,
    power: float = 2.0,
) -> float:
    """Confidence that the pre-update OD prediction and prior describe the same region."""
    if seg_prob.dim() != 4 or seg_prob.shape[0] != 1 or seg_prob.shape[1] < 1:
        return 0.0
    disc = torch.clamp(seg_prob[0, 0], 0.0, 1.0)
    prior = torch.clamp(prior01_hw.to(device=disc.device, dtype=disc.dtype), 0.0, 1.0)
    try:
        cy, cx = _soft_centroid_2d(disc)
        py, px = _soft_centroid_2d(prior)
        center_dist = float(torch.sqrt((cy - py) ** 2 + (cx - px) ** 2).detach().cpu().item())
        center_conf = max(0.0, min(1.0, (0.22 - center_dist) / 0.22))
        outside = float(
            _mass_outside_prior_ratio(disc, prior, roi_thr=roi_thr, power=power)
            .detach()
            .cpu()
            .item()
        )
        outside_conf = max(0.0, min(1.0, (0.65 - outside) / 0.65))
        area = float(disc.mean().detach().cpu().item())
        low_conf = max(0.0, min(1.0, area / 0.006))
        high_conf = max(0.0, min(1.0, (0.35 - area) / 0.20))
        area_conf = min(low_conf, high_conf)
    except Exception:
        return 0.0
    return float(max(0.0, min(1.0, 0.45 * center_conf + 0.35 * outside_conf + 0.20 * area_conf)))


def _prior_confidence_scale(conf: float, min_conf: float = 0.45, power: float = 1.0) -> float:
    q = float(max(0.0, min(1.0, conf)))
    lo = float(max(0.0, min(0.95, min_conf)))
    if q <= lo:
        return 0.0
    scale = (q - lo) / max(1e-6, 1.0 - lo)
    p = float(max(0.2, min(4.0, power)))
    return float(max(0.0, min(1.0, scale**p)))


def _normalize01_hw(x: torch.Tensor) -> torch.Tensor:
    """Normalize a 2D map to [0,1] with max scaling while preserving zero baseline."""
    t = torch.clamp(x.to(dtype=torch.float32), min=0.0)
    return t / (t.max() + 1e-6)


def _estimate_fov_mask_from_x01(x01: torch.Tensor, thr: float = 0.05, erode: int = 5) -> np.ndarray:
    """Estimate fundus field-of-view mask from normalized image tensor (1,C,H,W)."""
    from scipy.ndimage import binary_closing, binary_erosion, label

    x = x01.detach().float().cpu().numpy()[0]  # C,H,W
    if x.shape[0] >= 3:
        img = np.transpose(x[:3], (1, 2, 0)).mean(axis=2)
    else:
        img = x.mean(axis=0)
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    m = (img > float(thr))
    labels, num = label(m)
    if num <= 0:
        return np.ones_like(img, dtype=np.uint8)
    areas = [(labels == k).sum() for k in range(1, num + 1)]
    k = 1 + int(np.argmax(np.asarray(areas, dtype=np.int64)))
    m = (labels == k)
    m = binary_closing(m, structure=np.ones((11, 11), dtype=bool))
    er = int(max(0, erode))
    if er > 0:
        m = binary_erosion(m, structure=np.ones((er, er), dtype=bool))
    return m.astype(np.uint8)


def _estimate_disc_geom_center_radius(x01: torch.Tensor) -> Tuple[Optional[Tuple[float, float, float]], float, np.ndarray]:
    """Estimate disc (cy,cx,r) in pixels + confidence + fov mask (H,W uint8)."""
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    from scipy.ndimage import gaussian_filter, label

    if x01.dim() != 4 or x01.shape[0] != 1:
        return None, 0.0, np.ones((1, 1), dtype=np.uint8)

    x = x01.detach().float().cpu().numpy()[0]  # C,H,W
    h, w = int(x.shape[1]), int(x.shape[2])
    if h < 32 or w < 32:
        return None, 0.0, np.ones((h, w), dtype=np.uint8)

    fov = _estimate_fov_mask_from_x01(x01)

    if x.shape[0] >= 2:
        img = x[1]
    else:
        img = x.mean(axis=0)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    img_blur = gaussian_filter(img, sigma=2.5)

    # Candidate threshold on fov pixels only (no "center bias" — disc can be off-center).
    vals = img_blur[fov > 0]
    if vals.size < 64:
        return None, 0.0, fov
    thr = float(np.percentile(vals, 92.0))
    bw = ((img_blur >= thr) & (fov > 0)).astype(np.uint8)

    if cv2 is not None:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num <= 1:
            return None, 0.0, fov

        best_k = None
        best_score = -1e18
        best_feat = None
        for k in range(1, num):
            area = float(stats[k, cv2.CC_STAT_AREA])
            if area < 0.002 * h * w or area > 0.30 * h * w:
                continue
            m = (labels_cc == k).astype(np.uint8)
            if m.sum() <= 0:
                continue
            mean_int = float(img_blur[m > 0].mean())
            circularity = 0.0
            try:
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    per = float(cv2.arcLength(c, True))
                    if per > 1e-6:
                        circularity = float(4.0 * np.pi * area / (per * per + 1e-6))
            except Exception:
                circularity = 0.0
            circularity = float(max(0.0, min(1.0, circularity)))
            size_pref = 1.0 - abs(area / (0.06 * h * w) - 1.0)
            size_pref = float(max(0.0, min(1.0, size_pref)))
            score = area * (0.35 + 0.45 * circularity + 0.20 * mean_int) * (0.55 + 0.45 * size_pref)
            if score > best_score:
                best_score = score
                best_k = k
                best_feat = (area, circularity, mean_int, size_pref)

        if best_k is None:
            return None, 0.0, fov
        m = (labels_cc == best_k).astype(np.uint8)
        ys, xs = np.where(m > 0)
        if ys.size == 0:
            return None, 0.0, fov
        gy = float(ys.mean())
        gx = float(xs.mean())
        area, circularity, mean_int, size_pref = best_feat
    else:
        # Fallback: scipy connected-components (no circularity).
        labels, num = label(bw > 0)
        if num <= 0:
            return None, 0.0, fov
        best = None
        best_score = -1e18
        for k in range(1, num + 1):
            area = float((labels == k).sum())
            if area < 0.002 * h * w or area > 0.30 * h * w:
                continue
            ys, xs = np.where(labels == k)
            if ys.size == 0:
                continue
            mean_int = float(img_blur[labels == k].mean())
            score = area * (0.55 + 0.45 * mean_int)
            if score > best_score:
                best_score = score
                best = (ys.mean(), xs.mean(), area, mean_int)
        if best is None:
            return None, 0.0, fov
        gy, gx, area, mean_int = best
        circularity = 0.5
        size_pref = 0.5

    r = float(np.sqrt(max(float(area), 1.0) / np.pi))

    # Optional refinement: use strongest edge along multiple rays to avoid under-estimating radius
    # when the bright-core threshold captures only the disc center.
    rays_ratio = 0.0
    if cv2 is not None and r > 2.0:
        cx0 = float(gx)
        cy0 = float(gy)
        gx_img = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy_img = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx_img * gx_img + gy_img * gy_img).astype(np.float32)
        gmag *= (fov > 0).astype(np.float32)
        gmag = gmag / (float(gmag.max()) + 1e-6)

        r_min = float(max(2.0, 0.70 * r))
        r_max = float(min(0.49 * float(min(h, w)), max(r_min + 2.0, 1.85 * r)))
        steps = int(max(28, min(200, (r_max - r_min) * 2.0)))
        rs = np.linspace(r_min, r_max, steps, dtype=np.float32)

        num_rays = 72
        radii: List[float] = []
        for t in np.linspace(0.0, 2.0 * np.pi, int(max(8, num_rays)), endpoint=False):
            dx = float(np.cos(t))
            dy = float(np.sin(t))
            xs = cx0 + rs * dx
            ys = cy0 + rs * dy
            xi = np.clip(xs, 0, w - 1).reshape(1, -1).astype(np.float32)
            yi = np.clip(ys, 0, h - 1).reshape(1, -1).astype(np.float32)
            v = cv2.remap(gmag, xi, yi, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1)
            if float(v.max()) <= 1e-6:
                continue
            k = int(np.argmax(v))
            if float(v[k]) < 0.18:
                continue
            radii.append(float(rs[k]))
        rays_ratio = float(len(radii) / max(1, int(num_rays)))
        if len(radii) >= 10:
            r_rays = float(np.median(np.asarray(radii, dtype=np.float32)))
            r = float(max(r, r_rays))
    area_pref = max(0.0, min(1.0, float(area) / (0.08 * h * w)))
    conf = float(0.20 + 0.45 * area_pref + 0.25 * float(circularity) + 0.10 * float(mean_int))
    if rays_ratio > 0.0:
        conf *= float(0.80 + 0.20 * max(0.0, min(1.0, rays_ratio)))
    conf = max(0.0, min(1.0, conf))
    return (float(gy), float(gx), float(r)), float(conf), fov


def _estimate_cup_radius_by_ring(
    x01: torch.Tensor,
    fov_u8: np.ndarray,
    disc_geom: Tuple[float, float, float],
    r_min_scale: float = 0.18,
    r_max_scale: float = 0.70,
    num_rays: int = 72,
) -> Tuple[Optional[float], float]:
    """Estimate cup radius (px) by a weak edge-ring cue inside the disc.

    Returns (r_cup_px or None, ring_rel in [0,1]).
    Used only to calibrate cup size (postprocess/prior shaping), not as segmentation supervision.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None, 0.0

    if x01.dim() != 4 or x01.shape[0] != 1:
        return None, 0.0
    x = x01.detach().float().cpu().numpy()[0]  # C,H,W
    h, w = int(x.shape[1]), int(x.shape[2])
    if h < 32 or w < 32:
        return None, 0.0

    cy_px, cx_px, r_disc = float(disc_geom[0]), float(disc_geom[1]), float(disc_geom[2])
    r_disc = float(max(2.0, r_disc))

    if x.shape[0] >= 2:
        gray = x[1]
    else:
        gray = x.mean(axis=0)
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    fov = (np.asarray(fov_u8) > 0).astype(np.float32)
    if fov.shape != gmag.shape:
        fov = np.ones_like(gmag, dtype=np.float32)
    gmag *= fov
    gmag = gmag / (float(gmag.max()) + 1e-6)

    r_min = float(max(2.0, r_min_scale * r_disc))
    r_max = float(min(0.49 * float(min(h, w)), max(r_min + 2.0, r_max_scale * r_disc)))
    steps = int(max(24, min(140, (r_max - r_min) * 2.0)))
    rs = np.linspace(r_min, r_max, steps, dtype=np.float32)

    best_r = None
    best_score = -1e18
    for r in rs:
        vals: List[float] = []
        for t in np.linspace(0.0, 2.0 * np.pi, int(max(8, num_rays)), endpoint=False):
            xq = float(cx_px + r * np.cos(t))
            yq = float(cy_px + r * np.sin(t))
            xi = np.array([[np.clip(xq, 0, w - 1)]], dtype=np.float32)
            yi = np.array([[np.clip(yq, 0, h - 1)]], dtype=np.float32)
            v = float(
                cv2.remap(
                    gmag,
                    xi,
                    yi,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )[0, 0]
            )
            vals.append(v)
        score = float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else 0.0
        if score > best_score:
            best_score = score
            best_r = float(r)

    ring_rel = float(max(0.0, min(1.0, best_score / 0.30)))
    if best_r is None or ring_rel < 0.10:
        return None, ring_rel
    return float(best_r), ring_rel

def _estimate_disc_center_geom(x01: torch.Tensor) -> Tuple[Optional[Tuple[float, float]], float]:
    """Estimate optic disc center from image intensity (0..1).

    Returns ((cy,cx) in pixels, conf in [0,1]).
    Heuristic: disc is a bright, compact region; search in a central ROI to avoid borders.
    """
    from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, label

    if x01.dim() != 4 or x01.shape[0] != 1 or x01.shape[1] < 1:
        return None, 0.0
    x = x01.detach().float().cpu().numpy()[0]  # C,H,W
    h, w = int(x.shape[1]), int(x.shape[2])
    if h < 32 or w < 32:
        return None, 0.0

    # Prefer green channel (fundus often has strongest contrast there), fallback to mean.
    if x.shape[0] >= 2:
        img = x[1]
    else:
        img = x.mean(axis=0)

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    img_blur = gaussian_filter(img, sigma=3.0)

    y1, y2 = int(0.10 * h), int(0.90 * h)
    x1, x2 = int(0.10 * w), int(0.90 * w)
    roi = img_blur[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0

    thr = float(np.percentile(roi, 92.0))
    bw = (roi >= thr).astype(bool)
    bw = binary_opening(bw, structure=np.ones((3, 3), dtype=bool))
    bw = binary_closing(bw, structure=np.ones((5, 5), dtype=bool))

    labels, num = label(bw)
    if num <= 0:
        return None, 0.0

    best = None
    best_score = -1e18
    for k in range(1, num + 1):
        area = float((labels == k).sum())
        if area < 120:
            continue
        ys, xs = np.where(labels == k)
        if ys.size == 0:
            continue
        cy_k = float(ys.mean())
        cx_k = float(xs.mean())
        gx = float(cx_k + x1)
        gy = float(cy_k + y1)

        cxn = gx / max(1.0, float(w))
        cyn = gy / max(1.0, float(h))
        c_pref = 1.0 - 0.8 * abs(cxn - 0.5) - 0.8 * abs(cyn - 0.5)
        c_pref = max(0.0, min(1.0, c_pref))

        # We avoid contour-based circularity here to keep dependencies minimal (no OpenCV).
        score = area * (0.60 + 0.40 * c_pref)
        if score > best_score:
            best_score = score
            best = (gy, gx, area, c_pref)

    if best is None:
        return None, 0.0
    gy, gx, area, c_pref = best
    area_pref = max(0.0, min(1.0, area / (0.08 * h * w)))
    conf = float(0.25 + 0.55 * area_pref + 0.20 * c_pref)
    conf = max(0.0, min(1.0, conf))
    return (float(gy), float(gx)), conf


@dataclass
class GateStats:
    score: float
    ent: float
    center_dist: float
    disc_outside: float
    cup_outside: float
    disc_area: float
    cup_area: float
    clip_agree: float
    clip_conf: float
    bright_pen: float


def _clip_agreement_cost(disc: torch.Tensor, prior01: torch.Tensor) -> float:
    # disc: (H,W) in [0,1]; prior01: (H,W) in [0,1]
    # low is good
    d = torch.clamp(disc, 0.0, 1.0)
    p = torch.clamp(prior01, 0.0, 1.0)
    mass_in = float((d * p).mean().detach().cpu().item())
    mass_out = float((d * (1.0 - p)).mean().detach().cpu().item())
    return float(mass_out / (mass_in + 1e-6))


def _largest_cc_u8(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask (uint8 0/1)."""
    from scipy.ndimage import label

    m = (mask > 0).astype(np.uint8)
    labels, num = label(m > 0)
    if num <= 0:
        return m
    areas = [(labels == k).sum() for k in range(1, num + 1)]
    k = 1 + int(np.argmax(np.asarray(areas, dtype=np.int64)))
    return (labels == k).astype(np.uint8)


def _best_cc_by_probmass_u8(mask: np.ndarray, prob: np.ndarray, support: Optional[np.ndarray] = None) -> np.ndarray:
    """Pick a connected component by maximizing probability mass (optionally inside support)."""
    from scipy.ndimage import label

    m = (mask > 0).astype(np.uint8)
    if m.sum() <= 0:
        return m
    labels, num = label(m > 0)
    if num <= 0:
        return m

    prob = np.asarray(prob, dtype=np.float32)
    if support is not None:
        support = (np.asarray(support) > 0).astype(np.uint8)

    best_k = None
    best_score = -1e18
    for k in range(1, num + 1):
        comp = (labels == k).astype(np.uint8)
        if support is not None:
            comp = (comp & support).astype(np.uint8)
        if comp.sum() <= 0:
            continue
        score = float((prob * comp.astype(np.float32)).sum())
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        return _largest_cc_u8(m)

    out = (labels == best_k).astype(np.uint8)
    if support is not None:
        out = (out & support).astype(np.uint8)
    return out


def _postprocess_disc_cup(
    seg_prob: torch.Tensor,
    disc_thr: float = 0.5,
    cup_thr: float = 0.35,
    cup_target_ratio: float = 0.18,
    disc_target_ratio: float = 0.08,
    disc_cc_mode: str = "largest",
    cup_cc_mode: str = "largest",
    fov_u8: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Structural postprocess for OD/OC predictions.

    - threshold to binary
    - keep a stable CC for disc and cup
    - enforce cup inside disc
    Returns float tensor with {0,1}.
    """
    from scipy.ndimage import binary_closing

    if seg_prob.dim() != 4 or seg_prob.shape[1] != 2:
        return seg_prob
    disc = seg_prob[0, 0].detach().cpu().numpy()
    cup = seg_prob[0, 1].detach().cpu().numpy()
    disc_thr = float(disc_thr)
    cup_thr = float(cup_thr)
    # If disc_thr <= 0, adaptively pick a threshold so disc area is roughly stable across domains.
    # This helps datasets like Drishti_GS where a fixed threshold tends to under-cover the disc.
    if disc_thr <= 0:
        disc_target_ratio = float(disc_target_ratio)
        disc_target_ratio = max(0.02, min(0.35, disc_target_ratio))
        if fov_u8 is not None:
            sup = (np.asarray(fov_u8) > 0).astype(np.uint8)
            vals = disc[sup > 0]
        else:
            vals = disc.reshape(-1)
        if vals.size > 64:
            disc_thr = float(np.quantile(vals, 1.0 - disc_target_ratio))
        else:
            disc_thr = 0.5
    disc_thr = max(0.05, min(0.95, float(disc_thr)))
    disc_bw = (disc >= disc_thr).astype(np.uint8)
    disc_cc_mode = (disc_cc_mode or "largest").strip().lower()
    if disc_cc_mode in ("probmass", "mass", "prob"):
        disc_m = _best_cc_by_probmass_u8(disc_bw, prob=disc)
    else:
        disc_m = _largest_cc_u8(disc_bw)
    if fov_u8 is not None:
        disc_m = (disc_m & (np.asarray(fov_u8) > 0).astype(np.uint8)).astype(np.uint8)
    if cup_thr <= 0:
        # Adaptive threshold: pick a quantile inside the disc so cup area ratio is stable across domains.
        cup_target_ratio = float(cup_target_ratio)
        cup_target_ratio = max(0.05, min(0.45, cup_target_ratio))
        vals = cup[disc_m > 0]
        if vals.size > 16:
            q = float(np.quantile(vals, 1.0 - cup_target_ratio))
            cup_thr = q
        else:
            cup_thr = 0.35
    cup_thr = max(0.05, min(0.95, float(cup_thr)))
    cup_bw = (cup >= cup_thr).astype(np.uint8)
    cup_cc_mode = (cup_cc_mode or "largest").strip().lower()
    if cup_cc_mode in ("probmass_in_disc", "probmass-disc", "probmass_disc", "probmassindisc"):
        cup_m = _best_cc_by_probmass_u8(cup_bw, prob=cup, support=disc_m)
    elif cup_cc_mode in ("probmass", "mass", "prob"):
        cup_m = _best_cc_by_probmass_u8(cup_bw, prob=cup)
    else:
        cup_m = _largest_cc_u8(cup_bw)

    disc_m = binary_closing(disc_m.astype(bool), structure=np.ones((5, 5), dtype=bool)).astype(np.uint8)
    cup_m = binary_closing(cup_m.astype(bool), structure=np.ones((3, 3), dtype=bool)).astype(np.uint8)
    cup_m = (cup_m & disc_m).astype(np.uint8)

    out = np.stack([disc_m, cup_m], axis=0).astype(np.float32)
    return torch.from_numpy(out).unsqueeze(0).to(device=seg_prob.device, dtype=seg_prob.dtype)


def _gate_score(
    seg_prob: torch.Tensor,
    prior: torch.Tensor,
    x01: Optional[torch.Tensor] = None,
    clip_prior: Optional[object] = None,
    clip_gate_conf_min: float = 0.15,
    clip_gate_weight: float = 0.15,
    prior_roi_thr: float = 0.20,
    prior_barrier_power: float = 2.0,
) -> GateStats:
    # seg_prob: (1,2,H,W) after sigmoid
    # prior: (H,W) in [0,1]
    disc = seg_prob[0, 0]
    cup = seg_prob[0, 1]

    # entropy (prefer confident)
    ent = (_soft_entropy_bernoulli(disc) + _soft_entropy_bernoulli(cup)).mean().item()

    # localization preference: disc centroid close to the prior's centroid (domain-robust)
    cy, cx = _soft_centroid_2d(disc)
    p = prior.to(device=disc.device, dtype=disc.dtype)
    py, px = _soft_centroid_2d(p)
    center_dist = float(torch.sqrt((cy - py) ** 2 + (cx - px) ** 2).detach().cpu().item())

    # cup inside disc penalty
    cup_outside = float(torch.relu(cup - disc).mean().detach().cpu().item())
    disc_outside = float(
        _mass_outside_prior_ratio(
            disc,
            prior,
            roi_thr=float(prior_roi_thr),
            power=float(prior_barrier_power),
        )
        .detach()
        .cpu()
        .item()
    )

    # soft area ratios (normalized)
    disc_area = float(disc.mean().detach().cpu().item())
    cup_area = float(cup.mean().detach().cpu().item())

    # area regularization (very weak; avoids full-mask collapse)
    area_pen = float(
        (
            torch.relu(disc.mean() - 0.35)
            + torch.relu(0.01 - disc.mean())
            + 0.5 * torch.relu(cup.mean() - disc.mean())
        )
        .detach()
        .cpu()
        .item()
    )

    clip_agree = 0.0
    clip_conf = 0.0
    if clip_prior is not None:
        clip_conf = float(clip_prior.conf_disc)
        if clip_conf >= float(clip_gate_conf_min):
            p = torch.from_numpy(clip_prior.disc_prior).to(device=disc.device, dtype=disc.dtype)
            clip_agree = _clip_agreement_cost(disc, p)

    bright_pen = 0.0
    if x01 is not None and x01.dim() == 4 and x01.shape[0] == 1:
        # Encourage disc to align with bright region (optic disc is typically brighter than background).
        # Use green channel if available, fallback to mean.
        if x01.shape[1] >= 2:
            img = x01[0, 1].to(device=disc.device, dtype=disc.dtype)
        else:
            img = x01[0].mean(dim=0).to(device=disc.device, dtype=disc.dtype)
        img = torch.clamp(img, 0.0, 1.0)
        in_mean = (img * disc).sum() / (disc.sum() + 1e-6)
        out_mean = (img * (1.0 - disc)).sum() / ((1.0 - disc).sum() + 1e-6)
        diff = in_mean - out_mean
        # penalty if the disc is not noticeably brighter than background
        bright_pen = float(torch.relu(0.05 - diff).detach().cpu().item())

    # Lower is better. Gate blends: entropy + localization + invalid-structure + weak appearance prior.
    # Use CLIP as a weak *extra* veto only when it's confident.
    score = (
        1.0 * ent
        + 1.2 * center_dist
        + 0.4 * disc_outside
        + 2.0 * cup_outside
        + 1.0 * area_pen
        + 0.6 * bright_pen
        + float(clip_gate_weight) * clip_conf * clip_agree
    )
    return GateStats(
        score=float(score),
        ent=float(ent),
        center_dist=float(center_dist),
        disc_outside=float(disc_outside),
        cup_outside=float(cup_outside),
        disc_area=float(disc_area),
        cup_area=float(cup_area),
        clip_agree=float(clip_agree),
        clip_conf=float(clip_conf),
        bright_pen=float(bright_pen),
    )


def _refine_prior_from_pred_disc(seg_prob: torch.Tensor, x01: torch.Tensor, sigma: float) -> Tuple[Optional[torch.Tensor], float]:
    """Refine the Gaussian prior center from the model's *pre-update* disc prediction.

    This stabilizes localization when the dataset is not center-cropped (e.g., REFUGE_Valid),
    while remaining fully unsupervised. Returns (prior, conf) where conf in [0,1].
    """
    if seg_prob.dim() != 4 or seg_prob.shape[0] != 1 or seg_prob.shape[1] < 1:
        return None, 0.0
    disc = torch.clamp(seg_prob[0, 0], 0.0, 1.0)
    h, w = int(disc.shape[-2]), int(disc.shape[-1])
    if h < 32 or w < 32:
        return None, 0.0

    disc_area = float(disc.mean().detach().cpu().item())
    if disc_area < 0.006 or disc_area > 0.40:
        return None, 0.0

    # Disc should be a comparatively bright region.
    bright_pen = 0.0
    if x01 is not None and x01.dim() == 4 and x01.shape[0] == 1:
        if x01.shape[1] >= 2:
            img = x01[0, 1].to(device=disc.device, dtype=disc.dtype)
        else:
            img = x01[0].mean(dim=0).to(device=disc.device, dtype=disc.dtype)
        img = torch.clamp(img, 0.0, 1.0)
        in_mean = (img * disc).sum() / (disc.sum() + 1e-6)
        out_mean = (img * (1.0 - disc)).sum() / ((1.0 - disc).sum() + 1e-6)
        diff = float((in_mean - out_mean).detach().cpu().item())
        bright_pen = float(max(0.0, 0.05 - diff))

    # Prefer confident disc.
    ent = float(_soft_entropy_bernoulli(disc).mean().detach().cpu().item())

    if ent > 0.55 or bright_pen > 0.20:
        return None, 0.0

    cy_n, cx_n = _soft_centroid_2d(disc)
    cy_px = float(cy_n.detach().cpu().item() * max(1.0, float(h - 1)))
    cx_px = float(cx_n.detach().cpu().item() * max(1.0, float(w - 1)))
    prior = _gaussian_prior_at(h, w, cy_px=cy_px, cx_px=cx_px, sigma=float(sigma)).to(device=disc.device, dtype=torch.float32)

    # confidence combines low entropy and brightness sanity.
    conf = float(max(0.0, min(1.0, 1.0 - (ent / 0.55))) * max(0.0, min(1.0, 1.0 - (bright_pen / 0.20))))
    conf = max(0.0, min(1.0, conf))
    return prior, conf


def _estimate_pred_disc_center_radius(
    seg_prob: torch.Tensor,
    fov_u8: Optional[np.ndarray] = None,
) -> Tuple[Optional[Tuple[float, float, float]], float]:
    """Estimate OD center/radius from pre-update source prediction."""
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    from scipy.ndimage import label

    if seg_prob.dim() != 4 or seg_prob.shape[0] != 1 or seg_prob.shape[1] < 1:
        return None, 0.0
    p = torch.clamp(seg_prob[0, 0], 0.0, 1.0).detach().cpu().numpy().astype(np.float32)
    h, w = int(p.shape[0]), int(p.shape[1])
    if h < 32 or w < 32:
        return None, 0.0
    if fov_u8 is None or np.asarray(fov_u8).shape != p.shape:
        fov = np.ones_like(p, dtype=np.uint8)
    else:
        fov = (np.asarray(fov_u8) > 0).astype(np.uint8)
    vals = p[fov > 0]
    if vals.size < 64:
        return None, 0.0

    candidates: List[Tuple[float, np.ndarray, float, float]] = []
    thresholds = [0.70, 0.60, 0.50, 0.40, max(0.20, float(np.percentile(vals, 92.0)))]
    for thr in thresholds:
        bw = ((p >= float(thr)) & (fov > 0)).astype(np.uint8)
        if cv2 is not None:
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            for k in range(1, num):
                area = float(stats[k, cv2.CC_STAT_AREA])
                if area < 0.0015 * h * w or area > 0.35 * h * w:
                    continue
                comp = (labels_cc == k)
                mean_prob = float(p[comp].mean()) if area > 0 else 0.0
                mass = float((p * comp.astype(np.float32)).sum())
                size_pref = 1.0 - abs(area / (0.08 * h * w) - 1.0)
                size_pref = float(max(0.0, min(1.0, size_pref)))
                score = mass * (0.55 + 0.45 * size_pref) * (0.50 + 0.50 * mean_prob)
                candidates.append((score, comp.astype(np.uint8), area, mean_prob))
        else:
            labels_cc, num = label(bw > 0)
            for k in range(1, num + 1):
                comp = (labels_cc == k)
                area = float(comp.sum())
                if area < 0.0015 * h * w or area > 0.35 * h * w:
                    continue
                mean_prob = float(p[comp].mean()) if area > 0 else 0.0
                mass = float((p * comp.astype(np.float32)).sum())
                size_pref = 1.0 - abs(area / (0.08 * h * w) - 1.0)
                size_pref = float(max(0.0, min(1.0, size_pref)))
                score = mass * (0.55 + 0.45 * size_pref) * (0.50 + 0.50 * mean_prob)
                candidates.append((score, comp.astype(np.uint8), area, mean_prob))

    if not candidates:
        return None, 0.0
    _, comp, area, mean_prob = max(candidates, key=lambda x: x[0])
    weights = p * comp.astype(np.float32)
    if float(weights.sum()) <= 1e-6:
        ys, xs = np.where(comp > 0)
        if ys.size <= 0:
            return None, 0.0
        cy_px = float(ys.mean())
        cx_px = float(xs.mean())
    else:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        s = float(weights.sum())
        cy_px = float((weights * yy).sum() / s)
        cx_px = float((weights * xx).sum() / s)
    r_px = float(np.sqrt(max(float(area), 1.0) / np.pi))
    area_pref = float(max(0.0, min(1.0, area / (0.10 * h * w))))
    conf = float(max(0.0, min(1.0, 0.20 + 0.45 * area_pref + 0.35 * float(mean_prob))))
    return (cy_px, cx_px, r_px), conf


def _adaptive_select_od_prior(
    *,
    geom_prior: Optional[torch.Tensor],
    geom_info: Optional[Tuple[float, float, float]],
    geom_conf: float,
    geom_ray_valid: float,
    pred_prior: Optional[torch.Tensor],
    pred_info: Optional[Tuple[float, float, float]],
    pred_conf: float,
    fallback: torch.Tensor,
    agree_thr: float = 1.25,
    pred_conf_min: float = 0.50,
    geom_good_conf: float = 0.55,
    pred_adv_margin: float = 0.15,
) -> Tuple[torch.Tensor, float, str]:
    """Select a reliable coarse OD ROI prior from geometry and pre-update prediction."""
    if pred_prior is not None and pred_info is not None and (geom_prior is None or geom_info is None):
        return _normalize01_hw(pred_prior.detach().to(dtype=torch.float32)), float(pred_conf), "pred"
    if geom_prior is not None and geom_info is not None and (pred_prior is None or pred_info is None):
        return _normalize01_hw(geom_prior.detach().to(dtype=torch.float32)), float(geom_conf), "geom"
    if geom_prior is None or geom_info is None or pred_prior is None or pred_info is None:
        return _normalize01_hw(fallback.detach().to(dtype=torch.float32)), 0.0, "default"

    gy, gx, gr = geom_info
    py, px, pr = pred_info
    dist = float(np.sqrt((float(gy) - float(py)) ** 2 + (float(gx) - float(px)) ** 2))
    ref_r = max(1.0, 0.5 * (float(gr) + float(pr)))
    dist_r = dist / ref_r
    radius_ratio = float(pr) / max(1.0, float(gr))

    gp = _normalize01_hw(geom_prior.detach().to(dtype=torch.float32))
    pp = _normalize01_hw(pred_prior.detach().to(device=gp.device, dtype=torch.float32))
    if dist_r <= float(agree_thr):
        wg = max(0.05, float(geom_conf))
        wp = max(0.05, float(pred_conf))
        fused = _normalize01_hw((wg * gp + wp * pp) / max(1e-6, wg + wp))
        conf = max(float(geom_conf), float(pred_conf)) * max(0.55, 1.0 - 0.20 * dist_r)
        return fused, float(max(0.0, min(1.0, conf))), "adaptive_fused"

    pred_reasonable = float(pred_conf) >= float(pred_conf_min) and 0.35 <= radius_ratio <= 2.50
    geom_reliable = float(geom_conf) >= float(geom_good_conf) and float(geom_ray_valid) >= 0.35
    geom_unreliable = float(geom_conf) < 0.45 or float(geom_ray_valid) < 0.25
    if pred_reasonable and (geom_unreliable or float(pred_conf) >= float(geom_conf) + float(pred_adv_margin)):
        return pp, float(pred_conf), "adaptive_pred"
    if pred_reasonable and (not geom_reliable) and float(pred_conf) >= float(geom_conf):
        return pp, float(pred_conf), "adaptive_pred"
    return gp, float(geom_conf), "adaptive_geom"


class RATTA_OPTIC:
    def __init__(self, config: argparse.Namespace):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, "RATTA")
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + ".log")
        sys.stdout = Logger(log_path, sys.stdout)

        # Protocol: default matches vptta.py (Table 1): use both train+test csvs per domain.
        # For quick experimentation, allow evaluating on test-only csvs.
        target_test_csv = _build_target_csvs_mode(config.Target_Dataset, getattr(config, "eval_split", "train_test"))
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list, config.image_size, img_normalize=True)
        self.target_test_loader = DataLoader(
            dataset=target_test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_wo_transform,
            num_workers=config.num_workers,
        )
        self.max_cases = int(getattr(config, "max_cases", 0))
        self.save_visuals = bool(getattr(config, "save_visuals", False))
        self.visual_dir = str(getattr(config, "visual_dir", "")) if getattr(config, "visual_dir", "") else os.path.join(config.path_save_log, "visuals")
        self.save_case_outputs = bool(getattr(config, "save_case_outputs", False))
        self.case_output_dir = (
            str(getattr(config, "case_output_dir", ""))
            if getattr(config, "case_output_dir", "")
            else os.path.join(config.path_save_log, "case_outputs")
        )
        if self.save_case_outputs:
            os.makedirs(self.case_output_dir, exist_ok=True)

        self.image_size = int(config.image_size)
        self.device = config.device

        # Model + Prompt
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))
        self.backbone = config.backbone
        self.out_ch = config.out_ch

        self.warm_n = config.warm_n
        self.prompt_alpha = config.prompt_alpha
        self.iters = config.iters
        self.rounds = int(getattr(config, "rounds", 1))

        # Gate/rollback settings
        self.enable_gate = bool(config.enable_gate)
        self.disable_restore = bool(getattr(config, "disable_restore", False))
        self.gate_margin = float(config.gate_margin)
        self.gate_min_improve = float(config.gate_min_improve)
        self.prior_sigma = float(config.prior_sigma)
        self.use_geom_prior = bool(getattr(config, "use_geom_prior", False))
        self.geom_min_conf = float(getattr(config, "geom_min_conf", 0.15))
        self.use_adaptive_od_prior = bool(getattr(config, "use_adaptive_od_prior", False))
        self.od_roi_scale = float(getattr(config, "od_roi_scale", 2.50))
        self.od_prior_min_sigma = float(getattr(config, "od_prior_min_sigma", 0.035))
        self.od_prior_max_sigma = float(getattr(config, "od_prior_max_sigma", 0.36))
        self.adaptive_agree_thr = float(getattr(config, "adaptive_agree_thr", 1.25))
        self.adaptive_pred_conf_min = float(getattr(config, "adaptive_pred_conf_min", 0.50))
        self.adaptive_geom_good_conf = float(getattr(config, "adaptive_geom_good_conf", 0.55))
        self.adaptive_pred_adv_margin = float(getattr(config, "adaptive_pred_adv_margin", 0.15))
        self.use_hybrid_prior = bool(getattr(config, "use_hybrid_prior", False))
        self.hybrid_w_default = float(getattr(config, "hybrid_w_default", 0.12))
        self.hybrid_w_geom = float(getattr(config, "hybrid_w_geom", 0.35))
        self.hybrid_w_pred = float(getattr(config, "hybrid_w_pred", 0.35))
        self.hybrid_w_clip = float(getattr(config, "hybrid_w_clip", 0.30))
        self.refine_prior_from_pred = bool(getattr(config, "refine_prior_from_pred", False))
        self.refine_prior_min_conf = float(getattr(config, "refine_prior_min_conf", 0.30))
        self.roi_w_min = float(getattr(config, "roi_w_min", 0.20))
        self.roi_w_gamma = float(getattr(config, "roi_w_gamma", 1.0))
        self.prior_use_roi_weight = bool(int(getattr(config, "prior_use_roi_weight", 1)))
        self.geom_sigma_scale = float(getattr(config, "geom_sigma_scale", 0.55))

        # Loss weights
        self.w_bn = float(config.w_bn)
        self.w_prior = float(config.w_prior)
        self.prior_loss_mode = str(getattr(config, "prior_loss_mode", "bce")).strip().lower()
        if self.prior_loss_mode not in ("bce", "barrier"):
            self.prior_loss_mode = "bce"
        self.prior_roi_thr = float(getattr(config, "prior_roi_thr", 0.20))
        self.prior_barrier_power = float(getattr(config, "prior_barrier_power", 2.0))
        self.prior_conf_gate = bool(getattr(config, "prior_conf_gate", False))
        self.prior_conf_min = float(getattr(config, "prior_conf_min", 0.45))
        self.prior_conf_power = float(getattr(config, "prior_conf_power", 1.0))
        self.w_cup_prior = float(getattr(config, "w_cup_prior", 0.0))
        self.w_inside = float(config.w_inside)
        self.w_entropy = float(config.w_entropy)
        self.w_clip = float(getattr(config, "w_clip", 0.0))
        self.w_cup_ratio = float(getattr(config, "w_cup_ratio", 0.0))
        self.cup_ratio_target = float(getattr(config, "cup_ratio_target", 0.18))
        self.cup_prior_sigma_scale = float(getattr(config, "cup_prior_sigma_scale", 0.60))
        self.postprocess_cc = bool(getattr(config, "postprocess_cc", False))
        self.pp_disc_thr = float(getattr(config, "pp_disc_thr", 0.50))
        self.pp_disc_target_ratio = float(getattr(config, "pp_disc_target_ratio", 0.08))
        self.pp_cup_thr = float(getattr(config, "pp_cup_thr", 0.35))
        self.pp_cup_target_ratio = float(getattr(config, "pp_cup_target_ratio", 0.18))
        self.pp_disc_cc_mode = str(getattr(config, "pp_disc_cc_mode", "largest"))
        self.pp_cup_cc_mode = str(getattr(config, "pp_cup_cc_mode", "largest"))

        # EMA prompt (teacher)
        self.use_prompt_ema = bool(config.use_prompt_ema)
        self.ema_m = float(config.ema_m)

        # BioMedCLIP weak localization
        self.use_clip_prior = bool(getattr(config, "use_clip_prior", False))
        self.clip_root = str(getattr(config, "clip_root", "")) if getattr(config, "clip_root", "") else ""
        self.clip_cache = str(getattr(config, "clip_cache", "")) if getattr(config, "clip_cache", "") else ""
        self.clip_softmax_temp = float(getattr(config, "clip_softmax_temp", 0.08))
        self.clip_patch_sigma = float(getattr(config, "clip_patch_sigma", 4.0))
        self.clip_prior_sigma = float(getattr(config, "clip_prior_sigma", 0.18))
        self.clip_gate_only = bool(getattr(config, "clip_gate_only", False))
        self.clip_gate_conf_min = float(getattr(config, "clip_gate_conf_min", 0.15))
        self.clip_gate_weight = float(getattr(config, "clip_gate_weight", 0.10))
        self.clip_prior_mode = str(getattr(config, "clip_prior_mode", "replace"))
        self.clip_conf_min = float(getattr(config, "clip_conf_min", 0.15))
        self.clip_blend_floor = float(getattr(config, "clip_blend_floor", 0.15))
        self.clip_blend_power = float(getattr(config, "clip_blend_power", 1.0))
        self.clip_weight_by_conf = bool(getattr(config, "clip_weight_by_conf", False))
        self.clip = None

        # Memory bank
        self.neighbor = int(config.neighbor)
        self.memory_size = int(config.memory_size)
        self.memory_bank = None

        # Optimizer
        self.optim = config.optimizer
        self.lr = float(config.lr)
        self.model_lr = float(getattr(config, "model_lr", 1e-4))
        self.weight_decay = float(config.weight_decay)
        self.momentum = float(config.momentum)
        self.betas = (float(config.beta1), float(config.beta2))

        # Adaptation scope (beyond prompt)
        self.adapt_bn_affine = bool(getattr(config, "adapt_bn_affine", False))
        self.adapt_head = bool(getattr(config, "adapt_head", False))
        self.use_vida = bool(getattr(config, "use_vida", False))
        self.vida_rank = int(getattr(config, "vida_rank", 4))
        self.vida_init_scale = float(getattr(config, "vida_init_scale", 0.0))

        # Model EMA (teacher)
        self.use_model_ema = bool(getattr(config, "use_model_ema", False))
        self.model_ema_m = float(getattr(config, "model_ema_m", 0.99))
        self.w_consistency = float(getattr(config, "w_consistency", 0.0))
        self.ema_model = None
        self.vida_adapter: Optional[ShiftDecomposedAdapter] = None

        self._build()

        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self._print_prompt()
        print("***" * 20)

    def _build(self) -> None:
        self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        self.model = ResUnet(
            resnet=self.backbone,
            num_classes=self.out_ch,
            pretrained=False,
            newBN=AdaBN,
            warm_n=self.warm_n,
        ).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, "last-Res_Unet.pth"), map_location="cpu")
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Optionally adapt a tiny subset of model params for stronger CTTA
        model_params: List[torch.nn.Parameter] = []
        if self.adapt_bn_affine:
            for m in self.model.modules():
                if isinstance(m, AdaBN):
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
                    model_params.append(m.weight)
                    model_params.append(m.bias)
        if self.adapt_head:
            for p in self.model.seg_head.parameters():
                p.requires_grad_(True)
                model_params.append(p)

        vida_params: List[torch.nn.Parameter] = []
        if self.use_vida:
            self.vida_adapter = ShiftDecomposedAdapter(in_ch=3, rank=self.vida_rank, init_scale=self.vida_init_scale).to(self.device)
            for p in self.vida_adapter.parameters():
                p.requires_grad_(True)
                vida_params.append(p)

        if self.use_model_ema and (self.adapt_bn_affine or self.adapt_head):
            import copy

            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        if self.use_clip_prior and self.clip_root:
            try:
                from biomedclip_prior import BioMedCLIPPrior  # optional dependency (open_clip)

                cache_dir = self.clip_cache if self.clip_cache else None
                self.clip = BioMedCLIPPrior(
                    root=self.clip_root,
                    device=torch.device(self.device),
                    softmax_temp=self.clip_softmax_temp,
                    patch_sigma=self.clip_patch_sigma,
                    prior_sigma_disc=self.clip_prior_sigma,
                    cache_dir=cache_dir,
                )
            except Exception as e:
                self.clip = None
                print(f"[WARN] BioMedCLIP prior disabled (import/init failed): {e}")

        param_groups = [{"params": list(self.prompt.parameters()), "lr": self.lr}]
        if model_params:
            param_groups.append({"params": model_params, "lr": self.model_lr})
        if vida_params:
            param_groups.append({"params": vida_params, "lr": self.model_lr})

        if self.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                param_groups,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                param_groups,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )

        self.memory_bank = Memory(size=self.memory_size, dimension=self.prompt.data_prompt.numel())

        self.ema_prompt = None
        if self.use_prompt_ema:
            self.ema_prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
            self.ema_prompt.load_state_dict(self.prompt.state_dict(), strict=True)
            for p in self.ema_prompt.parameters():
                p.requires_grad_(False)

    def _model_trainable_state(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                state[n] = p.detach().clone()
        return state

    def _model_restore_trainable(self, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in state:
                    p.copy_(state[n])

    def _ema_model_update(self) -> None:
        if self.ema_model is None:
            return
        m = float(self.model_ema_m)
        with torch.no_grad():
            for (n_s, p_s), (n_t, p_t) in zip(self.model.named_parameters(), self.ema_model.named_parameters()):
                if p_s.requires_grad:
                    p_t.mul_(m).add_(p_s.detach(), alpha=(1.0 - m))

    def _print_prompt(self) -> None:
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    @torch.no_grad()
    def _ema_update(self) -> None:
        if self.ema_prompt is None:
            return
        m = float(self.ema_m)
        for (n_s, p_s), (n_t, p_t) in zip(self.prompt.named_parameters(), self.ema_prompt.named_parameters()):
            assert n_s == n_t
            p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))

    def _prompt_snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.prompt.state_dict().items()}

    def _prompt_restore(self, state: Dict[str, torch.Tensor]) -> None:
        self.prompt.load_state_dict(state, strict=True)

    def _vida_snapshot(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.vida_adapter is None:
            return None
        return {k: v.detach().clone() for k, v in self.vida_adapter.state_dict().items()}

    def _vida_restore(self, state: Optional[Dict[str, torch.Tensor]]) -> None:
        if self.vida_adapter is None or state is None:
            return
        self.vida_adapter.load_state_dict(state, strict=True)

    def _infer_with_prompt(self, x: torch.Tensor, use_ema: bool = False) -> Tuple[torch.Tensor, np.ndarray]:
        self.model.eval()
        if self.vida_adapter is not None:
            self.vida_adapter.eval()
        # Do not advance AdaBN "new sample" counter during pure inference.
        self.model.change_BN_status(new_sample=False)
        if use_ema and self.ema_prompt is not None:
            self.ema_prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq = self.ema_prompt(x)
                if self.vida_adapter is not None:
                    prompt_x = self.vida_adapter(prompt_x)
                pred_logit, _, _ = self.model(prompt_x)
        else:
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq = self.prompt(x)
                if self.vida_adapter is not None:
                    prompt_x = self.vida_adapter(prompt_x)
                pred_logit, _, _ = self.model(prompt_x)
        return pred_logit, low_freq.detach().cpu().numpy()

    def _derive_cup_prior(self, disc_prior01: torch.Tensor) -> torch.Tensor:
        """Derive a weak cup localization prior from a disc localization prior.

        Use the soft centroid of the disc prior and a smaller sigma to form a cup prior.
        """
        p = torch.clamp(disc_prior01.detach().to(device="cpu", dtype=torch.float32), 0.0, 1.0)
        h, w = int(p.shape[0]), int(p.shape[1])
        mass = float(p.sum().item())
        if mass <= 1e-6:
            cy_px = 0.5 * float(h - 1)
            cx_px = 0.5 * float(w - 1)
        else:
            yy = torch.arange(h, dtype=torch.float32).view(h, 1)
            xx = torch.arange(w, dtype=torch.float32).view(1, w)
            cy_px = float((p * yy).sum().item() / mass)
            cx_px = float((p * xx).sum().item() / mass)
        # Keep cup prior reasonably compact but not too peaky (cup is smaller yet not a point).
        s = float(max(0.25, min(0.95, self.cup_prior_sigma_scale)))
        sigma = float(max(0.06, float(self.prior_sigma) * s))
        return _gaussian_prior_at(h, w, cy_px=cy_px, cx_px=cx_px, sigma=sigma)

    def _train_prompt_one(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
        cup_ratio_target: Optional[float] = None,
        clip_conf: float = 1.0,
        prior_conf: float = 1.0,
    ) -> float:
        # Train prompt for iters steps (default 1, like VPTTA)
        self.model.eval()
        self.prompt.train()
        if self.vida_adapter is not None:
            self.vida_adapter.train()
        self.model.change_BN_status(new_sample=True)

        loss_last = 0.0
        for _ in range(self.iters):
            prompt_x, _ = self.prompt(x)
            if self.vida_adapter is not None:
                prompt_x = self.vida_adapter(prompt_x)
            pred_logit, _, _ = self.model(prompt_x)

            times, bn_loss = 0, 0.0
            for _, m in self.model.named_modules():
                if isinstance(m, AdaBN):
                    bn_loss = bn_loss + m.bn_loss
                    times += 1
            if times > 0:
                bn_loss = bn_loss / times

            seg_prob = torch.sigmoid(pred_logit)
            disc = seg_prob[:, 0:1]
            cup = seg_prob[:, 1:2]
            prior_t = prior.to(device=seg_prob.device, dtype=seg_prob.dtype).view(1, 1, prior.shape[0], prior.shape[1])
            w_roi = None
            if self.prior_use_roi_weight:
                w_roi = _roi_weight_from_prior(prior.to(device=seg_prob.device, dtype=torch.float32), w_min=self.roi_w_min, gamma=self.roi_w_gamma).view(
                    1, 1, prior.shape[0], prior.shape[1]
                )

            # Weak prior. BCE is kept for backward-compatible experiments; the
            # barrier mode is the safer APSA-OD/OC prior because it only penalizes
            # mass far outside the anatomical ROI and does not impose a Gaussian shape.
            if self.prior_loss_mode == "barrier":
                prior_loss = _mass_outside_prior_ratio(
                    disc,
                    prior,
                    roi_thr=self.prior_roi_thr,
                    power=self.prior_barrier_power,
                )
                prior_loss = prior_loss + 0.20 * torch.relu(torch.tensor(0.006, device=disc.device, dtype=disc.dtype) - disc.mean())
                prior_loss = prior_loss + 0.20 * torch.relu(disc.mean() - torch.tensor(0.35, device=disc.device, dtype=disc.dtype))
            elif w_roi is None:
                prior_loss = F.binary_cross_entropy(disc, prior_t)
            else:
                prior_loss = _weighted_mean(F.binary_cross_entropy(disc, prior_t, reduction="none"), w_roi)
            cup_prior_loss = 0.0
            if self.w_cup_prior > 0.0:
                cup_prior = self._derive_cup_prior(prior).to(device=seg_prob.device, dtype=seg_prob.dtype)
                cup_prior_t = cup_prior.view(1, 1, prior.shape[0], prior.shape[1])
                w_roi_cup = None
                if self.prior_use_roi_weight:
                    w_roi_cup = _roi_weight_from_prior(
                        cup_prior.to(device=seg_prob.device, dtype=torch.float32),
                        w_min=max(0.10, 0.80 * self.roi_w_min),
                        gamma=max(0.6, 0.85 * self.roi_w_gamma),
                    ).view(1, 1, prior.shape[0], prior.shape[1])
                if self.prior_loss_mode == "barrier":
                    cup_prior_loss = _mass_outside_prior_ratio(
                        cup,
                        cup_prior,
                        roi_thr=max(0.05, 1.15 * self.prior_roi_thr),
                        power=self.prior_barrier_power,
                    )
                elif w_roi is None:
                    cup_prior_loss = F.binary_cross_entropy(cup, cup_prior_t)
                else:
                    w_use = w_roi if w_roi_cup is None else torch.clamp(w_roi * w_roi_cup, 0.0, 1.0)
                    cup_prior_loss = _weighted_mean(F.binary_cross_entropy(cup, cup_prior_t, reduction="none"), w_use)
            inside_map = torch.relu(cup - disc)
            if w_roi is None:
                inside_loss = inside_map.mean()
            else:
                inside_loss = _weighted_mean(inside_map, w_roi)

            # Soft cup-size regularizer: encourages a reasonable cup/disc area ratio without any GT.
            cup_ratio_loss = 0.0
            if self.w_cup_ratio > 0.0:
                # ratio is relative to disc support (in [0,1])
                if w_roi is None:
                    disc_m = torch.clamp(disc, 0.0, 1.0)
                    cup_m = torch.clamp(cup, 0.0, 1.0)
                    disc_area = disc_m.mean()
                    cup_area = cup_m.mean()
                else:
                    disc_m = torch.clamp(disc, 0.0, 1.0)
                    cup_m = torch.clamp(cup, 0.0, 1.0)
                    disc_area = _weighted_mean(disc_m, w_roi)
                    cup_area = _weighted_mean(cup_m, w_roi)
                ratio = cup_area / (disc_area + 1e-6)
                tgt = float(cup_ratio_target) if cup_ratio_target is not None else float(self.cup_ratio_target)
                tgt = float(max(0.05, min(0.45, tgt)))
                cup_ratio_loss = (ratio - tgt) ** 2
            ent_map = _soft_entropy_bernoulli(disc) + _soft_entropy_bernoulli(cup)
            if w_roi is None:
                entropy_loss = ent_map.mean()
            else:
                entropy_loss = _weighted_mean(ent_map, w_roi)

            consistency = 0.0
            if self.ema_model is not None and self.w_consistency > 0:
                with torch.no_grad():
                    # teacher uses EMA model + current prompt input (prompt is the main adaptation handle)
                    t_logit, _, _ = self.ema_model(prompt_x)
                    t_prob = torch.sigmoid(t_logit)
                if w_roi is None:
                    consistency = F.mse_loss(seg_prob, t_prob)
                else:
                    consistency = _weighted_mean((seg_prob - t_prob) ** 2, w_roi)

            clip_loss = 0.0
            if self.w_clip > 0.0 and self.clip is not None and not self.clip_gate_only:
                # Use the same prior tensor (already CLIP-derived) as a very small additional stabilization term.
                if w_roi is None:
                    clip_loss = F.binary_cross_entropy(disc, prior_t)
                else:
                    clip_loss = _weighted_mean(F.binary_cross_entropy(disc, prior_t, reduction="none"), w_roi)
            clip_w = float(self.w_clip)
            if self.clip_weight_by_conf:
                clip_w *= float(max(0.0, min(1.0, clip_conf)))
            prior_scale = 1.0
            if self.prior_conf_gate:
                prior_scale = _prior_confidence_scale(
                    prior_conf,
                    min_conf=self.prior_conf_min,
                    power=self.prior_conf_power,
                )
            loss = (
                self.w_bn * bn_loss
                + (self.w_prior * prior_scale) * prior_loss
                + (self.w_cup_prior * prior_scale) * cup_prior_loss
                + self.w_inside * inside_loss
                + self.w_entropy * entropy_loss
                + clip_w * clip_loss
                + self.w_consistency * consistency
                + self.w_cup_ratio * cup_ratio_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.change_BN_status(new_sample=False)
            loss_last = float(loss.detach().cpu().item())

        return loss_last

    def run(self) -> None:
        rounds = max(1, int(self.rounds))
        round_mean_dice: List[float] = []
        for r in range(1, rounds + 1):
            _, md = self._run_loader(self.target_test_loader, print_prefix=f"Round {r}")
            round_mean_dice.append(float(md))
        if rounds > 1:
            overall = float(sum(round_mean_dice) / max(1, len(round_mean_dice)))
            degra = float(round_mean_dice[0] - overall)
            print(f"Overall Mean Dice (avg over rounds): {overall}")
            print(f"Performance Degradation (Round1 - Overall): {degra}")

    def _run_loader(self, loader: DataLoader, print_prefix: str = "Test") -> Tuple[Dict[str, float], float]:
        metric_dict = ["Disc_Dice", "Disc_ASD", "Cup_Dice", "Cup_ASD"]
        metrics_test = [[], [], [], []]
        case_rows: List[Dict[str, object]] = []

        prior_default = _central_gaussian_prior(self.image_size, self.image_size, sigma=self.prior_sigma).to(self.device)

        for sample_idx, data in enumerate(loader):
            if self.max_cases > 0 and sample_idx >= self.max_cases:
                break
            x, y, name = data["data"], data["mask"], data["name"]
            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)

            clip_prior: Optional[object] = None
            prior = prior_default
            prior_method = "default"
            adaptive_source_conf: Optional[float] = None
            geom_prior_for_adaptive: Optional[torch.Tensor] = None
            geom_info_for_adaptive: Optional[Tuple[float, float, float]] = None
            geom_conf_for_adaptive = 0.0
            geom_ray_valid_for_adaptive = 1.0
            pred_conf_for_adaptive = 0.0
            prior_candidates: List[Tuple[torch.Tensor, float, float]] = []
            prior_conf_votes: List[Tuple[float, float]] = []
            if self.use_hybrid_prior:
                prior_candidates.append((prior_default.detach().cpu(), 1.0, self.hybrid_w_default))
            clip_conf_for_loss = 1.0
            try:
                fov_u8 = _estimate_fov_mask_from_x01(x.detach())
            except Exception:
                fov_u8 = None
            cup_ratio_override: Optional[float] = None
            if self.use_geom_prior or self.use_adaptive_od_prior:
                geom, conf, fov_u8_geom = _estimate_disc_geom_center_radius(x.detach())
                if fov_u8 is None:
                    fov_u8 = fov_u8_geom
                if geom is not None and float(conf) >= self.geom_min_conf:
                    cy_px, cx_px, r_px = geom
                    # sigma in normalized coords; tie it to radius so ROI coverage is more stable cross-domain
                    if self.use_adaptive_od_prior:
                        prior = _coarse_od_roi_prior_at(
                            self.image_size,
                            self.image_size,
                            cy_px,
                            cx_px,
                            r_px,
                            roi_scale=self.od_roi_scale,
                            roi_thr=self.prior_roi_thr,
                            min_sigma=self.od_prior_min_sigma,
                            max_sigma=self.od_prior_max_sigma,
                        ).to(self.device)
                    else:
                        sigma = max(float(self.prior_sigma), float(self.geom_sigma_scale) * (float(r_px) / max(1.0, float(self.image_size))))
                        prior = _gaussian_prior_at(self.image_size, self.image_size, cy_px, cx_px, sigma=sigma).to(self.device)
                    # apply fov mask to suppress black borders
                    fov = torch.from_numpy(fov_u8_geom.astype(np.float32)).to(device=self.device, dtype=torch.float32)
                    if fov.shape[0] == self.image_size and fov.shape[1] == self.image_size:
                        prior = torch.clamp(prior * fov, 0.0, 1.0)
                    geom_prior_for_adaptive = prior.detach().cpu()
                    geom_info_for_adaptive = (float(cy_px), float(cx_px), float(r_px))
                    geom_conf_for_adaptive = float(conf)
                    prior_method = "geom"
                    if self.use_hybrid_prior:
                        prior_candidates.append((prior.detach().cpu(), float(conf), self.hybrid_w_geom))
                        prior_conf_votes.append((float(conf), self.hybrid_w_geom))
                    # Calibrate cup size from a weak edge-ring cue (only if confident enough).
                    r_cup, ring_rel = _estimate_cup_radius_by_ring(x.detach(), fov_u8=fov_u8_geom, disc_geom=geom)
                    if r_cup is not None and float(r_px) > 1.0 and float(ring_rel) >= 0.20:
                        rr = float(max(0.10, min(0.80, float(r_cup) / float(r_px))))
                        cup_ratio_override = float(max(0.06, min(0.40, rr * rr)))
            if self.clip is not None:
                img_key = str(name[0])
                try:
                    clip_prior = self.clip.get(img_key=img_key, x01=x.detach())
                    if self.use_hybrid_prior or not self.clip_gate_only:
                        clip_map = torch.from_numpy(clip_prior.disc_prior).to(device=self.device, dtype=torch.float32)
                        clip_map = _normalize01_hw(clip_map)
                        clip_conf = float(max(0.0, min(1.0, getattr(clip_prior, "conf_disc", 0.0))))
                        clip_conf_for_loss = clip_conf
                        mode = str(self.clip_prior_mode).lower().strip()
                        if self.use_hybrid_prior:
                            prior_candidates.append((clip_map.detach().cpu(), clip_conf, self.hybrid_w_clip))
                            prior_conf_votes.append((clip_conf, self.hybrid_w_clip))
                        elif mode == "replace":
                            prior = clip_map
                        elif mode == "conf_blend":
                            conf_min = float(max(0.0, min(0.95, self.clip_conf_min)))
                            alpha = (clip_conf - conf_min) / max(1e-6, 1.0 - conf_min)
                            alpha = float(max(0.0, min(1.0, alpha)))
                            p = float(max(0.2, self.clip_blend_power))
                            alpha = alpha ** p
                            floor = float(max(0.0, min(0.95, self.clip_blend_floor)))
                            alpha = floor + (1.0 - floor) * alpha
                            prior = _normalize01_hw(alpha * clip_map + (1.0 - alpha) * prior)
                        else:
                            prior = clip_map
                except Exception:
                    clip_prior = None
                    prior = prior

            # Initialize prompt from memory bank (same as VPTTA)
            self.model.eval()
            self.prompt.train()
            # Only mark "new sample" during the actual AdaBN-loss forward in training.
            self.model.change_BN_status(new_sample=False)

            if len(self.memory_bank.memory.keys()) >= self.neighbor:
                _, low_freq0 = self.prompt(x)
                init_data, _ = self.memory_bank.get_neighbours(keys=low_freq0.cpu().numpy(), k=self.neighbor)
            else:
                init_data = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).data
            self.prompt.update(init_data)

            # Snapshot before update (for rollback)
            snap = self._prompt_snapshot()
            snap_model = self._model_trainable_state()
            snap_vida = self._vida_snapshot()

            # Pre-update prediction (after init, before training)
            pred_pre_logit, low_freq_pre = self._infer_with_prompt(x, use_ema=False)
            seg_pre = torch.sigmoid(pred_pre_logit)

            if self.use_adaptive_od_prior:
                pred_prior_for_adaptive: Optional[torch.Tensor] = None
                pred_info_for_adaptive: Optional[Tuple[float, float, float]] = None
                pred_info, pred_conf = _estimate_pred_disc_center_radius(seg_pre, fov_u8=fov_u8)
                pred_conf_for_adaptive = float(pred_conf)
                if pred_info is not None and float(pred_conf) >= self.refine_prior_min_conf:
                    py_px, px_px, pr_px = pred_info
                    pred_prior_for_adaptive = _coarse_od_roi_prior_at(
                        self.image_size,
                        self.image_size,
                        py_px,
                        px_px,
                        pr_px,
                        roi_scale=self.od_roi_scale,
                        roi_thr=self.prior_roi_thr,
                        min_sigma=self.od_prior_min_sigma,
                        max_sigma=self.od_prior_max_sigma,
                    )
                    if fov_u8 is not None and np.asarray(fov_u8).shape == (self.image_size, self.image_size):
                        fov = torch.from_numpy(np.asarray(fov_u8).astype(np.float32)).to(dtype=torch.float32)
                        pred_prior_for_adaptive = torch.clamp(pred_prior_for_adaptive * fov, 0.0, 1.0)
                    pred_info_for_adaptive = (float(py_px), float(px_px), float(pr_px))

                selected_prior, selected_conf, selected_method = _adaptive_select_od_prior(
                    geom_prior=geom_prior_for_adaptive,
                    geom_info=geom_info_for_adaptive,
                    geom_conf=float(geom_conf_for_adaptive),
                    geom_ray_valid=float(geom_ray_valid_for_adaptive),
                    pred_prior=pred_prior_for_adaptive,
                    pred_info=pred_info_for_adaptive,
                    pred_conf=float(pred_conf_for_adaptive),
                    fallback=prior_default.detach().cpu(),
                    agree_thr=self.adaptive_agree_thr,
                    pred_conf_min=self.adaptive_pred_conf_min,
                    geom_good_conf=self.adaptive_geom_good_conf,
                    pred_adv_margin=self.adaptive_pred_adv_margin,
                )
                prior = selected_prior.to(device=self.device, dtype=torch.float32)
                adaptive_source_conf = float(selected_conf)
                prior_method = str(selected_method)

            # Estimate a per-image cup target ratio by a weak edge-ring cue inside the *predicted* disc.
            cup_ratio_target_img: Optional[float] = None
            try:
                disc_pre = torch.clamp(seg_pre[0, 0], 0.0, 1.0)
                h, w = int(disc_pre.shape[-2]), int(disc_pre.shape[-1])
                disc_bw = (disc_pre >= 0.5).detach().cpu().numpy().astype(np.uint8)
                if disc_bw.sum() > 64:
                    ys, xs = np.where(disc_bw > 0)
                    cy0 = float(ys.mean())
                    cx0 = float(xs.mean())
                    area0 = float(disc_bw.sum())
                    r0 = float(np.sqrt(max(area0, 1.0) / np.pi))
                    if fov_u8 is None:
                        fov_u8 = _estimate_fov_mask_from_x01(x.detach())
                    r_cup, ring_rel = _estimate_cup_radius_by_ring(x.detach(), fov_u8=fov_u8, disc_geom=(cy0, cx0, r0))
                    if r_cup is not None and r0 > 1.0 and ring_rel >= 0.20:
                        rr = float(max(0.10, min(0.80, float(r_cup) / float(r0))))
                        cup_ratio_target_img = float(max(0.06, min(0.40, rr * rr)))
            except Exception:
                cup_ratio_target_img = None

            # Optional: refine prior center from the (trusted) pre-update disc prediction.
            if self.refine_prior_from_pred and not self.use_adaptive_od_prior:
                refined, conf = _refine_prior_from_pred_disc(seg_pre, x, sigma=self.prior_sigma)
                if refined is not None and float(conf) >= self.refine_prior_min_conf:
                    if self.use_hybrid_prior:
                        prior_candidates.append((refined.detach().cpu(), float(conf), self.hybrid_w_pred))
                        prior_conf_votes.append((float(conf), self.hybrid_w_pred))
                    else:
                        prior = refined.to(device=self.device, dtype=torch.float32)
            if self.use_hybrid_prior and not self.use_adaptive_od_prior:
                prior = _fuse_prior_candidates(prior_candidates, fallback=prior_default.detach().cpu()).to(device=self.device, dtype=torch.float32)
            source_prior_conf = float(adaptive_source_conf) if adaptive_source_conf is not None else _weighted_confidence(prior_conf_votes)
            agreement_prior_conf = _prior_prediction_agreement_conf(
                seg_pre,
                prior,
                roi_thr=self.prior_roi_thr,
                power=self.prior_barrier_power,
            )
            prior_conf = float(max(0.0, min(1.0, 0.65 * source_prior_conf + 0.35 * agreement_prior_conf)))
            prior_loss_scale = _prior_confidence_scale(
                prior_conf,
                min_conf=self.prior_conf_min,
                power=self.prior_conf_power,
            )
            pre_stats = _gate_score(
                seg_pre,
                prior,
                x01=x,
                clip_prior=clip_prior,
                clip_gate_conf_min=self.clip_gate_conf_min,
                clip_gate_weight=self.clip_gate_weight,
                prior_roi_thr=self.prior_roi_thr,
                prior_barrier_power=self.prior_barrier_power,
            )

            # Train prompt (one step)
            _ = self._train_prompt_one(
                x,
                prior,
                cup_ratio_target=cup_ratio_target_img,
                clip_conf=clip_conf_for_loss,
                prior_conf=prior_conf,
            )

            # Post-update prediction
            pred_post_logit, low_freq_post = self._infer_with_prompt(x, use_ema=False)
            seg_post = torch.sigmoid(pred_post_logit)
            post_stats = _gate_score(
                seg_post,
                prior,
                x01=x,
                clip_prior=clip_prior,
                clip_gate_conf_min=self.clip_gate_conf_min,
                clip_gate_weight=self.clip_gate_weight,
                prior_roi_thr=self.prior_roi_thr,
                prior_barrier_power=self.prior_barrier_power,
            )

            # Decide which to keep
            choose_post = True
            if self.enable_gate:
                # Require improvement by at least gate_min_improve, otherwise prefer pre.
                # Also allow a small margin (gate_margin) to avoid over-rejecting.
                if post_stats.score > pre_stats.score - self.gate_min_improve + self.gate_margin:
                    choose_post = False
            gate_reject = bool(self.enable_gate and not choose_post)
            rpr_restored = bool(gate_reject and not self.disable_restore)

            if not choose_post:
                # rollback prompt to snapshot
                if self.disable_restore:
                    pred_use_logit = pred_post_logit
                    low_freq_use = low_freq_post
                else:
                    self._prompt_restore(snap)
                    self._model_restore_trainable(snap_model)
                    self._vida_restore(snap_vida)
                    pred_use_logit = pred_pre_logit
                    low_freq_use = low_freq_pre
            else:
                pred_use_logit = pred_post_logit
                low_freq_use = low_freq_post
                if self.use_prompt_ema:
                    self._ema_update()
                if self.ema_model is not None:
                    self._ema_model_update()

            # Update memory bank with chosen prompt state
            self.memory_bank.push(keys=low_freq_use, logits=self.prompt.data_prompt.detach().cpu().numpy())

            # Final output: if EMA is enabled, treat it as another candidate and pick the more reliable one.
            seg_output = torch.sigmoid(pred_use_logit)
            if self.use_prompt_ema and self.ema_prompt is not None:
                pred_logit_ema, _ = self._infer_with_prompt(x, use_ema=True)
                seg_ema = torch.sigmoid(pred_logit_ema)
                if _gate_score(
                    seg_ema,
                    prior,
                    x01=x,
                    clip_prior=clip_prior,
                    clip_gate_conf_min=self.clip_gate_conf_min,
                    clip_gate_weight=self.clip_gate_weight,
                    prior_roi_thr=self.prior_roi_thr,
                    prior_barrier_power=self.prior_barrier_power,
                ).score + 1e-6 < _gate_score(
                    seg_output,
                    prior,
                    x01=x,
                    clip_prior=clip_prior,
                    clip_gate_conf_min=self.clip_gate_conf_min,
                    clip_gate_weight=self.clip_gate_weight,
                    prior_roi_thr=self.prior_roi_thr,
                    prior_barrier_power=self.prior_barrier_power,
                ).score:
                    seg_output = seg_ema

            # If model EMA is enabled, also consider its output (teacher) as a candidate (no update).
            if self.ema_model is not None:
                with torch.no_grad():
                    prompt_x, _ = self.prompt(x)
                    if self.vida_adapter is not None:
                        prompt_x = self.vida_adapter(prompt_x)
                    t_logit, _, _ = self.ema_model(prompt_x)
                    t_prob = torch.sigmoid(t_logit)
                if _gate_score(
                    t_prob,
                    prior,
                    x01=x,
                    clip_prior=clip_prior,
                    clip_gate_conf_min=self.clip_gate_conf_min,
                    clip_gate_weight=self.clip_gate_weight,
                    prior_roi_thr=self.prior_roi_thr,
                    prior_barrier_power=self.prior_barrier_power,
                ).score + 1e-6 < _gate_score(
                    seg_output,
                    prior,
                    x01=x,
                    clip_prior=clip_prior,
                    clip_gate_conf_min=self.clip_gate_conf_min,
                    clip_gate_weight=self.clip_gate_weight,
                    prior_roi_thr=self.prior_roi_thr,
                    prior_barrier_power=self.prior_barrier_power,
                ).score:
                    seg_output = t_prob

            seg_output_raw = seg_output.detach().clone()
            if self.postprocess_cc:
                seg_output = _postprocess_disc_cup(
                    seg_output,
                    disc_thr=self.pp_disc_thr,
                    cup_thr=self.pp_cup_thr,
                    cup_target_ratio=float(cup_ratio_target_img) if cup_ratio_target_img is not None else (float(cup_ratio_override) if cup_ratio_override is not None else self.pp_cup_target_ratio),
                    disc_target_ratio=float(self.pp_disc_target_ratio),
                    disc_cc_mode=self.pp_disc_cc_mode,
                    cup_cc_mode=self.pp_cup_cc_mode,
                    fov_u8=fov_u8,
                )

            if self.save_visuals:
                _save_optic_prediction_visuals(
                    out_dir=self.visual_dir,
                    index=sample_idx,
                    name=str(name[0]),
                    image_chw01=data["data"][0],
                    gt_2hw=data["mask"][0],
                    pred_2hw=seg_output.detach().cpu().numpy()[0],
                    disc_thr=0.5,
                    cup_thr=0.5,
                )

            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                metrics_test[i] += metrics[i]

            if self.save_case_outputs:
                pred_np = seg_output.detach().cpu().numpy()
                gt_np = y.detach().cpu().numpy()
                for b in range(pred_np.shape[0]):
                    case_name = str(name[b])
                    stem = f"{sample_idx:04d}_{_optic_domain_from_name(case_name)}_{_safe_case_stem(case_name)}"
                    pred_2hw = (pred_np[b] > 0.5).astype(np.uint8)
                    gt_2hw = (gt_np[b] > 0.5).astype(np.uint8)
                    np.savez_compressed(
                        os.path.join(self.case_output_dir, stem + ".npz"),
                        pred=pred_2hw,
                        gt=gt_2hw,
                        name=case_name,
                        domain=_optic_domain_from_name(case_name),
                    )
                    row: Dict[str, object] = {
                        "index": sample_idx,
                        "name": case_name,
                        "domain": _optic_domain_from_name(case_name),
                        "Disc_Dice": float(metrics[0][b]),
                        "Disc_ASD": float(metrics[1][b]),
                        "Cup_Dice": float(metrics[2][b]),
                        "Cup_ASD": float(metrics[3][b]),
                        "Mean_Dice": float((float(metrics[0][b]) + float(metrics[2][b])) / 2.0),
                    }
                    row.update(_optic_violation_metrics(pred_2hw))
                    row.update(_prefix_metrics(_optic_binary_anatomy_metrics(gt_2hw), "gt_"))
                    row.update(_prefix_metrics(_optic_prob_anatomy_metrics(seg_pre[b : b + 1]), "pre_raw_"))
                    row.update(_prefix_metrics(_optic_prob_anatomy_metrics(seg_post[b : b + 1]), "post_raw_"))
                    row.update(_prefix_metrics(_optic_prob_anatomy_metrics(seg_output_raw[b : b + 1]), "selected_raw_"))
                    row.update(_prefix_metrics(_optic_prob_anatomy_metrics(seg_output[b : b + 1]), "final_"))
                    row.update(
                        {
                            "prior_method": str(prior_method),
                            "prior_geom_conf": float(geom_conf_for_adaptive),
                            "prior_pred_conf": float(pred_conf_for_adaptive),
                            "prior_conf": float(prior_conf),
                            "prior_loss_scale": float(prior_loss_scale),
                            "prior_source_conf": float(source_prior_conf),
                            "prior_agreement_conf": float(agreement_prior_conf),
                            "gate_reject": float(gate_reject),
                            "rpr_restored": float(rpr_restored),
                            "disable_restore": float(self.disable_restore),
                            "pre_gate_score": float(pre_stats.score),
                            "post_gate_score": float(post_stats.score),
                            "gate_score_delta": float(post_stats.score - pre_stats.score),
                            "pre_gate_entropy": float(pre_stats.ent),
                            "post_gate_entropy": float(post_stats.ent),
                            "pre_gate_center_dist": float(pre_stats.center_dist),
                            "post_gate_center_dist": float(post_stats.center_dist),
                            "pre_gate_disc_outside": float(pre_stats.disc_outside),
                            "post_gate_disc_outside": float(post_stats.disc_outside),
                            "pre_gate_cup_outside": float(pre_stats.cup_outside),
                            "post_gate_cup_outside": float(post_stats.cup_outside),
                        }
                    )
                    case_rows.append(row)

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean: Dict[str, float] = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = float(test_metrics_y[i])
        mean_dice = float((print_test_metric_mean["Disc_Dice"] + print_test_metric_mean["Cup_Dice"]) / 2.0)
        print(f"{print_prefix} Metrics: ", print_test_metric_mean)
        print(f"Mean Dice: {mean_dice}")
        if self.save_case_outputs and case_rows:
            csv_name = f"case_metrics_{print_prefix.replace(' ', '_').replace('/', '_')}.csv"
            csv_path = os.path.join(self.case_output_dir, csv_name)
            fieldnames = list(case_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(case_rows)
            print(f"Saved case outputs: {self.case_output_dir}")
            print(f"Saved case metrics: {csv_path}")
        return print_test_metric_mean, mean_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset / protocol
    parser.add_argument("--Source_Dataset", type=str, default="RIM_ONE_r3", help="RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS")
    parser.add_argument("--Target_Dataset", type=list)
    parser.add_argument(
        "--Target_Domains",
        type=str,
        default="",
        help="Optional comma-separated target domains for visualization/debug, e.g. REFUGE. Default: all except source.",
    )
    parser.add_argument("--eval_split", type=str, default="train_test", choices=["train_test", "test"], help="Which csv split(s) to evaluate on per domain.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--out_ch", type=int, default=2)

    # Optimizer (prompt only)
    parser.add_argument("--optimizer", type=str, default="Adam", help="SGD/Adam")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--model_lr", type=float, default=1e-4, help="LR for adapted model params (BN affine / head).")
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Long-term CTTA rounds (repeat the target stream without reset). Table-4 style uses 3.",
    )

    # Memory / prompt / BN warm-up
    parser.add_argument("--memory_size", type=int, default=40)
    parser.add_argument("--neighbor", type=int, default=16)
    parser.add_argument("--prompt_alpha", type=float, default=0.01)
    parser.add_argument("--warm_n", type=int, default=5)

    # Gate/rollback + prior
    parser.add_argument("--enable_gate", action="store_true")
    parser.add_argument("--disable_restore", action="store_true", help="Compute gate risk but do not restore snapshots on unsafe updates (w/o RPR ablation).")
    parser.add_argument("--gate_margin", type=float, default=0.02, help="Allow small worse score before rejecting post-update.")
    parser.add_argument("--gate_min_improve", type=float, default=0.00, help="Require post to improve by this amount to accept.")
    parser.add_argument("--prior_sigma", type=float, default=0.22, help="Central Gaussian sigma in normalized coordinates.")
    parser.add_argument("--use_geom_prior", action="store_true", help="Use a geometric (intensity-based) disc localization prior.")
    parser.add_argument("--geom_min_conf", type=float, default=0.15, help="Minimum confidence to accept geometric prior.")
    parser.add_argument("--use_adaptive_od_prior", action="store_true", help="Use adaptive coarse OD ROI prior from geometry + pre-update prediction.")
    parser.add_argument("--od_roi_scale", type=float, default=2.50, help="Coarse ROI support radius as a multiple of estimated OD radius.")
    parser.add_argument("--od_prior_min_sigma", type=float, default=0.035, help="Minimum Gaussian sigma for adaptive OD ROI prior.")
    parser.add_argument("--od_prior_max_sigma", type=float, default=0.36, help="Maximum Gaussian sigma for adaptive OD ROI prior.")
    parser.add_argument("--adaptive_agree_thr", type=float, default=1.25, help="Geom/pred center agreement threshold in estimated OD radii.")
    parser.add_argument("--adaptive_pred_conf_min", type=float, default=0.50, help="Minimum pred-prior confidence when geom and pred disagree.")
    parser.add_argument("--adaptive_geom_good_conf", type=float, default=0.55, help="Geom confidence threshold for trusting geometry on disagreement.")
    parser.add_argument("--adaptive_pred_adv_margin", type=float, default=0.15, help="Pred confidence margin over geom required on disagreement.")
    parser.add_argument("--use_hybrid_prior", action="store_true", help="Fuse central/geometric/pre-prediction/CLIP OD priors into one soft ROI prior.")
    parser.add_argument("--hybrid_w_default", type=float, default=0.12, help="Base fusion weight for the central fallback prior.")
    parser.add_argument("--hybrid_w_geom", type=float, default=0.35, help="Base fusion weight for the intensity/geometric OD prior.")
    parser.add_argument("--hybrid_w_pred", type=float, default=0.35, help="Base fusion weight for the pre-update prediction prior.")
    parser.add_argument("--hybrid_w_clip", type=float, default=0.30, help="Base fusion weight for the BioMedCLIP OD prior.")
    parser.add_argument(
        "--refine_prior_from_pred",
        action="store_true",
        help="Refine the prior center from the model's trusted pre-update disc prediction (unsupervised).",
    )
    parser.add_argument(
        "--refine_prior_min_conf",
        type=float,
        default=0.30,
        help="Min confidence to accept refined prior from prediction (higher => safer).",
    )
    parser.add_argument(
        "--prior_use_roi_weight",
        type=int,
        default=1,
        help="Use the disc prior to weight losses (ROI-weighted). 1=on (default), 0=off.",
    )
    parser.add_argument("--roi_w_min", type=float, default=0.20, help="Minimum weight outside ROI when prior_use_roi_weight.")
    parser.add_argument("--roi_w_gamma", type=float, default=1.0, help="Gamma on prior before mapping to weights.")
    parser.add_argument("--geom_sigma_scale", type=float, default=0.55, help="Sigma floor = max(prior_sigma, geom_sigma_scale * r_disc_norm).")

    # Loss weights
    parser.add_argument("--w_bn", type=float, default=1.0)
    parser.add_argument("--w_prior", type=float, default=0.03)
    parser.add_argument(
        "--prior_loss_mode",
        type=str,
        default="bce",
        choices=["bce", "barrier"],
        help="bce=legacy Gaussian target; barrier=penalize disc mass outside soft prior ROI.",
    )
    parser.add_argument("--prior_roi_thr", type=float, default=0.20, help="Prior value threshold defining the soft ROI for barrier mode.")
    parser.add_argument("--prior_barrier_power", type=float, default=2.0, help="Power for outside-ROI barrier weights.")
    parser.add_argument("--prior_conf_gate", action="store_true", help="Scale localization-prior losses by sample-level prior confidence q_t.")
    parser.add_argument("--prior_conf_min", type=float, default=0.45, help="Minimum q_t before prior losses are enabled.")
    parser.add_argument("--prior_conf_power", type=float, default=1.0, help="Nonlinear power for q_t -> loss scale.")
    parser.add_argument("--w_cup_prior", type=float, default=0.0, help="Weak cup prior weight (derived from disc prior).")
    parser.add_argument("--cup_prior_sigma_scale", type=float, default=0.60, help="Cup prior sigma = prior_sigma * scale.")
    parser.add_argument("--w_inside", type=float, default=0.20)
    parser.add_argument("--w_entropy", type=float, default=0.00)
    parser.add_argument("--w_clip", type=float, default=0.0, help="Tiny CLIP-guided loss weight (default off).")
    parser.add_argument("--w_consistency", type=float, default=0.0, help="EMA teacher consistency weight (default off).")
    parser.add_argument("--postprocess_cc", action="store_true", help="Postprocess output: largest CC + cup-in-disc constraint.")
    parser.add_argument("--pp_disc_thr", type=float, default=0.50, help="Postprocess threshold for disc.")
    parser.add_argument(
        "--pp_disc_target_ratio",
        type=float,
        default=0.08,
        help="Adaptive disc area ratio if pp_disc_thr<=0 (fraction of valid FOV pixels).",
    )
    parser.add_argument("--pp_cup_thr", type=float, default=0.35, help="Postprocess threshold for cup.")
    parser.add_argument("--pp_cup_target_ratio", type=float, default=0.18, help="Adaptive cup area ratio if pp_cup_thr<=0.")
    parser.add_argument("--w_cup_ratio", type=float, default=0.0, help="Soft cup-size (cup/disc ratio) regularizer weight.")
    parser.add_argument("--cup_ratio_target", type=float, default=0.18, help="Default cup/disc area ratio target for w_cup_ratio.")
    parser.add_argument(
        "--pp_disc_cc_mode",
        type=str,
        default="largest",
        choices=["largest", "probmass"],
        help="How to pick the disc connected component during postprocess_cc.",
    )
    parser.add_argument(
        "--pp_cup_cc_mode",
        type=str,
        default="largest",
        choices=["largest", "probmass", "probmass_in_disc"],
        help="How to pick the cup connected component during postprocess_cc (optionally constrained in disc).",
    )

    # BioMedCLIP weak prior (optional)
    parser.add_argument("--use_clip_prior", action="store_true", help="Use BioMedCLIP to derive a weak disc localization prior.")
    parser.add_argument("--clip_root", type=str, default="", help="Path to BioMedCLIP weights folder.")
    parser.add_argument("--clip_cache", type=str, default="", help="Optional cache dir for CLIP priors (npz).")
    parser.add_argument("--clip_softmax_temp", type=float, default=0.08)
    parser.add_argument("--clip_patch_sigma", type=float, default=4.0)
    parser.add_argument("--clip_prior_sigma", type=float, default=0.18)
    parser.add_argument("--clip_gate_only", action="store_true", help="Only use CLIP for gating (no loss term).")
    parser.add_argument("--clip_gate_conf_min", type=float, default=0.15, help="Min CLIP confidence for clip_gate_only signal.")
    parser.add_argument("--clip_gate_weight", type=float, default=0.10, help="Weight of CLIP agreement term inside reliability gate.")
    parser.add_argument(
        "--clip_prior_mode",
        type=str,
        default="replace",
        choices=["replace", "conf_blend"],
        help="How CLIP prior is injected: direct replace or confidence-based blend with geometric/default prior.",
    )
    parser.add_argument("--clip_conf_min", type=float, default=0.15, help="Minimum CLIP confidence for conf_blend transition.")
    parser.add_argument("--clip_blend_floor", type=float, default=0.15, help="Minimum CLIP blend ratio in conf_blend mode.")
    parser.add_argument("--clip_blend_power", type=float, default=1.0, help="Confidence-to-blend exponent in conf_blend mode.")
    parser.add_argument("--clip_weight_by_conf", action="store_true", help="Scale CLIP loss weight by CLIP confidence.")

    # Stronger adaptation scope (optional)
    parser.add_argument("--use_vida", action="store_true", help="Enable lightweight ViDA/SDA adapter on prompted input.")
    parser.add_argument("--vida_rank", type=int, default=4, help="Low-rank width for ViDA/SDA adapter.")
    parser.add_argument("--vida_init_scale", type=float, default=0.0, help="Initial residual strength for ViDA adapter (0 keeps identity at start).")
    parser.add_argument("--adapt_bn_affine", action="store_true", help="Also adapt AdaBN affine (gamma/beta).")
    parser.add_argument("--adapt_head", action="store_true", help="Also adapt seg_head conv weights.")
    parser.add_argument("--use_model_ema", action="store_true", help="Maintain EMA of adapted model params as teacher.")
    parser.add_argument("--model_ema_m", type=float, default=0.99)

    # Prompt EMA
    parser.add_argument("--use_prompt_ema", action="store_true")
    parser.add_argument("--ema_m", type=float, default=0.90)

    # Paths
    parser.add_argument("--path_save_log", type=str, default="./logs")
    parser.add_argument("--model_root", type=str, default="./models")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=0, help="Optional early stop for visualization/debug. 0 means full stream.")
    parser.add_argument("--save_visuals", action="store_true", help="Save image/GT/pred overlays for qualitative comparison.")
    parser.add_argument("--visual_dir", type=str, default="", help="Directory used when --save_visuals is set.")
    parser.add_argument("--save_case_outputs", action="store_true", help="Save per-case prediction/GT masks and anatomical violation metrics.")
    parser.add_argument("--case_output_dir", type=str, default="", help="Directory used when --save_case_outputs is set.")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")

    config = parser.parse_args()
    if str(config.Target_Domains).strip():
        config.Target_Dataset = [x.strip() for x in str(config.Target_Domains).split(",") if x.strip()]
    else:
        config.Target_Dataset = ["RIM_ONE_r3", "REFUGE", "ORIGA", "REFUGE_Valid", "Drishti_GS"]
        config.Target_Dataset.remove(config.Source_Dataset)

    tta = RATTA_OPTIC(config)
    tta.run()
