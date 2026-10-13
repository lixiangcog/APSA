#!/usr/bin/env python3
"""APSA-style safety wrapper on top of the A3-TTA MMS protocol.

This runner keeps the original A3-TTA implementation untouched. A3 produces an
online adapted branch; the wrapper computes cardiac reliability cues without
target labels, optionally rolls A3 back, and chooses source/adapted/fused output.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from conf import cfg, merge_from_file, reset_cfg
from robustbench.data import convert_2d, get_dataset
from robustbench.tta import setup_a3_tta
from robustbench.utils import load_model
from utils.evaluate import get_multi_class_evaluation_score


CLASS_NAMES = ["class_1", "class_2", "class_3"]
ADAPTER_NAMES = (
    "source",
    "tent",
    "a3-lite",
    "a3-no-anchor",
    "a3-no-ema",
    "a3-no-anchor-no-ema",
    "a3-full",
)
TENT_TRAIN_SCOPES = (
    "bn",
    "bn-head",
    "bn-last-decoder",
    "bn-decoder",
)
SOURCE_CALIBRATION_CORRUPTIONS = (
    "none",
    "gamma-low",
    "gamma-high",
    "bias-field",
    "mixed",
)
PREUPDATE_RPR_FEATURES = (
    "entropy_increase",
    "prior_align_drop",
    "center_distance_increase",
    "topology_drop",
    "confidence_drop",
    "pre_adapt_drift",
    "fg_log_ratio_abs",
    "rv_soft_area_drop",
    "rv_pred_area_drop",
    "rv_confidence_drop",
    "class_collapse_fraction",
)


def apply_source_calibration_corruption(
    volume: torch.Tensor,
    mode: str,
    *,
    batch_index: Optional[int] = None,
) -> torch.Tensor:
    """Deterministic MRI intensity shifts used only for source-domain calibration."""
    mode = str(mode).strip().lower()
    if mode == "mixed":
        if batch_index is None:
            raise ValueError("Mixed source calibration requires a deterministic batch index")
        mode = ("none", "gamma-low", "gamma-high", "bias-field")[int(batch_index) % 4]
    if mode == "none":
        return volume
    image01 = ((volume + 1.0) * 0.5).clamp(0.0, 1.0)
    if mode == "gamma-low":
        shifted = image01.pow(0.70)
    elif mode == "gamma-high":
        shifted = image01.pow(1.50)
    elif mode == "bias-field":
        height, width = image01.shape[-2:]
        yy = torch.linspace(-1.0, 1.0, height, device=volume.device, dtype=volume.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=volume.device, dtype=volume.dtype)
        field = 1.0 + 0.30 * xx.view(1, 1, 1, width) + 0.15 * yy.view(1, 1, height, 1)
        shifted = (image01 * field).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported source calibration corruption: {mode!r}")
    return shifted.mul(2.0).sub(1.0)


def _build_online_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    lr_override: Optional[float] = None,
) -> torch.optim.Optimizer:
    params = list(params)
    if not params:
        raise ValueError("The selected adapter has no trainable parameters.")
    method = str(cfg.OPTIM.METHOD).lower()
    learning_rate = float(cfg.OPTIM.LR) if lr_override is None else float(lr_override)
    if learning_rate <= 0.0:
        raise ValueError(f"Adapter learning rate must be positive, got {learning_rate}")
    if method == "adam":
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(float(cfg.OPTIM.BETA), 0.999),
            weight_decay=float(cfg.OPTIM.WD),
        )
    if method == "sgd":
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=float(cfg.OPTIM.MOMENTUM),
            dampening=float(cfg.OPTIM.DAMPENING),
            weight_decay=float(cfg.OPTIM.WD),
            nesterov=bool(cfg.OPTIM.NESTEROV),
        )
    raise ValueError(f"Unsupported optimizer for adapter: {cfg.OPTIM.METHOD!r}")


def _set_adaptation_mode(model: nn.Module, *, disable_dropout: bool) -> None:
    """Keep BN adaptive while optionally making all stochastic layers deterministic."""
    if not disable_dropout:
        model.train()
        return
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()


class SourceAdapter(nn.Module):
    """Frozen source inference with the same call contract as online adapters."""

    adapter_name = "source"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        self.optimizer = None
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor, label=None, names=None) -> torch.Tensor:
        del label, names
        self.model.eval()
        with torch.no_grad():
            return self.model(x)


class TentAdapter(nn.Module):
    """Lightweight entropy/APL adaptation without any A3 state or loss."""

    adapter_name = "tent"

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: Optional[float] = None,
        train_scope: str = "bn",
        disable_dropout: bool = False,
    ):
        super().__init__()
        self.model = model
        self.disable_dropout = bool(disable_dropout)
        _set_adaptation_mode(self.model, disable_dropout=self.disable_dropout)
        self.model.requires_grad_(False)
        params: List[torch.nn.Parameter] = []
        self.trainable_names: List[str] = []
        self.train_scope = str(train_scope).strip().lower()
        if self.train_scope not in TENT_TRAIN_SCOPES:
            raise ValueError(
                f"Unknown TENT train scope {train_scope!r}; expected one of {TENT_TRAIN_SCOPES}"
            )
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ("weight", "bias"):
                    params.append(param)
                    self.trainable_names.append(f"{module_name}.{param_name}")

        # The default scope exactly preserves the original BN-only adapter.
        # Wider scopes only expose generic U-Net decoder parameters to APSA's
        # APL loss; no A3 memory, anchor, EMA, or feature replacement is used.
        bn_param_ids = {id(param) for param in params}
        if self.train_scope != "bn":
            for param_name, param in self.model.named_parameters():
                include = False
                if self.train_scope == "bn-head":
                    include = param_name.startswith("dec1.last.") or param_name.startswith("out_conv.")
                elif self.train_scope == "bn-last-decoder":
                    include = (
                        param_name.startswith("dec1.up_path.3.")
                        or param_name.startswith("dec1.last.")
                        or param_name.startswith("up4.")
                        or param_name.startswith("out_conv.")
                    )
                elif self.train_scope == "bn-decoder":
                    include = (
                        param_name.startswith("dec1.")
                        or param_name.startswith("decoder.")
                        or param_name.startswith("up1.")
                        or param_name.startswith("up2.")
                        or param_name.startswith("up3.")
                        or param_name.startswith("up4.")
                        or param_name.startswith("out_conv.")
                    )
                if include and id(param) not in bn_param_ids:
                    param.requires_grad_(True)
                    params.append(param)
                    self.trainable_names.append(param_name)
        self.trainable_parameter_count = int(sum(param.numel() for param in params))
        self.learning_rate = float(cfg.OPTIM.LR) if learning_rate is None else float(learning_rate)
        self.optimizer = _build_online_optimizer(params, lr_override=self.learning_rate)
        self.steps = max(1, int(cfg.OPTIM.STEPS))
        self.last_loss = float("nan")
        self.last_temporal_consistency_loss = 0.0
        self.last_temporal_reliability_fraction = 0.0
        self.model_ema: Optional[nn.Module] = None
        self.safe_ema_alpha = 0.999

    def enable_safe_ema(self, alpha: float) -> None:
        self.safe_ema_alpha = max(0.0, min(1.0, float(alpha)))
        self.model_ema = copy.deepcopy(self.model).eval()
        for param in self.model_ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def safe_ema_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_ema is None:
            raise RuntimeError("Safe EMA candidate requested before enable_safe_ema().")
        self.model_ema.eval()
        return self.model_ema(x)

    @torch.no_grad()
    def update_safe_ema(self) -> None:
        if self.model_ema is None:
            return
        alpha = float(self.safe_ema_alpha)
        for teacher_param, student_param in zip(self.model_ema.parameters(), self.model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)
        for teacher_buffer, student_buffer in zip(self.model_ema.buffers(), self.model.buffers()):
            if teacher_buffer.dtype.is_floating_point:
                teacher_buffer.data.mul_(alpha).add_(student_buffer.data, alpha=1.0 - alpha)
            else:
                teacher_buffer.data.copy_(student_buffer.data)

    def forward(self, x: torch.Tensor, label=None, names=None) -> torch.Tensor:
        del label, names
        _set_adaptation_mode(self.model, disable_dropout=self.disable_dropout)
        for _ in range(self.steps):
            with torch.enable_grad():
                logits = self.model(x)
                loss = softmax_entropy(logits).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.last_loss = float(loss.detach().cpu().item())
        # All adapters expose the prediction produced by the post-update state.
        with torch.no_grad():
            return self.model(x)

    def forward_apl(
        self,
        x: torch.Tensor,
        prior: Dict[str, torch.Tensor],
        source_logits: torch.Tensor,
        *,
        apl_weight: float,
        entropy_lambda: float,
        loc_lambda: float,
        bce_lambda: float,
        ts_lambda: float,
        source_consistency_lambda: float,
        weight_gamma: float,
        weight_min: float,
        class_weights: Optional[Iterable[float]],
        consistency_class_weights: Optional[Iterable[float]],
        temporal_consistency_lambda: float = 0.0,
        temporal_consistency_class_weights: Optional[Iterable[float]] = None,
        temporal_consistency_conf_threshold: float = 0.70,
        temporal_consistency_rv_conf_threshold: float = 0.85,
        temporal_consistency_prior_threshold: float = 0.25,
        temporal_consistency_temperature: float = 0.50,
    ) -> torch.Tensor:
        """BN-only online adaptation guided by APSA's anatomical prior."""
        blend = max(0.0, min(1.0, float(apl_weight)))
        _set_adaptation_mode(self.model, disable_dropout=self.disable_dropout)
        temporal_lambda = max(0.0, float(temporal_consistency_lambda))
        temporal_teacher_prob: Optional[torch.Tensor] = None
        if temporal_lambda > 0.0:
            if self.model_ema is None:
                raise RuntimeError("Temporal consistency requires an enabled rollback-protected EMA teacher.")
            self.model_ema.eval()
            with torch.no_grad():
                temporal_teacher_prob = torch.softmax(self.model_ema(x), dim=1).detach()
        temporal_loss_total = 0.0
        temporal_reliability_total = 0.0
        for _ in range(self.steps):
            with torch.enable_grad():
                logits = self.model(x)
                prob = torch.softmax(logits, dim=1)
                p_fg = prob[:, 1:].sum(dim=1, keepdim=True)
                w = prior["fg"].to(device=x.device, dtype=logits.dtype).clamp(0.0, 1.0)
                if w.shape[0] == 1 and logits.shape[0] > 1:
                    w = w.repeat(logits.shape[0], 1, 1, 1)
                wg = w.pow(max(0.1, float(weight_gamma)))
                wg = wg * (1.0 - float(weight_min)) + float(weight_min)

                base_ent = softmax_entropy(logits).mean()
                apl_ent = multiclass_weighted_entropy_per_sample(logits, wg).mean()
                loc = (p_fg * (1.0 - w)).mean() - (p_fg * w).mean()
                ce = soft_prior_ce(logits, prior["prob"], wg, class_weights=class_weights)
                src_prob = torch.softmax(source_logits.detach(), dim=1)
                src_cons = weighted_probability_consistency(prob, src_prob, consistency_class_weights)
                src_conf = src_prob.max(dim=1)[0].mean().clamp(0.0, 1.0)

                apl_loss = float(entropy_lambda) * apl_ent
                apl_loss = apl_loss + float(loc_lambda) * loc
                apl_loss = apl_loss + float(bce_lambda) * ce
                apl_loss = apl_loss + float(ts_lambda) * src_cons * src_conf
                apl_loss = apl_loss + float(source_consistency_lambda) * src_cons
                temporal_loss = logits.new_tensor(0.0)
                temporal_reliability = 0.0
                if temporal_teacher_prob is not None:
                    temporal_loss, temporal_reliability = reliability_gated_temporal_consistency(
                        logits,
                        temporal_teacher_prob,
                        prior["prob"],
                        class_weights=temporal_consistency_class_weights,
                        confidence_threshold=temporal_consistency_conf_threshold,
                        rv_confidence_threshold=temporal_consistency_rv_conf_threshold,
                        prior_alignment_threshold=temporal_consistency_prior_threshold,
                        teacher_temperature=temporal_consistency_temperature,
                    )
                    apl_loss = apl_loss + temporal_lambda * temporal_loss
                loss = (1.0 - blend) * base_ent + blend * apl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.last_loss = float(loss.detach().cpu().item())
                temporal_loss_total += float(temporal_loss.detach().cpu().item())
                temporal_reliability_total += float(temporal_reliability)
        self.last_temporal_consistency_loss = temporal_loss_total / float(self.steps)
        self.last_temporal_reliability_fraction = temporal_reliability_total / float(self.steps)
        with torch.no_grad():
            return self.model(x)


class A3LiteAdapter(nn.Module):
    """A3 ablation retaining only EMA student-teacher consistency."""

    adapter_name = "a3-lite"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.train()
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.model_ema = copy.deepcopy(self.model).eval()
        for param in self.model_ema.parameters():
            param.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(cfg.OPTIM.LR),
            betas=(0.5, 0.999),
        )
        self.steps = max(1, int(cfg.OPTIM.STEPS))
        self.ema_alpha = float(cfg.OPTIM.MT)
        self.last_loss = float("nan")

    @torch.no_grad()
    def _update_teacher(self) -> None:
        alpha = max(0.0, min(1.0, self.ema_alpha))
        student_params = self.model.enc.parameters() if hasattr(self.model, "enc") else self.model.parameters()
        teacher_params = self.model_ema.enc.parameters() if hasattr(self.model_ema, "enc") else self.model_ema.parameters()
        for teacher_param, student_param in zip(teacher_params, student_params):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)

    def forward(self, x: torch.Tensor, label=None, names=None) -> torch.Tensor:
        del label, names
        self.model.train()
        self.model_ema.eval()
        for _ in range(self.steps):
            with torch.no_grad():
                teacher_prob = torch.softmax(self.model_ema(x), dim=1)
            with torch.enable_grad():
                student_logits = self.model(x)
                loss = -(teacher_prob.detach() * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.last_loss = float(loss.detach().cpu().item())
            self._update_teacher()
        with torch.no_grad():
            return self.model(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_paired_audit_runtime() -> None:
    """Fix precision settings while retaining a same-state numerical control."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def _write_rows(path: Path, head: Iterable[str], rows: Iterable[Iterable]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(head))
        writer.writerows(rows)


def _format_pm(mean: float, std: float) -> str:
    return f"{mean:.4f}+/-{std:.4f}"


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1).clamp_min(1e-6)
    return -(prob * prob.log()).sum(dim=1)


def mean_normalized_entropy(logits: torch.Tensor) -> float:
    ent = softmax_entropy(logits).mean()
    return float((ent / math.log(logits.shape[1])).detach().cpu().item())


def mean_confidence(logits: torch.Tensor) -> float:
    prob = torch.softmax(logits, dim=1)
    return float(prob.max(dim=1)[0].mean().detach().cpu().item())


def probability_drift(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    prob_a = torch.softmax(logits_a, dim=1)
    prob_b = torch.softmax(logits_b, dim=1)
    return float(torch.mean(torch.abs(prob_a - prob_b)).detach().cpu().item())


def confidence_aware_fusion(
    adapted_logits: torch.Tensor,
    source_logits: torch.Tensor,
    source_weight: float,
    source_weight_by_class: Optional[Iterable[float]] = None,
    adaptive_class_fusion: bool = False,
    adaptive_fusion_strength: float = 0.0,
    adaptive_fusion_max: float = 0.75,
) -> torch.Tensor:
    adapted_prob = torch.softmax(adapted_logits, dim=1)
    source_prob = torch.softmax(source_logits, dim=1)
    confidence = adapted_prob.max(dim=1, keepdim=True)[0]
    uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
    w = max(0.0, min(1.0, float(source_weight)))
    if source_weight_by_class is None:
        source_mix = w * uncertainty
    else:
        weights = [max(0.0, min(1.0, float(x))) for x in source_weight_by_class]
        class_count = int(adapted_prob.shape[1])
        if len(weights) == class_count - 1:
            weights = [w] + weights
        if len(weights) != class_count:
            raise ValueError(f"Expected {class_count - 1} foreground or {class_count} class fusion weights, got {len(weights)}")
        weight_tensor = torch.tensor(weights, device=adapted_prob.device, dtype=adapted_prob.dtype).view(1, class_count, 1, 1)
        if adaptive_class_fusion and float(adaptive_fusion_strength) > 0.0:
            src_area = source_prob.mean(dim=(2, 3), keepdim=True)
            ada_area = adapted_prob.mean(dim=(2, 3), keepdim=True)
            src_peak = source_prob.amax(dim=(2, 3), keepdim=True)
            ada_peak = adapted_prob.amax(dim=(2, 3), keepdim=True)
            area_drift = (ada_area - src_area).abs() / src_area.clamp_min(0.005)
            conf_drop = (src_peak - ada_peak).clamp_min(0.0) / 0.20
            boost = float(adaptive_fusion_strength) * (0.65 * area_drift.clamp(0.0, 1.0) + 0.35 * conf_drop.clamp(0.0, 1.0))
            boost[:, 0:1] = 0.0
            max_w = max(0.0, min(1.0, float(adaptive_fusion_max)))
            weight_tensor = (weight_tensor + boost).clamp(0.0, max_w)
        source_mix = weight_tensor * uncertainty
    fused = (1.0 - source_mix) * adapted_prob + source_mix * source_prob
    fused = fused / fused.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.log(fused.clamp_min(1e-6))


def _scaled_fusion_weights(weights: Optional[Iterable[float]], scale: float) -> Optional[List[float]]:
    if weights is None:
        return None
    s = max(0.0, float(scale))
    return [max(0.0, min(1.0, float(x) * s)) for x in weights]


def _tta_flip_dims(name: str) -> Optional[Tuple[int, ...]]:
    token = str(name).strip().lower()
    if token in ("h", "horizontal", "width", "x"):
        return (-1,)
    if token in ("v", "vertical", "height", "y"):
        return (-2,)
    if token in ("hv", "vh", "both"):
        return (-2, -1)
    return None


def cleanup_cardiac_logits(
    logits: torch.Tensor,
    *,
    min_area_abs: int,
    min_area_ratio: float,
    logit_bias: float,
    keep_largest_classes: Iterable[int],
    fill_hole_classes: Iterable[int],
    reference_logits: Optional[torch.Tensor] = None,
    consensus_logits: Optional[torch.Tensor] = None,
    prior_prob: Optional[torch.Tensor] = None,
    protected_classes: Optional[Iterable[int]] = None,
    protection_min_area_abs: int = 8,
    protection_candidate_confidence: float = 0.85,
    protection_reference_overlap: float = 0.20,
    protection_consensus_overlap: float = 0.50,
    protection_prior_mean: float = 0.20,
    protection_presence_margin: float = 0.02,
    protection_max_components: int = 1,
) -> torch.Tensor:
    from scipy import ndimage

    pred_np = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    cleaned_np = pred_np.copy()
    keep_largest = {int(x) for x in keep_largest_classes}
    fill_classes = {int(x) for x in fill_hole_classes}
    protected = {int(x) for x in (protected_classes or [])}
    use_protection = bool(protected) and reference_logits is not None
    candidate_prob_np = None
    reference_prob_np = None
    reference_pred_np = None
    consensus_pred_np = None
    prior_prob_np = None
    if use_protection:
        candidate_prob_np = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        reference_prob = torch.softmax(reference_logits.detach(), dim=1)
        reference_prob_np = reference_prob.cpu().numpy()
        reference_pred_np = reference_prob.argmax(dim=1).cpu().numpy().astype(np.int64)
        if consensus_logits is not None:
            consensus_pred_np = consensus_logits.detach().argmax(dim=1).cpu().numpy().astype(np.int64)
        if prior_prob is not None:
            prior_tensor = prior_prob.detach().to(device=logits.device, dtype=logits.dtype)
            if prior_tensor.shape[0] == 1 and logits.shape[0] > 1:
                prior_tensor = prior_tensor.repeat(logits.shape[0], 1, 1, 1)
            prior_prob_np = prior_tensor.cpu().numpy()
    class_count = int(logits.shape[1])
    for bi in range(pred_np.shape[0]):
        original = pred_np[bi]
        out = original.copy()
        h, w = out.shape[-2:]
        min_area = max(int(min_area_abs), int(round(float(min_area_ratio) * float(h * w))))
        min_area = max(1, min_area)
        for cls in range(1, class_count):
            mask = out == cls
            if cls in fill_classes and mask.any():
                filled = ndimage.binary_fill_holes(mask)
                out[filled & (out == 0)] = cls
                mask = out == cls
            if not mask.any():
                continue
            labels, num = ndimage.label(mask)
            if num <= 0:
                continue
            counts = np.bincount(labels.reshape(-1), minlength=num + 1)
            keep = np.zeros(num + 1, dtype=bool)
            if cls in keep_largest:
                if len(counts) > 1:
                    keep[int(np.argmax(counts[1:]) + 1)] = True
            else:
                keep[counts >= min_area] = True
                keep[0] = False
            if use_protection and cls in protected and cls not in keep_largest:
                assert candidate_prob_np is not None
                assert reference_prob_np is not None
                assert reference_pred_np is not None
                ref_cls = reference_prob_np[bi, cls]
                ref_other = np.max(
                    np.concatenate(
                        [reference_prob_np[bi, :cls], reference_prob_np[bi, cls + 1 :]],
                        axis=0,
                    ),
                    axis=0,
                )
                presence_margin = float(np.max(np.maximum(ref_cls - ref_other, 0.0)))
                eligible: List[Tuple[float, int]] = []
                if presence_margin >= float(protection_presence_margin):
                    for component_id in range(1, num + 1):
                        area = int(counts[component_id])
                        if keep[component_id] or area < max(1, int(protection_min_area_abs)):
                            continue
                        component = labels == component_id
                        candidate_conf = float(candidate_prob_np[bi, cls][component].mean())
                        reference_overlap = float((reference_pred_np[bi][component] == cls).mean())
                        consensus_overlap = (
                            float((consensus_pred_np[bi][component] == cls).mean())
                            if consensus_pred_np is not None
                            else 1.0
                        )
                        component_prior = (
                            float(prior_prob_np[bi, cls][component].mean())
                            if prior_prob_np is not None
                            else 0.0
                        )
                        supported = (
                            consensus_overlap >= float(protection_consensus_overlap)
                            and (
                                reference_overlap >= float(protection_reference_overlap)
                                or component_prior >= float(protection_prior_mean)
                            )
                        )
                        if candidate_conf < float(protection_candidate_confidence) or not supported:
                            continue
                        support_score = (
                            max(reference_overlap, component_prior)
                            + 0.50 * consensus_overlap
                            + 0.25 * candidate_conf
                        )
                        eligible.append((support_score, component_id))
                eligible.sort(reverse=True)
                for _, component_id in eligible[: max(0, int(protection_max_components))]:
                    keep[component_id] = True
            remove = (labels > 0) & (~keep[labels])
            out[remove] = 0
        cleaned_np[bi] = out

    cleaned = torch.from_numpy(cleaned_np).to(device=logits.device, dtype=torch.long)
    original = logits.argmax(dim=1)
    changed = cleaned != original
    if not bool(changed.any().detach().cpu().item()):
        return logits
    out_logits = logits.clone()
    bias = float(logit_bias)
    for cls in range(class_count):
        channel = out_logits[:, cls]
        channel = torch.where(changed & (cleaned == cls), channel + bias, channel)
        channel = torch.where(changed & (original == cls), channel - bias, channel)
        out_logits[:, cls] = channel
    return out_logits


_SLICE_NAME_RE = re.compile(r"^(?P<case>.+)_z(?P<slice>-?\d+)(?:\.nii(?:\.gz)?)?$")


def load_source_rv_atlas(path: str) -> Optional[Dict[str, object]]:
    atlas_path = Path(str(path or "").strip())
    if not str(path or "").strip():
        return None
    if not atlas_path.is_file():
        raise FileNotFoundError(f"RV atlas not found: {atlas_path}")
    with np.load(atlas_path, allow_pickle=False) as payload:
        required = {"occupancy", "presence", "area_ratio_q90", "metadata_json"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"RV atlas {atlas_path} is missing arrays: {sorted(missing)}")
        occupancy = np.asarray(payload["occupancy"], dtype=np.float32)
        presence = np.asarray(payload["presence"], dtype=np.float32)
        area_ratio_q90 = np.asarray(payload["area_ratio_q90"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if occupancy.ndim != 3 or occupancy.shape[0] != presence.shape[0]:
        raise ValueError(f"Invalid RV atlas occupancy shape: {occupancy.shape}")
    if area_ratio_q90.shape != presence.shape:
        raise ValueError("RV atlas area_ratio_q90 and presence shapes do not match")
    if str(metadata.get("source_domain")) != "A":
        raise ValueError("RV atlas must be fitted exclusively on Source-A")
    return {
        "path": str(atlas_path.resolve()),
        "occupancy": torch.from_numpy(occupancy),
        "presence": torch.from_numpy(presence),
        "area_ratio_q90": torch.from_numpy(area_ratio_q90),
        "extent": float(metadata["extent"]),
        "z_bins": int(metadata["z_bins"]),
        "metadata": metadata,
    }


def dataset_slice_position_map(dataset) -> Dict[str, float]:
    """Map each image basename to its normalized within-volume z position."""
    raw_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    csv_items = getattr(raw_dataset, "csv_items", None)
    if csv_items is None or len(csv_items.columns) == 0:
        return {}
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for value in csv_items.iloc[:, 0].tolist():
        name = os.path.basename(str(value))
        match = _SLICE_NAME_RE.match(name)
        if match is None:
            continue
        grouped.setdefault(match.group("case"), []).append((int(match.group("slice")), name))
    positions: Dict[str, float] = {}
    for items in grouped.values():
        z_values = [item[0] for item in items]
        z_min, z_max = min(z_values), max(z_values)
        z_span = max(1, z_max - z_min)
        for z_index, name in items:
            positions[name] = float(z_index - z_min) / float(z_span)
    return positions


def volume_continuity_logits(
    logits: torch.Tensor,
    names: Iterable[str],
    *,
    blend: float,
) -> torch.Tensor:
    """Blend predictions only across adjacent slices from the same volume."""
    strength = float(blend)
    if strength <= 0.0 or int(logits.shape[0]) <= 1:
        return logits
    strength = min(strength, 0.95)
    parsed: List[Optional[Tuple[str, int]]] = []
    for name in names:
        match = _SLICE_NAME_RE.match(os.path.basename(str(name)))
        parsed.append(None if match is None else (match.group("case"), int(match.group("slice"))))
    if len(parsed) != int(logits.shape[0]):
        return logits

    index = {item: i for i, item in enumerate(parsed) if item is not None}
    prob = torch.softmax(logits, dim=1)
    smoothed = prob.clone()
    changed = False
    for i, item in enumerate(parsed):
        if item is None:
            continue
        case_id, slice_idx = item
        neighbor_indices = [
            index[key]
            for key in ((case_id, slice_idx - 1), (case_id, slice_idx + 1))
            if key in index
        ]
        if not neighbor_indices:
            continue
        neighbor_prob = prob[neighbor_indices].mean(dim=0)
        smoothed[i] = (1.0 - strength) * prob[i] + strength * neighbor_prob
        changed = True
    if not changed:
        return logits
    return smoothed.clamp_min(1e-7).log()


def _normalize01_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return ((x - mn) / (mx - mn).clamp_min(eps)).clamp(0.0, 1.0)


def _gaussian_blur_2d_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        return x
    k = int(max(3, round(6.0 * sigma)))
    if k % 2 == 0:
        k += 1
    half = k // 2
    t = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-(t * t) / (2.0 * sigma * sigma))
    g = g / g.sum().clamp_min(1e-8)
    c = int(x.shape[1])
    g1 = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    g2 = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(x, g1, padding=(0, half), groups=c)
    x = F.conv2d(x, g2, padding=(half, 0), groups=c)
    return x


def _soft_centroid_xy(weight: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return x/y centroid for a [H,W] nonnegative map."""
    mass = weight.sum()
    if float(mass.detach().cpu().item()) <= 1e-7:
        return None
    h, w = weight.shape[-2:]
    yy = torch.arange(h, device=weight.device, dtype=weight.dtype).view(h, 1)
    xx = torch.arange(w, device=weight.device, dtype=weight.dtype).view(1, w)
    cy = (weight * yy).sum() / mass.clamp_min(1e-7)
    cx = (weight * xx).sum() / mass.clamp_min(1e-7)
    return cx, cy


def _argmax_xy(weight: torch.Tensor) -> Tuple[float, float]:
    h, w = weight.shape[-2:]
    idx = int(weight.reshape(-1).argmax().detach().cpu().item())
    return float(idx % w), float(idx // w)


def _saliency_confidence(sal_b1hw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s = sal_b1hw
    if s.dim() == 3:
        s = s.unsqueeze(1)
    s = s.clamp(0.0, 1.0).to(dtype=torch.float32)
    b, _, h, w = s.shape
    p = s / (s.sum(dim=(-2, -1), keepdim=True) + float(eps))
    ent = -(p * (p + float(eps)).log()).sum(dim=(-2, -1)).view(b) / float(np.log(max(2, h * w)))
    peak_ent = (1.0 - ent).clamp(0.0, 1.0)
    pmax = p.amax(dim=(-2, -1)).view(b)
    sharp = ((pmax * float(h * w)) - 1.0) / max(1.0, float(h * w - 1))
    return (0.5 * peak_ent + 0.5 * sharp.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _gaussian_from_centers(
    centers: List[Tuple[torch.Tensor, torch.Tensor]],
    h: int,
    w: int,
    sy: float,
    sx: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
    maps = []
    for cx, cy in centers:
        cx = cx.to(device=device, dtype=dtype).view(1, 1, 1)
        cy = cy.to(device=device, dtype=dtype).view(1, 1, 1)
        blob = torch.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
        maps.append(blob)
    return _normalize01_torch(torch.cat(maps, dim=0).unsqueeze(1))


def _soft_dilate(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def _soft_erode(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return -_soft_dilate(-x, kernel_size)


def confidence_gated_soft_anatomy_refine(
    logits: torch.Tensor,
    prior_prob: Optional[torch.Tensor],
    *,
    myo_ring_strength: float,
    rv_adjacency_strength: float,
    ring_kernel: int,
    rv_kernel: int,
    gate_low: float,
    gate_high: float,
    max_delta: float,
) -> torch.Tensor:
    """Repair uncertain MYO/RV boundaries without changing confident regions."""
    myo_strength = max(0.0, min(1.0, float(myo_ring_strength)))
    rv_strength = max(0.0, min(1.0, float(rv_adjacency_strength)))
    if (myo_strength <= 0.0 and rv_strength <= 0.0) or int(logits.shape[1]) < 4:
        return logits

    prob = torch.softmax(logits, dim=1)
    if prior_prob is None or prior_prob.shape != prob.shape:
        structural_prob = prob
    else:
        prior = prior_prob.to(device=prob.device, dtype=prob.dtype)
        structural_prob = 0.65 * prob + 0.35 * prior

    confidence = prob.max(dim=1, keepdim=True)[0]
    uncertainty = 1.0 - confidence
    low = max(0.0, min(1.0, float(gate_low)))
    high = max(low + 1e-6, min(1.0, float(gate_high)))
    gate = ((uncertainty - low) / (high - low)).clamp(0.0, 1.0)
    delta_limit = max(0.0, min(1.0, float(max_delta)))

    updated = prob.clone()
    lv = structural_prob[:, 1:2]
    myo = updated[:, 2:3]
    rv = updated[:, 3:4]

    if myo_strength > 0.0:
        lv_dilated = _soft_dilate(lv, ring_kernel)
        lv_eroded = _soft_erode(lv, max(3, int(ring_kernel) // 2))
        ring = _normalize01_torch((lv_dilated - lv_eroded).clamp_min(0.0))
        ring = ring * (1.0 - 0.75 * lv).clamp(0.0, 1.0)
        lv_present = (lv.amax(dim=(-2, -1), keepdim=True) >= 0.25).to(prob.dtype)
        myo_target = torch.maximum(myo, ring * lv_present)
        myo_delta = (myo_target - myo).clamp(0.0, delta_limit)
        updated[:, 2:3] = myo + myo_strength * gate * myo_delta

    if rv_strength > 0.0:
        myo_support = structural_prob[:, 2:3]
        cardiac_support = _soft_dilate((lv + myo_support).clamp(0.0, 1.0), rv_kernel)
        rv_seed = torch.maximum(rv, 0.70 * structural_prob[:, 3:4])
        rv_reach = _soft_dilate(rv_seed, max(3, int(rv_kernel) // 2))
        available = (1.0 - 0.80 * lv).clamp(0.0, 1.0) * (1.0 - 0.50 * myo_support).clamp(0.0, 1.0)
        rv_present = (rv_seed.amax(dim=(-2, -1), keepdim=True) >= 0.20).to(prob.dtype)
        rv_target = torch.maximum(rv, rv_reach * cardiac_support * available * rv_present)
        rv_delta = (rv_target - rv).clamp(0.0, delta_limit)
        updated[:, 3:4] = rv + rv_strength * gate * rv_delta

    updated = updated / updated.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.log(updated.clamp_min(1e-6))


@torch.no_grad()
def rv_image_conditioned_candidate(
    volume: torch.Tensor,
    base_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    *,
    probability_cap: float,
    adjacency_kernel: int,
    trigger_ratio: float,
    max_area_ratio: float,
    background_margin: float,
    min_lv_pixels: int,
    lv_confidence: float,
    myo_confidence: float,
    intensity_sigma_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build a source-mask-independent RV recovery candidate inside APL.

    The frozen final prediction remains the base. Only background probability
    is transferred to RV, and only where three target-time cues agree:

    * reliable adapted/refined LV/MYO anatomy supplies an exterior search band;
    * the image resembles the current slice's confident LV blood-pool intensity;
    * background and RV probabilities remain sufficiently uncertain.

    The per-sample area gate limits this candidate to missing/tiny-RV cases and
    caps its possible support relative to LV. No target labels, source RV mask,
    temporal memory, or persistent parameter update is used.
    """
    if int(base_logits.shape[1]) < 4 or float(probability_cap) <= 0.0:
        return base_logits, {
            "trigger_fraction": 0.0,
            "support_fraction": 0.0,
            "changed_fraction": 0.0,
            "mean_probability_delta": 0.0,
        }

    base_prob = torch.softmax(base_logits.detach(), dim=1)
    refined_prob = torch.softmax(refined_logits.detach(), dim=1)
    adapted_prob = torch.softmax(adapted_logits.detach(), dim=1)
    evidence_prob = 0.5 * (refined_prob + adapted_prob)
    evidence_pred = evidence_prob.argmax(dim=1, keepdim=True)

    lv_threshold = max(0.0, min(1.0, float(lv_confidence)))
    myo_threshold = max(0.0, min(1.0, float(myo_confidence)))
    lv_mask = (evidence_pred == 1) & (evidence_prob[:, 1:2] >= lv_threshold)
    myo_mask = (evidence_pred == 2) & (evidence_prob[:, 2:3] >= myo_threshold)
    reliable_anatomy = lv_mask | myo_mask
    search_band = _soft_dilate(reliable_anatomy.to(base_prob.dtype), adjacency_kernel)
    search_band = (search_band > 0.5).to(base_prob.dtype)
    search_band = search_band * (~reliable_anatomy).to(base_prob.dtype)

    if volume.dim() != 4:
        raise ValueError(f"Expected BCHW image tensor for RV image candidate, got shape={tuple(volume.shape)}")
    image = volume.detach().to(dtype=base_prob.dtype)
    if int(image.shape[1]) != 1:
        image = image.mean(dim=1, keepdim=True)

    lv_weight = evidence_prob[:, 1:2] * lv_mask.to(base_prob.dtype)
    lv_weight_sum = lv_weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    lv_mean = (image * lv_weight).sum(dim=(-2, -1), keepdim=True) / lv_weight_sum
    lv_var = ((image - lv_mean) ** 2 * lv_weight).sum(dim=(-2, -1), keepdim=True) / lv_weight_sum
    sigma_scale = max(0.25, float(intensity_sigma_scale))
    lv_sigma = lv_var.clamp_min(0.0025).sqrt() * sigma_scale
    intensity_score = torch.exp(-0.5 * ((image - lv_mean) / lv_sigma.clamp_min(0.05)) ** 2)

    margin = max(1e-3, min(1.0, float(background_margin)))
    bg_rv_gap = base_prob[:, 0:1] - base_prob[:, 3:4]
    uncertainty_score = ((margin - bg_rv_gap) / margin).clamp(0.0, 1.0)
    rv_hint = torch.sqrt(
        (refined_prob[:, 3:4] * adapted_prob[:, 3:4]).clamp_min(0.0)
    )
    rv_hint = (rv_hint / 0.25).clamp(0.0, 1.0)
    support = search_band * intensity_score * uncertainty_score * (0.25 + 0.75 * rv_hint)

    lv_pixels = lv_mask.sum(dim=(-2, -1)).to(dtype=base_prob.dtype)
    base_pred = base_prob.argmax(dim=1, keepdim=True)
    rv_pixels = (base_pred == 3).sum(dim=(-2, -1)).to(dtype=base_prob.dtype)
    min_lv = max(1, int(min_lv_pixels))
    trigger_limit = max(0.0, float(trigger_ratio))
    area_limit = max(trigger_limit, float(max_area_ratio))
    trigger = (lv_pixels >= float(min_lv)) & (rv_pixels <= trigger_limit * lv_pixels)

    gated_support = torch.zeros_like(support)
    for bi in range(int(support.shape[0])):
        if not bool(trigger[bi, 0].item()):
            continue
        max_extra = max(0, int(math.floor(area_limit * float(lv_pixels[bi, 0].item()))) - int(rv_pixels[bi, 0].item()))
        if max_extra <= 0:
            continue
        flat = support[bi, 0].reshape(-1)
        positive = flat > 1e-6
        positive_count = int(positive.sum().item())
        if positive_count <= max_extra:
            gated_support[bi, 0] = support[bi, 0]
            continue
        max_extra = min(max_extra, int(flat.numel()))
        top_values, top_indices = torch.topk(flat, k=max_extra, largest=True, sorted=False)
        kept = torch.zeros_like(flat)
        kept[top_indices] = top_values
        gated_support[bi, 0] = kept.view_as(support[bi, 0])

    cap = max(0.0, min(0.49, float(probability_cap)))
    delta = cap * gated_support
    delta = torch.minimum(delta, 0.75 * base_prob[:, 0:1])
    updated = base_prob.clone()
    updated[:, 0:1] = base_prob[:, 0:1] - delta
    updated[:, 3:4] = base_prob[:, 3:4] + delta
    updated = updated / updated.sum(dim=1, keepdim=True).clamp_min(1e-6)

    spatial_size = float(max(1, int(support.shape[0] * support.shape[-2] * support.shape[-1])))
    stats = {
        "trigger_fraction": float(trigger.to(base_prob.dtype).mean().cpu().item()),
        "support_fraction": float((gated_support > 1e-6).to(base_prob.dtype).sum().cpu().item() / spatial_size),
        "changed_fraction": float((delta > 1e-6).to(base_prob.dtype).sum().cpu().item() / spatial_size),
        "mean_probability_delta": float(delta.sum().cpu().item() / spatial_size),
    }
    return torch.log(updated.clamp_min(1e-6)), stats


@torch.no_grad()
def rv_atlas_guided_candidate(
    names: Iterable[str],
    slice_positions: Dict[str, float],
    atlas: Dict[str, object],
    base_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    *,
    probability_cap: float,
    support_threshold: float,
    max_area_scale: float,
    background_margin: float,
    min_lv_pixels: int,
    lv_confidence: float,
    myo_confidence: float,
    competitor_margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Add a Source-A atlas-aligned RV candidate without changing model state.

    Source-A labels are represented only by a frozen canonical occupancy map.
    At target time the map is aligned using predicted LV/MYO geometry and the
    unlabeled within-volume slice position. It can transfer probability from
    background to RV only where the base output is uncertain and atlas support
    is sufficiently strong. Existing RV pixels and LV/MYO channels are kept.
    """
    zero_stats = {
        "trigger_fraction": 0.0,
        "support_fraction": 0.0,
        "changed_fraction": 0.0,
        "hard_changed_fraction": 0.0,
        "mean_probability_delta": 0.0,
        "mean_atlas_presence": 0.0,
    }
    batch, classes, height, width = base_logits.shape
    if classes < 4 or batch == 0 or float(probability_cap) <= 0.0:
        return base_logits, zero_stats

    occupancy = atlas["occupancy"]
    presence = atlas["presence"]
    area_ratio_q90 = atlas["area_ratio_q90"]
    if not isinstance(occupancy, torch.Tensor):
        raise TypeError("RV atlas occupancy must be a torch.Tensor")
    occupancy = occupancy.to(device=base_logits.device, dtype=base_logits.dtype)
    presence = presence.to(device=base_logits.device, dtype=base_logits.dtype)
    area_ratio_q90 = area_ratio_q90.to(device=base_logits.device, dtype=base_logits.dtype)
    z_bins = int(atlas["z_bins"])
    extent = max(1e-3, float(atlas["extent"]))

    base_prob = torch.softmax(base_logits.detach(), dim=1)
    refined_prob = torch.softmax(refined_logits.detach(), dim=1)
    adapted_prob = torch.softmax(adapted_logits.detach(), dim=1)
    evidence_prob = 0.5 * (refined_prob + adapted_prob)
    evidence_pred = evidence_prob.argmax(dim=1, keepdim=True)
    lv_threshold = max(0.0, min(1.0, float(lv_confidence)))
    myo_threshold = max(0.0, min(1.0, float(myo_confidence)))
    lv_mask = (evidence_pred == 1) & (evidence_prob[:, 1:2] >= lv_threshold)
    myo_mask = (evidence_pred == 2) & (evidence_prob[:, 2:3] >= myo_threshold)
    reference_mask = lv_mask | myo_mask
    base_pred = base_prob.argmax(dim=1, keepdim=True)
    base_rv_mask = base_pred == 3

    yy = torch.arange(height, device=base_prob.device, dtype=base_prob.dtype).view(height, 1)
    xx = torch.arange(width, device=base_prob.device, dtype=base_prob.dtype).view(1, width)
    support_maps = torch.zeros_like(base_prob[:, 0:1])
    delta_maps = torch.zeros_like(base_prob[:, 0:1])
    triggered = 0
    presence_total = 0.0
    names_list = list(names)
    threshold = max(0.0, min(0.99, float(support_threshold)))
    probability_cap = max(1e-4, min(0.49, float(probability_cap)))
    bg_margin = max(1e-4, min(1.0, float(background_margin)))
    area_scale = max(0.1, float(max_area_scale))
    win_margin = max(0.0, float(competitor_margin))

    for bi in range(batch):
        lv_pixels = int(lv_mask[bi, 0].sum().item())
        reference_pixels = int(reference_mask[bi, 0].sum().item())
        if lv_pixels < max(1, int(min_lv_pixels)) or reference_pixels < max(1, int(min_lv_pixels)):
            continue
        center = _soft_centroid_xy(
            evidence_prob[bi, 1] * lv_mask[bi, 0].to(base_prob.dtype)
        )
        if center is None:
            continue
        center_x, center_y = center
        scale = math.sqrt(float(reference_pixels) / math.pi)
        if not math.isfinite(scale) or scale < 2.0:
            continue

        name = os.path.basename(str(names_list[bi])) if bi < len(names_list) else ""
        z_position = max(0.0, min(1.0, float(slice_positions.get(name, 0.5))))
        z_bin = min(z_bins - 1, int(math.floor(z_position * z_bins)))
        presence_value = float(presence[z_bin].item())
        presence_total += presence_value

        grid_x = (xx - center_x) / (float(scale) * extent)
        grid_y = (yy - center_y) / (float(scale) * extent)
        grid = torch.stack(
            [grid_x.expand(height, width), grid_y.expand(height, width)], dim=-1
        ).unsqueeze(0)
        atlas_map = F.grid_sample(
            occupancy[z_bin : z_bin + 1].unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        atlas_support = ((atlas_map - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
        atlas_support = atlas_support * presence_value
        atlas_support = atlas_support * (~reference_mask[bi, 0]).to(base_prob.dtype)

        bg = base_prob[bi, 0]
        rv = base_prob[bi, 3]
        bg_rv_gap = bg - rv
        uncertainty = ((bg_margin - bg_rv_gap.abs()) / bg_margin).clamp(0.0, 1.0)
        support = atlas_support * uncertainty

        current_rv_pixels = int(base_rv_mask[bi, 0].sum().item())
        max_rv_pixels = int(
            math.ceil(float(area_ratio_q90[z_bin].item()) * float(reference_pixels) * area_scale)
        )
        max_extra = max(0, max_rv_pixels - current_rv_pixels)
        if max_extra <= 0:
            continue

        rival = torch.maximum(bg, base_prob[bi, 1:3].amax(dim=0))
        required_delta = (rival - rv + win_margin).clamp_min(0.0)
        # BG->RV transfer closes a BG-vs-RV gap from both sides.
        required_delta = torch.where(bg >= base_prob[bi, 1:3].amax(dim=0), 0.5 * required_delta, required_delta)
        available_delta = torch.minimum(probability_cap * support, 0.75 * bg)
        eligible = (support > 0.0) & (required_delta <= available_delta) & (~base_rv_mask[bi, 0])
        if not bool(eligible.any().item()):
            continue

        priority = support * eligible.to(base_prob.dtype)
        eligible_count = int(eligible.sum().item())
        keep_count = min(max_extra, eligible_count)
        if keep_count <= 0:
            continue
        if keep_count < eligible_count:
            _, indices = torch.topk(priority.reshape(-1), k=keep_count, largest=True, sorted=False)
            keep = torch.zeros_like(priority.reshape(-1), dtype=torch.bool)
            keep[indices] = True
            keep = keep.view_as(priority)
        else:
            keep = eligible

        delta = torch.where(
            keep,
            torch.maximum(required_delta + win_margin, 0.5 * available_delta),
            torch.zeros_like(available_delta),
        )
        delta = torch.minimum(delta, available_delta)
        if bool((delta > 0.0).any().item()):
            triggered += 1
            support_maps[bi, 0] = support
            delta_maps[bi, 0] = delta

    updated = base_prob.clone()
    updated[:, 0:1] = (base_prob[:, 0:1] - delta_maps).clamp_min(1e-7)
    updated[:, 3:4] = base_prob[:, 3:4] + delta_maps
    updated = updated / updated.sum(dim=1, keepdim=True).clamp_min(1e-6)
    updated_pred = updated.argmax(dim=1, keepdim=True)
    spatial_size = float(max(1, batch * height * width))
    stats = {
        "trigger_fraction": float(triggered) / float(max(1, batch)),
        "support_fraction": float((support_maps > 0.0).sum().item()) / spatial_size,
        "changed_fraction": float((delta_maps > 0.0).sum().item()) / spatial_size,
        "hard_changed_fraction": float((updated_pred != base_pred).sum().item()) / spatial_size,
        "mean_probability_delta": float(delta_maps.sum().item()) / spatial_size,
        "mean_atlas_presence": presence_total / float(max(1, batch)),
    }
    return torch.log(updated.clamp_min(1e-7)), stats


@torch.no_grad()
def rv_directional_slice_candidate(
    names: Iterable[str],
    base_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    *,
    probability_cap: float,
    max_slice_gap: int,
    require_two_sided: bool,
    seed_confidence: float,
    min_seed_pixels: int,
    min_lv_pixels: int,
    trigger_neighbor_ratio: float,
    max_area_scale: float,
    support_kernel: int,
    direction_sigma_scale: float,
    direction_consistency: float,
    background_margin: float,
    competitor_ratio: float,
    target_rv_hint: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Recover internal RV dropouts using only reliable adjacent target slices.

    This candidate has no persistent history. It groups slices already present
    in the current mini-batch, estimates LV->RV direction from confident
    P_adapt/P_ref RV seeds in the same case, and only proposes BG->RV changes on
    a tiny-RV slice when neighboring slice geometry agrees. The change must be
    strong enough to make RV, rather than LV/MYO, the foreground winner.
    """
    zero_stats = {
        "neighbor_seed_fraction": 0.0,
        "trigger_fraction": 0.0,
        "mean_neighbor_count": 0.0,
        "support_fraction": 0.0,
        "changed_fraction": 0.0,
        "mean_probability_delta": 0.0,
    }
    batch = int(base_logits.shape[0])
    if int(base_logits.shape[1]) < 4 or batch <= 1 or float(probability_cap) <= 0.0:
        return base_logits, zero_stats

    parsed: List[Optional[Tuple[str, int]]] = []
    for name in names:
        match = _SLICE_NAME_RE.match(os.path.basename(str(name)))
        parsed.append(None if match is None else (match.group("case"), int(match.group("slice"))))
    if len(parsed) != batch:
        return base_logits, zero_stats

    base_prob = torch.softmax(base_logits.detach(), dim=1)
    refined_prob = torch.softmax(refined_logits.detach(), dim=1)
    adapted_prob = torch.softmax(adapted_logits.detach(), dim=1)
    evidence_prob = 0.5 * (refined_prob + adapted_prob)
    evidence_pred = evidence_prob.argmax(dim=1, keepdim=True)
    seed_threshold = max(0.0, min(1.0, float(seed_confidence)))
    lv_threshold = min(0.95, max(0.50, seed_threshold))
    lv_mask = (evidence_pred == 1) & (evidence_prob[:, 1:2] >= lv_threshold)
    myo_mask = (evidence_pred == 2) & (evidence_prob[:, 2:3] >= 0.45)
    rv_seed_mask = (evidence_pred == 3) & (evidence_prob[:, 3:4] >= seed_threshold)
    lv_pixels = lv_mask.flatten(1).sum(dim=1)
    rv_seed_pixels = rv_seed_mask.flatten(1).sum(dim=1)
    seed_valid = (lv_pixels >= max(1, int(min_lv_pixels))) & (
        rv_seed_pixels >= max(1, int(min_seed_pixels))
    )
    base_pred = base_prob.argmax(dim=1, keepdim=True)
    base_rv_pixels = (base_pred == 3).flatten(1).sum(dim=1)

    h, w = base_prob.shape[-2:]
    yy = torch.arange(h, device=base_prob.device, dtype=base_prob.dtype).view(h, 1)
    xx = torch.arange(w, device=base_prob.device, dtype=base_prob.dtype).view(1, w)
    max_gap = max(1, int(max_slice_gap))
    min_consistency = max(0.0, min(1.0, float(direction_consistency)))
    trigger_ratio = max(0.0, float(trigger_neighbor_ratio))
    area_scale = max(0.1, float(max_area_scale))
    sigma_scale = max(0.25, float(direction_sigma_scale))
    margin = max(1e-3, min(1.0, float(background_margin)))
    rival_ratio = max(0.0, min(1.0, float(competitor_ratio)))
    hint_floor = max(0.0, min(1.0, float(target_rv_hint)))

    gated_support = torch.zeros_like(base_prob[:, 0:1])
    neighbor_seeded = 0
    triggered = 0
    neighbor_count_sum = 0.0
    for bi, item in enumerate(parsed):
        if item is None or int(lv_pixels[bi].item()) < max(1, int(min_lv_pixels)):
            continue
        case_id, slice_idx = item
        neighbors: List[Tuple[int, int]] = []
        for nj, neighbor_item in enumerate(parsed):
            if neighbor_item is None or nj == bi or not bool(seed_valid[nj].item()):
                continue
            neighbor_case, neighbor_slice = neighbor_item
            gap = int(neighbor_slice) - int(slice_idx)
            if neighbor_case == case_id and 0 < abs(gap) <= max_gap:
                neighbors.append((nj, gap))
        if not neighbors:
            continue
        if require_two_sided and not (
            any(gap < 0 for _, gap in neighbors) and any(gap > 0 for _, gap in neighbors)
        ):
            continue
        neighbor_seeded += 1

        target_lv_center = _soft_centroid_xy(
            evidence_prob[bi, 1] * lv_mask[bi, 0].to(base_prob.dtype)
        )
        if target_lv_center is None:
            continue
        vectors: List[torch.Tensor] = []
        rv_maps: List[torch.Tensor] = []
        rv_areas: List[float] = []
        weights: List[float] = []
        for nj, gap in neighbors:
            neighbor_lv_center = _soft_centroid_xy(
                evidence_prob[nj, 1] * lv_mask[nj, 0].to(base_prob.dtype)
            )
            neighbor_rv_center = _soft_centroid_xy(
                evidence_prob[nj, 3] * rv_seed_mask[nj, 0].to(base_prob.dtype)
            )
            if neighbor_lv_center is None or neighbor_rv_center is None:
                continue
            dx = neighbor_rv_center[0] - neighbor_lv_center[0]
            dy = neighbor_rv_center[1] - neighbor_lv_center[1]
            vector = torch.stack((dx, dy))
            if float(torch.norm(vector).item()) < 2.0:
                continue
            vectors.append(vector)
            rv_maps.append(rv_seed_mask[nj, 0].to(base_prob.dtype))
            rv_areas.append(float(rv_seed_pixels[nj].item()))
            weights.append(1.0 / float(abs(gap)))
        if not vectors:
            continue

        weight_tensor = base_prob.new_tensor(weights)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
        stacked_vectors = torch.stack(vectors, dim=0)
        vector_norms = torch.norm(stacked_vectors, dim=1, keepdim=True).clamp_min(1e-6)
        unit_vectors = stacked_vectors / vector_norms
        mean_unit = (unit_vectors * weight_tensor.view(-1, 1)).sum(dim=0)
        consistency = torch.norm(mean_unit)
        if float(consistency.item()) < min_consistency:
            continue
        direction = mean_unit / consistency.clamp_min(1e-6)
        mean_vector = (stacked_vectors * weight_tensor.view(-1, 1)).sum(dim=0)
        mean_neighbor_area = float(sum(a * b for a, b in zip(rv_areas, weights)) / max(1e-6, sum(weights)))
        if float(base_rv_pixels[bi].item()) > trigger_ratio * mean_neighbor_area:
            continue
        triggered += 1
        neighbor_count_sum += float(len(vectors))

        neighbor_map = torch.stack(rv_maps, dim=0)
        neighbor_map = (neighbor_map * weight_tensor.view(-1, 1, 1)).sum(dim=0)
        neighbor_map = _soft_dilate(neighbor_map.view(1, 1, h, w), support_kernel)[0, 0]
        target_cx, target_cy = target_lv_center
        projected_cx = target_cx + mean_vector[0]
        projected_cy = target_cy + mean_vector[1]
        lv_radius = math.sqrt(max(1.0, float(lv_pixels[bi].item())) / math.pi)
        sigma = max(3.0, sigma_scale * lv_radius)
        direction_score = torch.exp(
            -0.5 * (((xx - projected_cx) ** 2 + (yy - projected_cy) ** 2) / (sigma * sigma))
        )
        projection = (xx - target_cx) * direction[0] + (yy - target_cy) * direction[1]
        side_gate = (projection >= 0.0).to(base_prob.dtype)
        bg_rv_gap = base_prob[bi, 0] - base_prob[bi, 3]
        uncertainty = ((margin - bg_rv_gap) / margin).clamp(0.0, 1.0)
        other_fg = torch.maximum(base_prob[bi, 1], base_prob[bi, 2])
        rv_competitive = (base_prob[bi, 3] >= rival_ratio * other_fg).to(base_prob.dtype)
        target_hint_score = ((evidence_prob[bi, 3] - hint_floor) / max(1e-3, 0.25 - hint_floor)).clamp(0.0, 1.0)
        anatomy_exclusion = (~(lv_mask[bi, 0] | myo_mask[bi, 0])).to(base_prob.dtype)
        bg_gate = (base_pred[bi, 0] == 0).to(base_prob.dtype)
        support = (
            neighbor_map
            * direction_score
            * side_gate
            * uncertainty
            * rv_competitive
            * target_hint_score
            * anatomy_exclusion
            * bg_gate
        )
        max_total = int(math.floor(area_scale * mean_neighbor_area))
        max_extra = max(0, max_total - int(base_rv_pixels[bi].item()))
        if max_extra <= 0:
            continue
        flat = support.reshape(-1)
        positive_count = int((flat > 1e-6).sum().item())
        if positive_count <= max_extra:
            gated_support[bi, 0] = support
        else:
            k = min(max_extra, int(flat.numel()))
            top_values, top_indices = torch.topk(flat, k=k, largest=True, sorted=False)
            kept = torch.zeros_like(flat)
            kept[top_indices] = top_values
            gated_support[bi, 0] = kept.view(h, w)

    cap = max(0.0, min(0.49, float(probability_cap)))
    delta = cap * gated_support
    delta = torch.minimum(delta, 0.75 * base_prob[:, 0:1])
    other_fg = torch.maximum(base_prob[:, 1:2], base_prob[:, 2:3])
    becomes_rv = (base_prob[:, 3:4] + delta >= other_fg).to(base_prob.dtype)
    delta = delta * becomes_rv
    spatial_size = float(max(1, batch * h * w))
    stats = {
        "neighbor_seed_fraction": float(neighbor_seeded / max(1, batch)),
        "trigger_fraction": float(triggered / max(1, batch)),
        "mean_neighbor_count": float(neighbor_count_sum / max(1, triggered)),
        "support_fraction": float((gated_support > 1e-6).sum().item() / spatial_size),
        "changed_fraction": float((delta > 1e-6).sum().item() / spatial_size),
        "mean_probability_delta": float(delta.sum().item() / spatial_size),
    }
    if not bool((delta > 1e-6).any().item()):
        return base_logits, stats
    updated = base_prob.clone()
    updated[:, 0:1] = base_prob[:, 0:1] - delta
    updated[:, 3:4] = base_prob[:, 3:4] + delta
    updated = updated / updated.sum(dim=1, keepdim=True).clamp_min(1e-6)

    return torch.log(updated.clamp_min(1e-6)), stats


def _interval_score(value: torch.Tensor, low: float, high: float, eps: float = 1e-6) -> torch.Tensor:
    low_t = value.new_tensor(float(low))
    high_t = value.new_tensor(float(high))
    below = (value / low_t.clamp_min(eps)).clamp(0.0, 1.0)
    above = (high_t / value.clamp_min(eps)).clamp(0.0, 1.0)
    inside = ((value >= low_t) & (value <= high_t)).to(value.dtype)
    return torch.where(inside > 0, value.new_tensor(1.0), torch.minimum(below, above))


def _centroid_distance(mask_a: torch.Tensor, mask_b: torch.Tensor) -> Optional[torch.Tensor]:
    ca = _soft_centroid_xy(mask_a)
    cb = _soft_centroid_xy(mask_b)
    if ca is None or cb is None:
        return None
    ax, ay = ca
    bx, by = cb
    h, w = mask_a.shape[-2:]
    return torch.sqrt(((ax - bx) / max(1.0, float(w))) ** 2 + ((ay - by) / max(1.0, float(h))) ** 2)


def cardiac_topology_score(logits: torch.Tensor, ring_kernel: int = 9) -> float:
    """Score coarse LV/MYO/RV topology; higher means more cardiac-like.

    Assumes the MMS/ACDC-style label order used by this repo:
    class 1 ~= LV blood pool, class 2 ~= myocardium, class 3 ~= RV.
    The score is deliberately soft and only used for unsupervised branch ranking.
    """
    pred = logits.argmax(dim=1)
    scores: List[torch.Tensor] = []
    for bi in range(pred.shape[0]):
        sample = pred[bi]
        h, w = sample.shape[-2:]
        hw = float(max(1, h * w))
        lv = (sample == 1).float()
        myo = (sample == 2).float()
        rv = (sample == 3).float()
        fg = (sample > 0).float()

        lv_area = lv.mean()
        myo_area = myo.mean()
        rv_area = rv.mean()
        fg_area = fg.mean()

        area_score = _interval_score(fg_area, 0.002, 0.35)
        if float(lv_area.detach().cpu().item()) > 1e-7 and float(myo_area.detach().cpu().item()) > 1e-7:
            myo_lv_ratio = myo_area / lv_area.clamp_min(1e-7)
            area_score = area_score + _interval_score(myo_lv_ratio, 0.12, 2.8)
        else:
            area_score = area_score + logits.new_tensor(0.35)
        if float(lv_area.detach().cpu().item()) > 1e-7 and float(rv_area.detach().cpu().item()) > 1e-7:
            rv_lv_ratio = rv_area / lv_area.clamp_min(1e-7)
            area_score = area_score + _interval_score(rv_lv_ratio, 0.08, 3.5)
        else:
            area_score = area_score + logits.new_tensor(0.55)
        area_score = area_score / 3.0

        dil_lv = _soft_dilate(lv.view(1, 1, h, w), ring_kernel)[0, 0]
        dil_myo = _soft_dilate(myo.view(1, 1, h, w), ring_kernel)[0, 0]
        if float(myo_area.detach().cpu().item()) > 1e-7:
            myo_near_lv = (myo * dil_lv).sum() / myo.sum().clamp_min(1.0)
        else:
            myo_near_lv = logits.new_tensor(0.5)
        if float(lv_area.detach().cpu().item()) > 1e-7:
            lv_touched_by_myo = (lv * dil_myo).sum() / lv.sum().clamp_min(1.0)
        else:
            lv_touched_by_myo = logits.new_tensor(0.5)
        ring_score = (0.55 * myo_near_lv + 0.45 * lv_touched_by_myo).clamp(0.0, 1.0)

        lv_myo_dist = _centroid_distance(lv, myo)
        if lv_myo_dist is None:
            center_score = logits.new_tensor(0.55)
        else:
            center_score = (1.0 - 3.0 * lv_myo_dist).clamp(0.0, 1.0)
        lv_rv_dist = _centroid_distance(lv, rv)
        if lv_rv_dist is not None:
            rv_sep = _interval_score(lv_rv_dist, 0.025, 0.45)
            center_score = 0.70 * center_score + 0.30 * rv_sep

        compact_terms = []
        for mask in (lv, myo, rv):
            if float(mask.sum().detach().cpu().item()) <= 0.0:
                continue
            ys, xs = torch.where(mask > 0)
            box_area = (ys.max() - ys.min() + 1).float() * (xs.max() - xs.min() + 1).float()
            occupancy = mask.sum() / box_area.clamp_min(1.0)
            compact_terms.append(_interval_score(occupancy, 0.08, 1.0))
        compact_score = torch.stack(compact_terms).mean() if compact_terms else logits.new_tensor(0.5)

        scores.append((0.30 * area_score + 0.35 * ring_score + 0.20 * center_score + 0.15 * compact_score).clamp(0.0, 1.0))
    return float(torch.stack(scores).mean().detach().cpu().item())


def build_cardiac_prior(
    source_logits: torch.Tensor,
    source_blend: float,
    smooth_sigma: float,
    generic_sigma: float,
    min_prior: float,
    topology_strength: float,
    topology_rv_strength: float,
    topology_ring_kernel: int,
    presence_preserving: bool = False,
    presence_classes: Optional[Iterable[int]] = None,
    presence_min_pixels: int = 4,
    presence_area_reference: int = 32,
    presence_confidence_low: float = 0.50,
    presence_confidence_high: float = 0.90,
    presence_floor: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Build a target-time cardiac spatial prior from source predictions.

    This is the cardiac analogue of the prostate CLIP/ellipse cue: it gives a
    coarse class-aware localization target without using labels. It is source-
    relative and deliberately smooth, so it should guide updates rather than
    force exact source masks.
    """
    source_prob = torch.softmax(source_logits.detach(), dim=1)
    b, c, h, w = source_prob.shape
    device, dtype = source_prob.device, source_prob.dtype
    fg_source = source_prob[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    source_fg_area = fg_source.mean(dim=(1, 2, 3))
    source_conf = source_prob.max(dim=1)[0].mean(dim=(1, 2))

    smooth_fg = _normalize01_torch(_gaussian_blur_2d_torch(fg_source, smooth_sigma))
    class_maps = []
    for cls in range(1, c):
        cls_prob = source_prob[:, cls : cls + 1]
        smooth_cls = _normalize01_torch(_gaussian_blur_2d_torch(cls_prob, smooth_sigma))
        centers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for bi in range(b):
            ctr = _soft_centroid_xy(cls_prob[bi, 0])
            if ctr is None:
                ctr = _soft_centroid_xy(fg_source[bi, 0])
            if ctr is None:
                ctr = (
                    torch.tensor(0.5 * float(w - 1), device=device, dtype=dtype),
                    torch.tensor(0.5 * float(h - 1), device=device, dtype=dtype),
                )
            centers.append(ctr)
        sigma_y = max(1.0, float(generic_sigma) * float(h))
        sigma_x = max(1.0, float(generic_sigma) * float(w))
        generic = _gaussian_from_centers(centers, h, w, sigma_y, sigma_x, device, dtype)
        blend = max(0.0, min(1.0, float(source_blend)))
        cls_prior = _normalize01_torch(blend * smooth_cls + (1.0 - blend) * generic)
        class_maps.append(cls_prior)

    topology_strength = max(0.0, min(1.0, float(topology_strength)))
    topology_rv_strength = max(0.0, min(1.0, float(topology_rv_strength)))
    if topology_strength > 0.0 and len(class_maps) >= 3:
        lv_prior, myo_prior, rv_prior = class_maps[:3]
        lv_dil = _soft_dilate(lv_prior, int(topology_ring_kernel))
        lv_ero = _soft_erode(lv_prior, max(3, int(topology_ring_kernel) // 2))
        myo_ring = _normalize01_torch((lv_dil - lv_ero).clamp_min(0.0))
        myo_top = _normalize01_torch(0.75 * myo_ring + 0.25 * myo_prior)
        rv_top = _normalize01_torch(rv_prior * (1.0 - 0.35 * lv_dil).clamp(0.25, 1.0))
        rv_blend = topology_strength * topology_rv_strength
        class_maps[0] = lv_prior
        class_maps[1] = _normalize01_torch((1.0 - topology_strength) * myo_prior + topology_strength * myo_top)
        class_maps[2] = _normalize01_torch((1.0 - rv_blend) * rv_prior + rv_blend * rv_top)

    if presence_preserving:
        # Per-class min/max normalization above intentionally preserves spatial
        # shape, but it also gives an absent class a unit-height prior.  Retain
        # that shape only when the frozen source prediction supplies enough
        # hard support.  This is especially important for tiny/absent apical
        # and basal RV slices.
        gated_classes = {int(x) for x in (presence_classes or [])}
        source_pred = source_prob.argmax(dim=1)
        min_pixels = max(1, int(presence_min_pixels))
        area_reference = max(min_pixels, int(presence_area_reference))
        conf_low = max(0.0, min(0.999, float(presence_confidence_low)))
        conf_high = max(conf_low + 1e-6, min(1.0, float(presence_confidence_high)))
        floor = max(0.0, min(1.0, float(presence_floor)))
        for class_offset, cls_prior in enumerate(class_maps):
            cls_idx = class_offset + 1
            if cls_idx not in gated_classes:
                continue
            hard = source_pred == cls_idx
            pixel_count = hard.sum(dim=(-2, -1)).to(dtype=dtype)
            hard_conf_sum = (source_prob[:, cls_idx] * hard.to(dtype=dtype)).sum(dim=(-2, -1))
            hard_conf = hard_conf_sum / pixel_count.clamp_min(1.0)
            area_gate = (pixel_count / float(area_reference)).clamp(0.0, 1.0)
            conf_gate = ((hard_conf - conf_low) / (conf_high - conf_low)).clamp(0.0, 1.0)
            evidence_gate = torch.maximum(area_gate, conf_gate)
            evidence_gate = torch.where(
                pixel_count >= float(min_pixels),
                evidence_gate,
                torch.zeros_like(evidence_gate),
            )
            presence_gate = floor + (1.0 - floor) * evidence_gate
            class_maps[class_offset] = cls_prior * presence_gate.view(b, 1, 1, 1)

    if presence_preserving:
        class_stack = torch.cat(class_maps, dim=1)
        fg_prior = class_stack.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    else:
        fg_prior = torch.stack(class_maps, dim=1).sum(dim=1).clamp(0.0, 1.0)
    fg_prior = _normalize01_torch(0.65 * fg_prior + 0.35 * smooth_fg)
    fg_prior = fg_prior.clamp(float(min_prior), 1.0)
    bg_prior = (1.0 - fg_prior).clamp(0.0, 1.0)
    if presence_preserving:
        # APL location/entropy use fg_prior while CE/MRE use prior_prob.  Make
        # both views describe exactly the same foreground mass.
        class_sum = class_stack.sum(dim=1, keepdim=True)
        class_conditional = class_stack / class_sum.clamp_min(1e-6)
        source_foreground = source_prob[:, 1:]
        source_conditional = source_foreground / source_foreground.sum(dim=1, keepdim=True).clamp_min(1e-6)
        class_conditional = torch.where(class_sum > 1e-6, class_conditional, source_conditional)
        class_prior = class_conditional * fg_prior
        prior_prob = torch.cat([bg_prior, class_prior], dim=1)
    else:
        # Preserve the historical path byte-for-byte when the opt-in fix is
        # disabled so frozen runs remain reproducible.
        class_prior = torch.cat(class_maps, dim=1) * fg_prior
        prior_prob = torch.cat([bg_prior, class_prior], dim=1)
        prior_prob = prior_prob / prior_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)

    q = _saliency_confidence(fg_prior).to(device=device, dtype=dtype)
    q = (q * source_conf.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    area_ok = (source_fg_area >= 1e-4) & (source_fg_area <= 0.45)
    q = torch.where(area_ok, q, q * 0.35)
    return {
        "prob": prior_prob.detach(),
        "fg": fg_prior.detach(),
        "q": q.detach(),
    }


def build_neutral_prior(source_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Provide non-spatial weights for a no-anatomical-prior ablation."""
    b, c, h, w = source_logits.shape
    dtype, device = source_logits.dtype, source_logits.device
    return {
        "prob": torch.full((b, c, h, w), 1.0 / float(c), device=device, dtype=dtype),
        "fg": torch.ones((b, 1, h, w), device=device, dtype=dtype),
        "q": torch.ones((b,), device=device, dtype=dtype),
    }


def multiclass_weighted_entropy_per_sample(logits: torch.Tensor, weight_b1hw: Optional[torch.Tensor]) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1).clamp_min(1e-6)
    ent = -(prob * prob.log()).sum(dim=1) / math.log(logits.shape[1])
    if weight_b1hw is None:
        return ent.mean(dim=(1, 2))
    w = weight_b1hw
    if w.dim() == 4:
        w = w[:, 0]
    w = w.to(device=logits.device, dtype=ent.dtype).clamp(0.0, 1.0)
    if w.shape[0] == 1 and ent.shape[0] > 1:
        w = w.repeat(ent.shape[0], 1, 1)
    return (ent * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-6)


def prior_alignment_score_per_sample(logits: torch.Tensor, prior_prob: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    prior = prior_prob.to(device=logits.device, dtype=prob.dtype)
    if prior.shape[0] == 1 and prob.shape[0] > 1:
        prior = prior.repeat(prob.shape[0], 1, 1, 1)
    p_fg = prob[:, 1:].sum(dim=1, keepdim=True)
    w_fg = prior[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    mass_in = (p_fg * w_fg).mean(dim=(1, 2, 3))
    mass_out = (p_fg * (1.0 - w_fg)).mean(dim=(1, 2, 3))
    class_overlap = (prob[:, 1:] * prior[:, 1:]).mean(dim=(1, 2, 3))
    return mass_in - mass_out + class_overlap


def foreground_area_per_sample(logits: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    return (pred > 0).float().mean(dim=(1, 2))


def class_probability_stats(logits: torch.Tensor, cls_idx: int) -> Dict[str, float]:
    prob = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    cls_prob = prob[:, cls_idx]
    pred_mask = pred == cls_idx
    soft_area = cls_prob.mean()
    pred_area = pred_mask.float().mean()
    if bool(pred_mask.any().detach().cpu().item()):
        cls_conf = cls_prob[pred_mask].mean()
    else:
        cls_conf = logits.new_tensor(0.0)
    return {
        "soft_area": float(soft_area.detach().cpu().item()),
        "pred_area": float(pred_area.detach().cpu().item()),
        "confidence": float(cls_conf.detach().cpu().item()),
    }


def candidate_dice_summary(
    logits: torch.Tensor,
    label_np: np.ndarray,
    metric_fn,
    number_class: int,
) -> Dict[str, float]:
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    rows = []
    for i in range(pred.shape[0]):
        dice = metric_fn(pred[i], label_np[i], number_class, "dice")
        rows.append([float(x) for x in dice])
    if not rows:
        return {"avg": float("nan"), "class_1": float("nan"), "class_2": float("nan"), "class_3": float("nan")}
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "avg": float(arr.mean(axis=1).mean()),
        "class_1": float(arr[:, 0].mean()),
        "class_2": float(arr[:, 1].mean()),
        "class_3": float(arr[:, 2].mean()),
    }


def surface_metrics_per_class(
    pred: np.ndarray,
    label: np.ndarray,
    number_class: int,
    *,
    assd_protocol: str = "scipy_exact",
    metric_fn=None,
) -> Tuple[List[float], List[float]]:
    """Compute per-class ASSD and HD95 for MMS slices.

    ``scipy_exact`` preserves the clean-run evaluator used by the APSA runner:
    exact Euclidean 2-D ASSD, no cap, and zero for one-empty classes.
    ``sictta_capped`` reproduces the public A3-TTA/SiCTTA table protocol through
    its frozen metric implementation: per-slice 2-D ASSD capped at 10 pixels,
    with one-empty classes receiving 10 and both-empty classes receiving 0.
    HD95 retains the runner's exact Euclidean implementation in both modes.
    """
    from scipy import ndimage

    if assd_protocol not in {"scipy_exact", "sictta_capped"}:
        raise ValueError(f"Unknown ASSD protocol: {assd_protocol}")
    if assd_protocol == "sictta_capped" and metric_fn is None:
        raise ValueError("sictta_capped ASSD requires the frozen public metric function")

    assd_values: List[float] = []
    hd95_values: List[float] = []
    diagonal = float(math.sqrt(sum(float(size) ** 2 for size in pred.shape[-2:])))
    for cls_idx in range(1, int(number_class)):
        pred_any = bool((pred == cls_idx).any())
        label_any = bool((label == cls_idx).any())
        if not pred_any and not label_any:
            assd_values.append(0.0)
            hd95_values.append(float("nan"))
        elif pred_any != label_any:
            # Preserve the established MMS ASSD protocol for comparability.
            assd_values.append(0.0)
            hd95_values.append(diagonal)
        else:
            pred_mask = pred == cls_idx
            label_mask = label == cls_idx
            structure = ndimage.generate_binary_structure(pred_mask.ndim, 1)
            pred_surface = pred_mask ^ ndimage.binary_erosion(pred_mask, structure=structure, border_value=0)
            label_surface = label_mask ^ ndimage.binary_erosion(label_mask, structure=structure, border_value=0)
            distance_to_pred = ndimage.distance_transform_edt(~pred_surface)
            distance_to_label = ndimage.distance_transform_edt(~label_surface)
            label_to_pred = distance_to_pred[label_surface]
            pred_to_label = distance_to_label[pred_surface]
            assd_values.append(float((label_to_pred.sum() + pred_to_label.sum()) / (label_to_pred.size + pred_to_label.size)))
            hd95_values.append(
                float(max(np.percentile(label_to_pred, 95), np.percentile(pred_to_label, 95)))
            )
    if assd_protocol == "sictta_capped":
        assd_values = [
            float(value)
            for value in metric_fn(pred, label, int(number_class), "assd")
        ]
    return assd_values, hd95_values


def nanmean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(valid.mean()) if valid.size else float("nan")


def trace_candidate_name(name: str) -> str:
    key = str(name).strip()
    aliases = {
        "source": "P_source",
        "adapted": "P_adapt",
        "refined": "P_APL",
        "final": "P_final",
    }
    if key in aliases:
        return aliases[key]
    if key.startswith("clean_"):
        return f"P_{key}"
    return f"P_{key}"


def candidate_metric_summary(
    logits: torch.Tensor,
    label_np: np.ndarray,
    metric_fn,
    number_class: int,
    include_assd: bool = False,
    assd_protocol: str = "scipy_exact",
) -> Dict[str, float]:
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    dice_rows = []
    assd_rows = []
    for i in range(pred.shape[0]):
        dice_rows.append([float(x) for x in metric_fn(pred[i], label_np[i], number_class, "dice")])
        if include_assd:
            assd, _ = surface_metrics_per_class(
                pred[i],
                label_np[i],
                number_class,
                assd_protocol=assd_protocol,
                metric_fn=metric_fn,
            )
            assd_rows.append([float(x) for x in assd])
    if not dice_rows:
        values = {}
        for prefix in ["dice", "assd"]:
            values[f"{prefix}_avg"] = float("nan")
            for cls in range(1, 4):
                values[f"{prefix}_class_{cls}"] = float("nan")
        return values
    dice_arr = np.asarray(dice_rows, dtype=np.float64)
    values = {
        "dice_avg": float(dice_arr.mean(axis=1).mean()),
        "dice_class_1": float(dice_arr[:, 0].mean()),
        "dice_class_2": float(dice_arr[:, 1].mean()),
        "dice_class_3": float(dice_arr[:, 2].mean()),
    }
    if include_assd:
        assd_arr = np.asarray(assd_rows, dtype=np.float64)
        values.update(
            {
                "assd_avg": float(assd_arr.mean(axis=1).mean()),
                "assd_class_1": float(assd_arr[:, 0].mean()),
                "assd_class_2": float(assd_arr[:, 1].mean()),
                "assd_class_3": float(assd_arr[:, 2].mean()),
            }
        )
    else:
        values.update(
            {
                "assd_avg": float("nan"),
                "assd_class_1": float("nan"),
                "assd_class_2": float("nan"),
                "assd_class_3": float("nan"),
            }
        )
    return values


def probe_future_state_pair(
    *,
    model,
    source_model,
    pre_state: Dict[str, object],
    candidate_state: Dict[str, object],
    dataset,
    current_batch_idx: int,
    batch_size: int,
    max_horizon: int,
    device: torch.device,
    calibration_corruption: str,
    metric_fn,
    number_class: int,
    surface_assd_protocol: str,
    prior_kwargs: Dict[str, object],
    apl_kwargs: Dict[str, object],
) -> List[Dict[str, object]]:
    """Run matched local APL trajectories from accept/revert states.

    Labels are used only to populate the returned offline audit. They never
    influence the RPR decision, future APL updates, or the state propagated by
    the main stream. Additional RPR decisions are disabled inside the local
    trajectories so that the original trigger is the only intervention.
    """
    rows: List[Dict[str, object]] = []
    dataset_size = len(dataset)
    try:
        future_batches: List[Dict[str, object]] = []
        for future_step in range(1, int(max_horizon) + 1):
            start = (int(current_batch_idx) + future_step) * int(batch_size)
            if start >= dataset_size:
                break
            end = min(start + int(batch_size), dataset_size)
            future_loader = DataLoader(
                Subset(dataset, range(start, end)),
                batch_size=end - start,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            sampled = next(iter(future_loader))
            volume = sampled["image"].to(device)
            label = sampled["label"].to(device)
            names = sampled["names"]
            volume, label = convert_2d(volume, label)
            volume = apply_source_calibration_corruption(
                volume,
                calibration_corruption,
                batch_index=int(current_batch_idx) + future_step,
            )
            label_np = label.detach().cpu().numpy().squeeze(1)
            with torch.no_grad():
                source_logits = source_model(volume)
            prior = build_cardiac_prior(source_logits, **prior_kwargs)
            future_batches.append(
                {
                    "future_step": future_step,
                    "future_batch_idx": int(current_batch_idx) + future_step,
                    "future_names": ";".join(
                        os.path.basename(str(name)) for name in names
                    ),
                    "volume": volume,
                    "label_np": label_np,
                    "source_logits": source_logits,
                    "prior": prior,
                }
            )

        branch_rows: Dict[str, List[Dict[str, Dict[str, float]]]] = {}
        for branch_name, state in (
            ("revert", pre_state),
            ("accept", candidate_state),
            ("revert_repeat", pre_state),
        ):
            restore_a3_state(model, state)
            branch_rows[branch_name] = []
            for future in future_batches:
                volume = future["volume"]
                label_np = future["label_np"]
                source_logits = future["source_logits"]
                prior = future["prior"]
                _set_adaptation_mode(
                    model.model, disable_dropout=bool(model.disable_dropout)
                )
                with torch.no_grad():
                    pre_logits = model.model(volume)
                pre_metrics = candidate_metric_summary(
                    pre_logits,
                    label_np,
                    metric_fn,
                    number_class,
                    include_assd=True,
                    assd_protocol=surface_assd_protocol,
                )
                post_logits = model.forward_apl(
                    volume,
                    prior,
                    source_logits,
                    **apl_kwargs,
                )
                post_metrics = candidate_metric_summary(
                    post_logits,
                    label_np,
                    metric_fn,
                    number_class,
                    include_assd=True,
                    assd_protocol=surface_assd_protocol,
                )
                branch_rows[branch_name].append(
                    {"pre": pre_metrics, "post": post_metrics}
                )

        for future_index, future in enumerate(future_batches):
            row: Dict[str, object] = {
                "future_step": future["future_step"],
                "future_batch_idx": future["future_batch_idx"],
                "future_names": future["future_names"],
            }
            for stage in ("pre", "post"):
                accept = branch_rows["accept"][future_index][stage]
                revert = branch_rows["revert"][future_index][stage]
                revert_repeat = branch_rows["revert_repeat"][future_index][stage]
                for metric in ("dice", "assd"):
                    for suffix in ("avg", "class_1", "class_2", "class_3"):
                        accept_value = float(accept[f"{metric}_{suffix}"])
                        revert_value = float(revert[f"{metric}_{suffix}"])
                        row[f"accept_{stage}_{metric}_{suffix}"] = accept_value
                        row[f"revert_{stage}_{metric}_{suffix}"] = revert_value
                        if metric == "dice":
                            gain = revert_value - accept_value
                        else:
                            gain = accept_value - revert_value
                        row[f"revert_gain_{stage}_{metric}_{suffix}"] = gain
                        repeat_value = float(
                            revert_repeat[f"{metric}_{suffix}"]
                        )
                        row[
                            f"revert_repeat_{stage}_{metric}_{suffix}"
                        ] = repeat_value
                        row[
                            f"null_repeat_delta_{stage}_{metric}_{suffix}"
                        ] = repeat_value - revert_value
            rows.append(row)
    finally:
        # Candidate state was active when probing started. The caller applies
        # the actual RPR decision only after this state-neutral audit returns.
        restore_a3_state(model, candidate_state)
    return rows


def center_distance_from_prior(logits: torch.Tensor, prior_fg: torch.Tensor) -> Optional[float]:
    pred = logits.argmax(dim=1)
    dists = []
    w = prior_fg
    if w.dim() == 4:
        w = w[:, 0]
    for bi in range(pred.shape[0]):
        pred_fg = (pred[bi] > 0).float()
        ctr_pred = _soft_centroid_xy(pred_fg)
        if ctr_pred is None:
            continue
        ctr_prior = _soft_centroid_xy(w[bi])
        if ctr_prior is None:
            qx, qy = _argmax_xy(w[bi])
            ctr_prior = (
                torch.tensor(qx, device=logits.device, dtype=logits.dtype),
                torch.tensor(qy, device=logits.device, dtype=logits.dtype),
            )
        px, py = ctr_pred
        qx, qy = ctr_prior
        hh = float(pred.shape[-2])
        ww = float(pred.shape[-1])
        dists.append(torch.sqrt(((px - qx) / max(1.0, ww)) ** 2 + ((py - qy) / max(1.0, hh)) ** 2))
    if not dists:
        return None
    return float(torch.stack(dists).mean().detach().cpu().item())


def compute_mre_info(
    logits: torch.Tensor,
    source_logits: torch.Tensor,
    prior: Dict[str, torch.Tensor],
    name: str,
    area_min: float,
    area_max: float,
    center_max: float,
    mre_lambda_ap: float,
    topology_weight: float,
    topology_ring_kernel: int,
    use_entropy: bool,
    use_prior_align: bool,
    use_area: bool,
    use_center: bool,
    use_prior_conf: bool,
    use_topology: bool,
) -> Dict[str, float]:
    ent = multiclass_weighted_entropy_per_sample(logits, prior["fg"]).mean()
    align = prior_alignment_score_per_sample(logits, prior["prob"]).mean()
    area = foreground_area_per_sample(logits).mean()
    center_dist = center_distance_from_prior(logits, prior["fg"])
    q_mean = prior["q"].mean()
    drift = probability_drift(logits, source_logits)
    conf = mean_confidence(logits)
    topo = cardiac_topology_score(logits, ring_kernel=topology_ring_kernel)
    rv_stats = class_probability_stats(logits, cls_idx=3)

    area_mean = float(area.detach().cpu().item())
    area_ok = bool(float(area_min) <= area_mean <= float(area_max))
    center_ok = True if center_dist is None else bool(center_dist <= float(center_max))
    ent_v = float(ent.detach().cpu().item())
    align_v = float(align.detach().cpu().item())
    q_v = float(q_mean.detach().cpu().item())

    score_generic = 0.0
    if use_entropy:
        score_generic -= 5.0 * ent_v
        score_generic += 1.25 * conf
    score_anat = 0.0
    if use_prior_align:
        score_anat += 2.0 * align_v
    if use_area and not area_ok:
        score_anat -= 3.0
    if use_center and center_dist is not None:
        score_anat -= 2.0 * float(center_dist)
    if use_prior_conf and q_v < 0.30:
        score_anat -= 1.0 * (0.30 - q_v)
    if use_topology:
        score_anat += float(topology_weight) * float(topo)

    return {
        "branch": name,
        "score": float(score_generic + float(mre_lambda_ap) * score_anat),
        "score_generic": float(score_generic),
        "score_anatomical": float(score_anat),
        "entropy": ent_v,
        "entropy_mean": ent_v,
        "prior_align": align_v,
        "prior_align_mean": align_v,
        "area_mean": area_mean,
        "area_ok": area_ok,
        "center_dist": None if center_dist is None else float(center_dist),
        "center_ok": center_ok,
        "prior_q_mean": q_v,
        "topology_score": float(topo),
        "rv_soft_area": rv_stats["soft_area"],
        "rv_pred_area": rv_stats["pred_area"],
        "rv_confidence": rv_stats["confidence"],
        "confidence": conf,
        "drift": drift,
        "fg_area": area_mean,
        "geometry": cardiac_geometry_score(logits, source_logits),
    }


def _slice_prior(prior: Dict[str, torch.Tensor], index: int) -> Dict[str, torch.Tensor]:
    sliced: Dict[str, torch.Tensor] = {}
    for key, value in prior.items():
        if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) > index:
            sliced[key] = value[index : index + 1]
        else:
            sliced[key] = value
    return sliced


def safe_branch_select_mre(
    info_src: Dict[str, float],
    info_ada: Dict[str, float],
    info_fin: Dict[str, float],
    conservative: bool,
    adaptive: bool,
    allow_source_output: bool,
    source_output_penalty: float,
    final_min_gain: float,
    final_bias: float,
    final_geometry_guard: float,
    use_entropy: bool,
    use_prior_align: bool,
    use_area: bool,
    use_topology: bool,
    topology_guard: float,
    use_rv_guard: bool,
    rv_guard_min_area: float,
    rv_guard_min_ratio: float,
    rv_guard_conf_drop: float,
    final_takeover_v2: bool,
    final_takeover_score_gain: float,
    final_takeover_geometry_gain: float,
    final_takeover_align_gain: float,
    final_takeover_entropy_increase: float,
    source_rescue_v2: bool,
    source_rescue_margin: float,
    source_rescue_min_geometry: float,
    source_rescue_rv_min_ratio: float,
    source_rescue_rv_max_ratio: float,
    source_rescue_align_margin: float,
    source_rescue_entropy_margin: float,
) -> Tuple[str, str]:
    scores = {k: float(v["score"]) for k, v in {"source": info_src, "adapted": info_ada, "final": info_fin}.items()}
    ada_ent = info_ada["entropy_mean"]
    fin_ent = info_fin["entropy_mean"]
    ada_align = info_ada.get("prior_align_mean")
    fin_align = info_fin.get("prior_align_mean")
    ada_area = info_ada.get("area_mean", 0.0)
    fin_area = info_fin.get("area_mean", 0.0)
    ada_geom = info_ada.get("geometry")
    fin_geom = info_fin.get("geometry")
    ada_topo = info_ada.get("topology_score")
    fin_topo = info_fin.get("topology_score")
    ada_rv_soft = float(info_ada.get("rv_soft_area", 0.0))
    fin_rv_soft = float(info_fin.get("rv_soft_area", 0.0))
    ada_rv_pred = float(info_ada.get("rv_pred_area", 0.0))
    fin_rv_pred = float(info_fin.get("rv_pred_area", 0.0))
    ada_rv_conf = float(info_ada.get("rv_confidence", 0.0))
    fin_rv_conf = float(info_fin.get("rv_confidence", 0.0))

    final_blocked = False
    final_block_reasons = []
    if use_entropy and fin_ent > ada_ent * 1.8 and fin_ent > ada_ent + 0.006:
        final_blocked = True
        final_block_reasons.append("entropy")
    if use_area and ada_area > 0.003 and fin_area < ada_area * 0.5:
        final_blocked = True
        final_block_reasons.append("fg_area")
    if use_prior_align and ada_align is not None and fin_align is not None and fin_align < ada_align - 0.02:
        final_blocked = True
        final_block_reasons.append("prior_align")
    geom_guard = max(0.0, float(final_geometry_guard))
    if geom_guard > 0.0 and ada_geom is not None and fin_geom is not None and fin_geom < ada_geom - geom_guard:
        final_blocked = True
        final_block_reasons.append("geometry")
    if use_topology and ada_topo is not None and fin_topo is not None and fin_topo < ada_topo - float(topology_guard):
        final_blocked = True
        final_block_reasons.append("topology")
    if use_rv_guard:
        min_area = float(rv_guard_min_area)
        min_ratio = float(rv_guard_min_ratio)
        conf_drop = float(rv_guard_conf_drop)
        rv_soft_collapsed = ada_rv_soft > min_area and fin_rv_soft < ada_rv_soft * min_ratio
        rv_pred_collapsed = ada_rv_pred > min_area and fin_rv_pred < ada_rv_pred * min_ratio
        rv_conf_dropped = ada_rv_conf > 0.0 and fin_rv_conf < ada_rv_conf - conf_drop and fin_rv_soft < ada_rv_soft
        if rv_soft_collapsed or rv_pred_collapsed or rv_conf_dropped:
            final_blocked = True
            final_block_reasons.append("rv_guard")

    src_score = scores["source"]
    ada_score = scores["adapted"]
    fin_score = scores["final"]
    src_broken = (use_area and (not info_src.get("area_ok", True))) or src_score < -1.2

    viable: Dict[str, float] = {}
    if not final_blocked:
        viable["final"] = fin_score if conservative else fin_score + float(final_bias)
    viable["adapted"] = ada_score
    source_penalty = float(source_output_penalty)
    if use_area and (not info_ada.get("area_ok", True)) and (not info_fin.get("area_ok", True)):
        source_penalty = 0.0
    if allow_source_output:
        viable["source"] = src_score - source_penalty
    if allow_source_output and src_broken:
        max_other = max(ada_score, fin_score if not final_blocked else ada_score)
        if max_other > src_score + 0.05:
            viable.pop("source", None)

    best = max(viable, key=viable.get)
    adaptive_conservative = False
    if adaptive and not final_blocked:
        adapted_plausible = (not use_area or info_ada.get("area_ok", True)) and (not use_entropy or ada_ent < 0.20)
        source_plausible = (not use_area or info_src.get("area_ok", True)) and (not use_entropy or info_src["entropy_mean"] < 0.20)
        final_entropy_inflated = use_entropy and fin_ent > ada_ent + 0.025 and fin_ent > ada_ent * 1.35
        final_area_inflated = use_area and ada_area > 0.003 and fin_area > ada_area * 1.35
        final_align_gain = 0.0 if fin_align is None or ada_align is None else fin_align - ada_align
        final_topology_gain = 0.0 if fin_topo is None or ada_topo is None else fin_topo - ada_topo
        small_align_gain = (not use_prior_align) or final_align_gain < 0.004
        small_topology_gain = (not use_topology) or final_topology_gain < 0.015
        adaptive_conservative = (adapted_plausible or source_plausible) and (
            (final_entropy_inflated and small_align_gain and small_topology_gain) or final_area_inflated
        )

    extra = None
    if (conservative or adaptive_conservative) and best == "final":
        non_final = {"adapted": ada_score, "source": src_score - source_penalty}
        if "source" not in viable:
            non_final.pop("source", None)
        best_non_final = max(non_final, key=non_final.get)
        if fin_score < non_final[best_non_final] + float(final_min_gain):
            best = best_non_final
            extra = "conservative_final_rejected"
        else:
            extra = "conservative_final_accepted"

    v2_reasons = []
    if final_takeover_v2 and best == "final":
        final_score_gain = fin_score - ada_score
        final_geometry_gain = 0.0 if fin_geom is None or ada_geom is None else float(fin_geom) - float(ada_geom)
        final_align_gain = 0.0 if fin_align is None or ada_align is None else float(fin_align) - float(ada_align)
        final_entropy_inc = fin_ent - ada_ent
        lacks_score_gain = final_score_gain < float(final_takeover_score_gain)
        lacks_geometry_gain = final_geometry_gain < float(final_takeover_geometry_gain)
        lacks_align_gain = (not use_prior_align) or final_align_gain < float(final_takeover_align_gain)
        entropy_increased = (not use_entropy) or final_entropy_inc > float(final_takeover_entropy_increase)
        if lacks_score_gain and lacks_geometry_gain and lacks_align_gain and entropy_increased:
            best = "adapted"
            v2_reasons.append(
                "final_takeover_v2_rejected"
                f":dg={final_score_gain:.4f}/geom={final_geometry_gain:.4f}/align={final_align_gain:.4f}/ent={final_entropy_inc:.4f}"
            )

    if source_rescue_v2 and allow_source_output and not src_broken:
        src_close = src_score >= max(ada_score, fin_score) - float(source_rescue_margin)
        ada_geom_ok = ada_geom is None or float(ada_geom) >= float(source_rescue_min_geometry)
        fin_geom_ok = fin_geom is None or float(fin_geom) >= float(source_rescue_min_geometry)
        src_align_v = info_src.get("prior_align_mean")
        src_align_best = (
            (not use_prior_align)
            or src_align_v is None
            or (
                (ada_align is not None and fin_align is not None)
                and float(src_align_v) >= max(float(ada_align), float(fin_align)) + float(source_rescue_align_margin)
            )
        )
        src_entropy_ok = (not use_entropy) or (
            info_src["entropy_mean"] <= min(ada_ent, fin_ent) + float(source_rescue_entropy_margin)
        )
        src_area_ok = (not use_area) or bool(info_src.get("area_ok", True))
        rv_stable = True
        src_rv_soft = float(info_src.get("rv_soft_area", 0.0))
        if src_rv_soft > 1e-8:
            ada_rv_ratio = ada_rv_soft / src_rv_soft
            fin_rv_ratio = fin_rv_soft / src_rv_soft
            rv_stable = (
                float(source_rescue_rv_min_ratio) <= ada_rv_ratio <= float(source_rescue_rv_max_ratio)
                and float(source_rescue_rv_min_ratio) <= fin_rv_ratio <= float(source_rescue_rv_max_ratio)
            )
        if src_close and ada_geom_ok and fin_geom_ok and src_align_best and src_entropy_ok and src_area_ok and rv_stable:
            best = "source"
            v2_reasons.append("source_rescue_v2")

    reason = [f"scores={ {k: round(v, 3) for k, v in scores.items()} }"]
    if final_blocked:
        blocked = "+".join(final_block_reasons) if final_block_reasons else "unknown"
        reason.append(f"final_blocked:{blocked}")
    if src_broken:
        reason.append("src_broken")
    if adaptive:
        reason.append("adaptive_conservative" if adaptive_conservative else "adaptive_allow_final")
    if extra:
        reason.append(extra)
    reason.extend(v2_reasons)
    reason.append(f"best={best}")
    return best, ",".join(reason)


def _class_weight_tensor(
    logits: torch.Tensor,
    class_weights: Optional[Iterable[float]],
) -> Optional[torch.Tensor]:
    if class_weights is None:
        return None
    weights = [max(0.0, float(x)) for x in class_weights]
    class_count = int(logits.shape[1])
    if len(weights) == class_count - 1:
        weights = [1.0] + weights
    if len(weights) != class_count:
        raise ValueError(f"Expected {class_count - 1} foreground or {class_count} class prior weights, got {len(weights)}")
    return torch.tensor(weights, device=logits.device, dtype=logits.dtype).view(1, class_count, 1, 1)


def weighted_probability_consistency(
    probability: torch.Tensor,
    target_probability: torch.Tensor,
    class_weights: Optional[Iterable[float]],
) -> torch.Tensor:
    """Class-balanced probability MSE with foreground-only weight shorthand."""
    diff = (probability - target_probability.detach()).pow(2)
    weight = _class_weight_tensor(probability, class_weights)
    if weight is None:
        return diff.mean()
    normalizer = weight.sum().clamp_min(1e-6) * float(diff.shape[0] * diff.shape[2] * diff.shape[3])
    return (diff * weight).sum() / normalizer


def reliability_gated_temporal_consistency(
    logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    prior_probability: torch.Tensor,
    *,
    class_weights: Optional[Iterable[float]],
    confidence_threshold: float,
    rv_confidence_threshold: float,
    prior_alignment_threshold: float,
    teacher_temperature: float,
) -> Tuple[torch.Tensor, float]:
    """EMA consistency restricted to confident, anatomy-aligned pixels."""
    with torch.no_grad():
        teacher = teacher_probability.to(device=logits.device, dtype=logits.dtype).detach().clamp_min(1e-7)
        temperature = max(0.05, float(teacher_temperature))
        teacher_target = torch.softmax(teacher.log() / temperature, dim=1)
        prior = prior_probability.to(device=logits.device, dtype=logits.dtype).detach().clamp_min(1e-7)
        if prior.shape[0] == 1 and logits.shape[0] > 1:
            prior = prior.repeat(logits.shape[0], 1, 1, 1)
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-7)

        confidence, teacher_class = teacher.max(dim=1)
        base_threshold = max(0.0, min(0.999, float(confidence_threshold)))
        rv_threshold = max(0.0, min(0.999, float(rv_confidence_threshold)))
        threshold = torch.full_like(confidence, base_threshold)
        if int(logits.shape[1]) > 3:
            threshold = torch.where(teacher_class == 3, threshold.new_tensor(rv_threshold), threshold)
        confidence_gate = ((confidence - threshold) / (1.0 - threshold).clamp_min(1e-6)).clamp(0.0, 1.0)

        prior_alignment = F.cosine_similarity(teacher, prior, dim=1, eps=1e-6).clamp(0.0, 1.0)
        alignment_threshold = max(0.0, min(0.999, float(prior_alignment_threshold)))
        prior_gate = ((prior_alignment - alignment_threshold) / (1.0 - alignment_threshold)).clamp(0.0, 1.0)

        class_weight = _class_weight_tensor(logits, class_weights)
        if class_weight is None:
            class_gate = torch.ones_like(confidence)
        else:
            per_class = class_weight.view(-1)
            class_gate = per_class[teacher_class]
        reliability = confidence_gate * prior_gate * class_gate
        reliability_fraction = float((reliability > 1e-6).float().mean().detach().cpu().item())

    pixel_kl = F.kl_div(F.log_softmax(logits, dim=1), teacher_target, reduction="none").sum(dim=1).clamp_min(0.0)
    if float(reliability.sum().detach().cpu().item()) <= 1e-8:
        return logits.new_tensor(0.0), reliability_fraction
    loss = (pixel_kl * reliability).sum() / reliability.sum().clamp_min(1e-6)
    return loss, reliability_fraction


def soft_prior_ce(
    logits: torch.Tensor,
    target_prob: torch.Tensor,
    weight_b1hw: torch.Tensor,
    class_weights: Optional[Iterable[float]] = None,
) -> torch.Tensor:
    target = target_prob.to(device=logits.device, dtype=logits.dtype)
    weight = weight_b1hw.to(device=logits.device, dtype=logits.dtype)
    if weight.shape[0] == 1 and logits.shape[0] > 1:
        weight = weight.repeat(logits.shape[0], 1, 1, 1)
    log_prob = F.log_softmax(logits, dim=1)
    class_weight = _class_weight_tensor(logits, class_weights)
    if class_weight is not None:
        target = target * class_weight
    loss = -(target * log_prob).sum(dim=1, keepdim=True)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def uncertainty_gated_boundary_prior_ce(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_prob: torch.Tensor,
    *,
    classes: Optional[Iterable[int]],
    kernel_size: int,
    uncertainty_low: float,
    uncertainty_high: float,
) -> torch.Tensor:
    """Apply the anatomical prior only on uncertain MYO/RV boundary bands."""
    class_ids = [int(x) for x in (classes or []) if 0 < int(x) < int(logits.shape[1])]
    if not class_ids:
        return logits.new_tensor(0.0)

    with torch.no_grad():
        teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)
        uncertainty = 1.0 - teacher_prob.max(dim=1, keepdim=True)[0]
        low = max(0.0, min(1.0, float(uncertainty_low)))
        high = max(low + 1e-6, min(1.0, float(uncertainty_high)))
        uncertainty_gate = ((uncertainty - low) / (high - low)).clamp(0.0, 1.0)

        boundary_maps = []
        for cls in class_ids:
            cls_prob = teacher_prob[:, cls : cls + 1]
            dilated = _soft_dilate(cls_prob, kernel_size)
            eroded = _soft_erode(cls_prob, kernel_size)
            boundary_maps.append((dilated - eroded).clamp_min(0.0))
        boundary = torch.stack(boundary_maps, dim=0).amax(dim=0)
        boundary = _normalize01_torch(boundary)

        prior = target_prob.to(device=logits.device, dtype=logits.dtype)
        prior_class_conf = prior[:, class_ids].amax(dim=1, keepdim=True)
        spatial_weight = boundary * uncertainty_gate * (0.5 + 0.5 * prior_class_conf)

    if float(spatial_weight.sum().detach().cpu().item()) <= 1e-8:
        return logits.new_tensor(0.0)
    log_prob = F.log_softmax(logits, dim=1)
    pixel_ce = -(prior * log_prob).sum(dim=1, keepdim=True)
    return (pixel_ce * spatial_weight).sum() / spatial_weight.sum().clamp_min(1e-6)


def prior_guided_refine(
    a3_model,
    x: torch.Tensor,
    source_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    prior: Dict[str, torch.Tensor],
    *,
    entropy_lambda: float,
    loc_lambda: float,
    bce_lambda: float,
    ts_lambda: float,
    source_consistency_lambda: float,
    weight_gamma: float,
    weight_min: float,
    ema_alpha: float,
    refine_steps: int,
    refine_learning_rate: float,
    class_weights: Optional[Iterable[float]],
    consistency_class_weights: Optional[Iterable[float]],
    boundary_prior_lambda: float,
    boundary_prior_classes: Optional[Iterable[int]],
    boundary_prior_kernel: int,
    boundary_uncertainty_low: float,
    boundary_uncertainty_high: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    model = a3_model.model
    optimizer = a3_model.optimizer
    _set_adaptation_mode(
        model,
        disable_dropout=bool(getattr(a3_model, "disable_dropout", False)),
    )
    step_count = max(1, int(refine_steps))
    saved_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = float(refine_learning_rate)
    stats_accum = {"loss": 0.0, "ent": 0.0, "loc": 0.0, "ce": 0.0, "boundary": 0.0, "ts": 0.0, "src_cons": 0.0}
    try:
        for _ in range(step_count):
            with torch.enable_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)
                p_fg = prob[:, 1:].sum(dim=1, keepdim=True)
                w = prior["fg"].to(device=x.device, dtype=logits.dtype).clamp(0.0, 1.0)
                if w.shape[0] == 1 and logits.shape[0] > 1:
                    w = w.repeat(logits.shape[0], 1, 1, 1)
                wg = w.pow(max(0.1, float(weight_gamma)))
                wg = wg * (1.0 - float(weight_min)) + float(weight_min)

                ent = multiclass_weighted_entropy_per_sample(logits, wg).mean()
                mass_in = (p_fg * w).mean()
                mass_out = (p_fg * (1.0 - w)).mean()
                loc = mass_out - mass_in
                ce = soft_prior_ce(logits, prior["prob"], wg, class_weights=class_weights)

                loss = float(entropy_lambda) * ent
                loss = loss + float(loc_lambda) * loc
                loss = loss + float(bce_lambda) * ce
                boundary = logits.new_tensor(0.0)
                if float(boundary_prior_lambda) > 0.0:
                    boundary = uncertainty_gated_boundary_prior_ce(
                        logits,
                        teacher_logits,
                        prior["prob"],
                        classes=boundary_prior_classes,
                        kernel_size=boundary_prior_kernel,
                        uncertainty_low=boundary_uncertainty_low,
                        uncertainty_high=boundary_uncertainty_high,
                    )
                    loss = loss + float(boundary_prior_lambda) * boundary
                ts = logits.new_tensor(0.0)
                if float(ts_lambda) > 0.0:
                    with torch.no_grad():
                        teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)
                        teacher_conf = teacher_prob.max(dim=1)[0].mean()
                    ts = weighted_probability_consistency(
                        prob,
                        teacher_prob,
                        consistency_class_weights,
                    ) * teacher_conf.clamp(0.0, 1.0)
                    loss = loss + float(ts_lambda) * ts
                src_cons = logits.new_tensor(0.0)
                if float(source_consistency_lambda) > 0.0:
                    src_prob = torch.softmax(source_logits.detach(), dim=1)
                    src_cons = weighted_probability_consistency(prob, src_prob, consistency_class_weights)
                    loss = loss + float(source_consistency_lambda) * src_cons

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            stats_accum["loss"] += float(loss.detach().cpu().item())
            stats_accum["ent"] += float(ent.detach().cpu().item())
            stats_accum["loc"] += float(loc.detach().cpu().item())
            stats_accum["ce"] += float(ce.detach().cpu().item())
            stats_accum["boundary"] += float(boundary.detach().cpu().item())
            stats_accum["ts"] += float(ts.detach().cpu().item())
            stats_accum["src_cons"] += float(src_cons.detach().cpu().item())
    finally:
        for group, learning_rate in zip(optimizer.param_groups, saved_learning_rates):
            group["lr"] = learning_rate

    del ema_alpha
    with torch.no_grad():
        refined = model(x)
    return refined, {key: value / float(step_count) for key, value in stats_accum.items()}


def prior_guided_refine_candidate(
    a3_model,
    x: torch.Tensor,
    source_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    prior: Dict[str, torch.Tensor],
    *,
    entropy_lambda: float,
    loc_lambda: float,
    bce_lambda: float,
    ts_lambda: float,
    source_consistency_lambda: float,
    weight_gamma: float,
    weight_min: float,
    refine_steps: int,
    refine_learning_rate: float,
    class_weights: Optional[Iterable[float]],
    consistency_class_weights: Optional[Iterable[float]],
    boundary_prior_lambda: float,
    boundary_prior_classes: Optional[Iterable[int]],
    boundary_prior_kernel: int,
    boundary_uncertainty_low: float,
    boundary_uncertainty_high: float,
    template_model: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Generate a prior-guided final candidate without touching A3 online state."""
    template = template_model if template_model is not None else getattr(a3_model, "model_anchor", a3_model.model)
    model = copy.deepcopy(template).to(x.device)
    if template_model is None:
        model.load_state_dict(a3_model.model.state_dict(), strict=True)
    _set_adaptation_mode(
        model,
        disable_dropout=bool(getattr(a3_model, "disable_dropout", False)),
    )
    for param in model.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(refine_learning_rate), betas=(0.5, 0.999))
    step_count = max(1, int(refine_steps))
    stats_accum = {"loss": 0.0, "ent": 0.0, "loc": 0.0, "ce": 0.0, "boundary": 0.0, "ts": 0.0, "src_cons": 0.0}
    for _ in range(step_count):
        with torch.enable_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            p_fg = prob[:, 1:].sum(dim=1, keepdim=True)
            w = prior["fg"].to(device=x.device, dtype=logits.dtype).clamp(0.0, 1.0)
            if w.shape[0] == 1 and logits.shape[0] > 1:
                w = w.repeat(logits.shape[0], 1, 1, 1)
            wg = w.pow(max(0.1, float(weight_gamma)))
            wg = wg * (1.0 - float(weight_min)) + float(weight_min)

            ent = multiclass_weighted_entropy_per_sample(logits, wg).mean()
            mass_in = (p_fg * w).mean()
            mass_out = (p_fg * (1.0 - w)).mean()
            loc = mass_out - mass_in
            ce = soft_prior_ce(logits, prior["prob"], wg, class_weights=class_weights)

            loss = float(entropy_lambda) * ent
            loss = loss + float(loc_lambda) * loc
            loss = loss + float(bce_lambda) * ce
            boundary = logits.new_tensor(0.0)
            if float(boundary_prior_lambda) > 0.0:
                boundary = uncertainty_gated_boundary_prior_ce(
                    logits,
                    teacher_logits,
                    prior["prob"],
                    classes=boundary_prior_classes,
                    kernel_size=boundary_prior_kernel,
                    uncertainty_low=boundary_uncertainty_low,
                    uncertainty_high=boundary_uncertainty_high,
                )
                loss = loss + float(boundary_prior_lambda) * boundary
            ts = logits.new_tensor(0.0)
            if float(ts_lambda) > 0.0:
                with torch.no_grad():
                    teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)
                    teacher_conf = teacher_prob.max(dim=1)[0].mean()
                ts = weighted_probability_consistency(
                    prob,
                    teacher_prob,
                    consistency_class_weights,
                ) * teacher_conf.clamp(0.0, 1.0)
                loss = loss + float(ts_lambda) * ts
            src_cons = logits.new_tensor(0.0)
            if float(source_consistency_lambda) > 0.0:
                src_prob = torch.softmax(source_logits.detach(), dim=1)
                src_cons = weighted_probability_consistency(prob, src_prob, consistency_class_weights)
                loss = loss + float(source_consistency_lambda) * src_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        stats_accum["loss"] += float(loss.detach().cpu().item())
        stats_accum["ent"] += float(ent.detach().cpu().item())
        stats_accum["loc"] += float(loc.detach().cpu().item())
        stats_accum["ce"] += float(ce.detach().cpu().item())
        stats_accum["boundary"] += float(boundary.detach().cpu().item())
        stats_accum["ts"] += float(ts.detach().cpu().item())
        stats_accum["src_cons"] += float(src_cons.detach().cpu().item())

    with torch.no_grad():
        refined = model(x)
    del optimizer
    del model
    return refined, {key: value / float(step_count) for key, value in stats_accum.items()}


def _class_centroid(mask: torch.Tensor) -> Optional[torch.Tensor]:
    if mask.sum() <= 0:
        return None
    ys, xs = torch.where(mask > 0)
    return torch.stack([ys.float().mean(), xs.float().mean()])


def cardiac_geometry_score(candidate_logits: torch.Tensor, source_logits: torch.Tensor) -> float:
    """Compare candidate cardiac anatomy to source-model anatomy without labels.

    The score is intentionally conservative and relative: it rewards preserving
    source-predicted class presence, class area ratios, and centroids for
    LV/MYO/RV-like foreground classes. It is not a hard shape template.
    """
    cand = candidate_logits.argmax(dim=1)
    src = source_logits.argmax(dim=1)
    batch_scores: List[torch.Tensor] = []

    for b in range(cand.shape[0]):
        h, w = cand.shape[-2:]
        hw = float(max(1, h * w))
        per_class = []
        for cls in range(1, candidate_logits.shape[1]):
            cm = cand[b] == cls
            sm = src[b] == cls
            c_area = cm.float().sum() / hw
            s_area = sm.float().sum() / hw
            if c_area <= 1e-7 and s_area <= 1e-7:
                area_score = cand.new_tensor(1.0, dtype=torch.float32)
            elif c_area <= 1e-7 or s_area <= 1e-7:
                area_score = cand.new_tensor(0.0, dtype=torch.float32)
            else:
                area_score = torch.minimum(c_area / s_area, s_area / c_area).clamp(0.0, 1.0)

            c_ctr = _class_centroid(cm)
            s_ctr = _class_centroid(sm)
            if c_ctr is None and s_ctr is None:
                center_score = cand.new_tensor(1.0, dtype=torch.float32)
            elif c_ctr is None or s_ctr is None:
                center_score = cand.new_tensor(0.0, dtype=torch.float32)
            else:
                dist = torch.norm(c_ctr - s_ctr) / math.sqrt(hw)
                center_score = (1.0 - dist).clamp(0.0, 1.0)
            per_class.append(0.60 * area_score + 0.40 * center_score)

        fg_c = cand[b] > 0
        fg_s = src[b] > 0
        fg_area_c = fg_c.float().sum() / hw
        fg_area_s = fg_s.float().sum() / hw
        if fg_area_s > 1e-7 and fg_area_c > 1e-7:
            fg_ratio = torch.minimum(fg_area_c / fg_area_s, fg_area_s / fg_area_c).clamp(0.0, 1.0)
        elif fg_area_s <= 1e-7 and fg_area_c <= 1e-7:
            fg_ratio = cand.new_tensor(1.0, dtype=torch.float32)
        else:
            fg_ratio = cand.new_tensor(0.0, dtype=torch.float32)
        per_class.append(fg_ratio)
        batch_scores.append(torch.stack(per_class).mean())

    return float(torch.stack(batch_scores).mean().detach().cpu().item())


def class_area_mean(logits: torch.Tensor) -> List[float]:
    pred = logits.argmax(dim=1)
    hw = float(max(1, pred.shape[-2] * pred.shape[-1]))
    return [float(((pred == cls).float().sum(dim=(1, 2)) / hw).mean().detach().cpu().item()) for cls in range(1, logits.shape[1])]


def foreground_area_mean(logits: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    hw = float(max(1, pred.shape[-2] * pred.shape[-1]))
    return float(((pred > 0).float().sum(dim=(1, 2)) / hw).mean().detach().cpu().item())


def reliability_info(logits: torch.Tensor, source_logits: torch.Tensor, name: str) -> Dict[str, float]:
    ent = mean_normalized_entropy(logits)
    conf = mean_confidence(logits)
    drift = probability_drift(logits, source_logits)
    geom = cardiac_geometry_score(logits, source_logits)
    fg_area = foreground_area_mean(logits)
    areas = class_area_mean(logits)

    # Higher is better. Scales are chosen so no single cue dominates branch
    # selection across all batches.
    score = (2.0 * conf) + (2.5 * geom) - (2.0 * ent) - (1.0 * drift)
    if fg_area < 1e-5:
        score -= 2.0
    if fg_area > 0.45:
        score -= 1.0

    return {
        "branch": name,
        "score": float(score),
        "entropy": ent,
        "confidence": conf,
        "drift": drift,
        "geometry": geom,
        "fg_area": fg_area,
        "area_c1": areas[0],
        "area_c2": areas[1],
        "area_c3": areas[2],
    }


def is_risky_update(
    source_info: Dict[str, float],
    adapted_info: Dict[str, float],
    entropy_margin: float,
    drift_threshold: float,
    geometry_margin: float,
) -> bool:
    if adapted_info["entropy"] > source_info["entropy"] + float(entropy_margin):
        return True
    if adapted_info["drift"] > float(drift_threshold):
        return True
    if adapted_info["geometry"] < source_info["geometry"] - float(geometry_margin):
        return True
    if adapted_info["fg_area"] < 1e-5 and source_info["fg_area"] > 1e-5:
        return True
    if adapted_info["fg_area"] > max(0.45, source_info["fg_area"] * 3.0 + 0.02):
        return True
    return False


def select_branch(
    source_info: Dict[str, float],
    adapted_info: Dict[str, float],
    final_info: Dict[str, float],
    risky: bool,
    allow_source_output: bool,
    final_bias: float,
    final_min_gain: float,
    source_fallback_margin: float,
) -> Tuple[str, str]:
    if risky and allow_source_output:
        return "source", "risky_update"

    # Block fused final if fusion clearly worsens anatomy or confidence relative
    # to the raw adapted branch.
    final_blocked = (
        final_info["geometry"] < adapted_info["geometry"] - 0.03
        or final_info["confidence"] < adapted_info["confidence"] - 0.04
        or final_info["fg_area"] < adapted_info["fg_area"] * 0.35
    )

    if not final_blocked:
        if final_info["score"] + float(final_bias) >= adapted_info["score"] + float(final_min_gain):
            return "final", "final_safe"
        return "adapted", "adapted_beats_final"

    # Source is only a fallback. Its geometry cue is self-referential in cardiac
    # MRI, so require adapted to be meaningfully worse before selecting source.
    if adapted_info["score"] >= source_info["score"] - float(source_fallback_margin):
        return "adapted", "final_blocked_adapted_safe"
    if allow_source_output:
        return "source", "final_blocked_source_fallback"
    return "adapted", "final_blocked_source_disabled"


def snapshot_a3_state(model) -> Dict[str, object]:
    adapter_name = str(getattr(model, "adapter_name", "a3-full"))
    if adapter_name == "source":
        return {"adapter_name": adapter_name}
    optimizer = getattr(model, "optimizer", None)
    snap = {
        "adapter_name": adapter_name,
        "model": copy.deepcopy(model.model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None,
        "model_ema": (
            copy.deepcopy(model.model_ema.state_dict())
            if getattr(model, "model_ema", None) is not None
            else None
        ),
        "pool_memory": copy.deepcopy(getattr(getattr(model, "pool", None), "memory", None)),
        "entropy_list": copy.deepcopy(getattr(model, "entropy_list", None)),
        "score_history": copy.deepcopy(getattr(model, "score_history", None)),
        "counters": {
            "total_seen": getattr(model, "total_seen", None),
            "total_consistency_reject": getattr(model, "total_consistency_reject", None),
            "total_candidate": getattr(model, "total_candidate", None),
            "total_score": getattr(model, "total_score", None),
            "mt": getattr(model, "mt", None),
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    return snap


def restore_a3_state(
    model,
    snap: Dict[str, object],
    *,
    restore_rng: bool = True,
) -> None:
    if snap.get("adapter_name") == "source":
        return
    model.model.load_state_dict(snap["model"], strict=True)
    if snap.get("optimizer") is not None and getattr(model, "optimizer", None) is not None:
        model.optimizer.load_state_dict(snap["optimizer"])
    if snap.get("model_ema") is not None and getattr(model, "model_ema", None) is not None:
        model.model_ema.load_state_dict(snap["model_ema"], strict=True)
    if snap.get("pool_memory") is not None and hasattr(model, "pool"):
        model.pool.memory = copy.deepcopy(snap["pool_memory"])
    if snap.get("entropy_list") is not None:
        model.entropy_list = copy.deepcopy(snap["entropy_list"])
    if snap.get("score_history") is not None:
        model.score_history = copy.deepcopy(snap["score_history"])
    for key, value in snap.get("counters", {}).items():
        if value is not None:
            setattr(model, key, value)
    rng = snap.get("rng")
    if restore_rng and rng is not None:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])


class RPRCheckpointController:
    """Keep a lagged trusted state for checkpointed RPR.

    The interval is fitted only on Source-A. Candidate states remain live for
    adaptation, but they are committed as a trusted recovery point only after
    ``interval`` consecutive safe updates. A risk event discards the current
    update and every still-uncommitted safe update.
    """

    def __init__(self, interval: int):
        interval = int(interval)
        if interval < 2:
            raise ValueError("Checkpointed RPR requires an interval of at least 2")
        self.interval = interval
        self.trusted_state: Optional[Dict[str, object]] = None
        self.safe_streak = 0
        self.commit_count = 0
        self.rollback_count = 0
        self.rollback_span_total = 0

    def ensure_initialized(self, state: Dict[str, object]) -> None:
        if self.trusted_state is None:
            self.trusted_state = state

    def accept(self, state: Dict[str, object]) -> bool:
        self.ensure_initialized(state)
        self.safe_streak += 1
        if self.safe_streak < self.interval:
            return False
        self.trusted_state = state
        self.safe_streak = 0
        self.commit_count += 1
        return True

    def reject(self, pre_update_state: Dict[str, object]) -> Tuple[Dict[str, object], int]:
        self.ensure_initialized(pre_update_state)
        if self.trusted_state is None:
            raise RuntimeError("RPR checkpoint controller was not initialized")
        rollback_span = self.safe_streak + 1
        self.safe_streak = 0
        self.rollback_count += 1
        self.rollback_span_total += rollback_span
        return self.trusted_state, rollback_span


def load_config(cfg_file: str, opts: List[str]) -> None:
    reset_cfg()
    merge_from_file(cfg_file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def parse_float_list(v: Optional[str]) -> Optional[List[float]]:
    if v is None or str(v).strip() == "":
        return None
    return [float(x.strip()) for x in str(v).split(",") if x.strip()]


def parse_string_list(v: Optional[str]) -> List[str]:
    if v is None or str(v).strip() == "":
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


def parse_int_list(v: Optional[str]) -> List[int]:
    if v is None or str(v).strip() == "":
        return []
    return [int(x.strip()) for x in str(v).split(",") if x.strip()]


def parse_name_float_map(v: Optional[str]) -> Dict[str, float]:
    if v is None or str(v).strip() == "":
        return {}
    values: Dict[str, float] = {}
    for item in str(v).split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Expected name:value in candidate_bank_name_bias, got {token!r}")
        name, value = token.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty candidate name in candidate_bank_name_bias item {token!r}")
        values[name] = float(value.strip())
    return values


def candidate_bank_adjusted_score(name: str, info: Dict[str, float], name_bias: Optional[Dict[str, float]]) -> float:
    return float(info["score"]) + float((name_bias or {}).get(name, 0.0))


def guard_safe_ema_candidate(
    selected_name: str,
    candidate_infos: Dict[str, Dict[str, float]],
    name_bias: Optional[Dict[str, float]],
    min_gain: float,
) -> Tuple[str, float]:
    references = {
        "ema": "adapted",
        "ema_refined": "final",
        "ema_final": "final",
        "clean_ema_final": "clean_final",
    }
    selected_score = candidate_bank_adjusted_score(selected_name, candidate_infos[selected_name], name_bias)
    reference_name = references.get(selected_name)
    if reference_name is None or reference_name not in candidate_infos:
        return selected_name, selected_score
    reference_score = candidate_bank_adjusted_score(reference_name, candidate_infos[reference_name], name_bias)
    if selected_score < reference_score + float(min_gain):
        return reference_name, reference_score
    return selected_name, selected_score


def load_source_calibrated_mre(path: Optional[str]) -> Optional[Dict[str, object]]:
    if path is None or str(path).strip() == "":
        return None
    model_path = Path(path).expanduser().resolve()
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    required = {
        "baseline_candidate",
        "candidate_names",
        "continuous_features",
        "candidate_features",
        "feature_mean",
        "feature_scale",
        "coefficients",
        "intercept",
        "selection_threshold",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"MRE calibrator {model_path} is missing fields: {missing}")
    feature_count = len(payload["continuous_features"]) + len(payload["candidate_features"])
    for key in ("feature_mean", "feature_scale", "coefficients"):
        if len(payload[key]) != feature_count:
            raise ValueError(
                f"MRE calibrator {model_path} has {len(payload[key])} values in {key}; "
                f"expected {feature_count}"
            )
    payload["path"] = str(model_path)
    return payload


def load_source_calibrated_rpr(path: Optional[str]) -> Optional[Dict[str, object]]:
    if path is None or str(path).strip() == "":
        return None
    model_path = Path(path).expanduser().resolve()
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    required = {"method", "feature_names", "source"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"RPR calibrator {model_path} is missing fields: {missing}")
    if list(payload["feature_names"]) != list(PREUPDATE_RPR_FEATURES):
        raise ValueError(
            f"RPR calibrator {model_path} uses feature_names={payload['feature_names']!r}; "
            f"expected {list(PREUPDATE_RPR_FEATURES)!r}"
        )
    feature_count = len(PREUPDATE_RPR_FEATURES)
    method = str(payload["method"]).strip().lower()
    if method == "source_a_preupdate_logistic_rpr":
        logistic_required = {
            "feature_mean",
            "feature_scale",
            "coefficients",
            "intercept",
            "decision_threshold",
        }
        missing = sorted(logistic_required.difference(payload))
        if missing:
            raise ValueError(f"Logistic RPR calibrator {model_path} is missing fields: {missing}")
        for key in ("feature_mean", "feature_scale", "coefficients"):
            if len(payload[key]) != feature_count:
                raise ValueError(
                    f"RPR calibrator {model_path} has {len(payload[key])} values in {key}; "
                    f"expected {feature_count}"
                )
    elif method == "source_a_preupdate_quantile_rpr":
        quantile_required = {"feature_thresholds", "feature_groups", "vote_threshold"}
        missing = sorted(quantile_required.difference(payload))
        if missing:
            raise ValueError(f"Quantile RPR calibrator {model_path} is missing fields: {missing}")
        if len(payload["feature_thresholds"]) != feature_count:
            raise ValueError(
                f"RPR calibrator {model_path} has {len(payload['feature_thresholds'])} thresholds; "
                f"expected {feature_count}"
            )
        valid_features = set(PREUPDATE_RPR_FEATURES)
        for group_name, group_features in payload["feature_groups"].items():
            unknown = sorted(set(group_features).difference(valid_features))
            if unknown:
                raise ValueError(
                    f"RPR calibrator group {group_name!r} uses unknown features: {unknown}"
                )
    else:
        raise ValueError(f"Unsupported RPR calibrator method: {payload['method']!r}")
    if str(payload["source"]).strip().lower() not in ("source-a", "domain-a"):
        raise ValueError("RPR calibrator must be fitted exclusively on Source-A")
    payload["path"] = str(model_path)
    return payload


def _finite_info_value(info: Dict[str, float], key: str) -> float:
    value = info.get(key, 0.0)
    if value is None:
        return 0.0
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _safe_ratio_log(numerator: float, denominator: float) -> float:
    eps = 1e-8
    return abs(math.log((max(0.0, float(numerator)) + eps) / (max(0.0, float(denominator)) + eps)))


def preupdate_rpr_evidence(
    pre_info: Dict[str, float],
    adapted_info: Dict[str, float],
    pre_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
) -> Dict[str, object]:
    pre_center = pre_info.get("center_dist")
    adapted_center = adapted_info.get("center_dist")
    center_increase = 0.0
    if pre_center is not None and adapted_center is not None:
        center_increase = float(adapted_center) - float(pre_center)

    pre_areas = class_area_mean(pre_logits)
    adapted_areas = class_area_mean(adapted_logits)
    collapse_flags = [
        float(pre_area > 0.001 and adapted_area < 0.20 * pre_area)
        for pre_area, adapted_area in zip(pre_areas, adapted_areas)
    ]
    class_collapse_fraction = float(np.mean(collapse_flags)) if collapse_flags else 0.0

    feature_map = {
        "entropy_increase": _finite_info_value(adapted_info, "entropy") - _finite_info_value(pre_info, "entropy"),
        "prior_align_drop": _finite_info_value(pre_info, "prior_align_mean")
        - _finite_info_value(adapted_info, "prior_align_mean"),
        "center_distance_increase": float(center_increase),
        "topology_drop": _finite_info_value(pre_info, "topology_score")
        - _finite_info_value(adapted_info, "topology_score"),
        "confidence_drop": _finite_info_value(pre_info, "confidence")
        - _finite_info_value(adapted_info, "confidence"),
        "pre_adapt_drift": probability_drift(adapted_logits, pre_logits),
        "fg_log_ratio_abs": _safe_ratio_log(
            _finite_info_value(adapted_info, "fg_area"),
            _finite_info_value(pre_info, "fg_area"),
        ),
        "rv_soft_area_drop": _finite_info_value(pre_info, "rv_soft_area")
        - _finite_info_value(adapted_info, "rv_soft_area"),
        "rv_pred_area_drop": _finite_info_value(pre_info, "rv_pred_area")
        - _finite_info_value(adapted_info, "rv_pred_area"),
        "rv_confidence_drop": _finite_info_value(pre_info, "rv_confidence")
        - _finite_info_value(adapted_info, "rv_confidence"),
        "class_collapse_fraction": class_collapse_fraction,
    }
    return {
        "features": feature_map,
        "pre_class_areas": pre_areas,
        "adapted_class_areas": adapted_areas,
    }


def decide_preupdate_rpr(
    evidence: Dict[str, object],
    *,
    calibrator: Optional[Dict[str, object]],
    vote_threshold: int,
    entropy_margin: float,
    prior_align_margin: float,
    center_margin: float,
    topology_margin: float,
    confidence_margin: float,
    drift_threshold: float,
    fg_min_ratio: float,
    fg_max_ratio: float,
) -> Dict[str, object]:
    features = evidence["features"]
    pre_areas = evidence["pre_class_areas"]
    adapted_areas = evidence["adapted_class_areas"]
    pre_fg = float(sum(pre_areas))
    adapted_fg = float(sum(adapted_areas))

    hard_reasons: List[str] = []
    if pre_fg > 0.002 and adapted_fg < float(fg_min_ratio) * pre_fg:
        hard_reasons.append("foreground_collapse")
    if adapted_fg > 0.45 or (pre_fg > 0.002 and adapted_fg > float(fg_max_ratio) * pre_fg + 0.02):
        hard_reasons.append("foreground_expansion")
    if float(features["class_collapse_fraction"]) > 0.0:
        hard_reasons.append("class_collapse")

    vote_reasons = []
    checks = (
        ("entropy", float(features["entropy_increase"]) > float(entropy_margin)),
        ("prior_align", float(features["prior_align_drop"]) > float(prior_align_margin)),
        ("center", float(features["center_distance_increase"]) > float(center_margin)),
        ("topology", float(features["topology_drop"]) > float(topology_margin)),
        ("confidence", float(features["confidence_drop"]) > float(confidence_margin)),
        ("drift", float(features["pre_adapt_drift"]) > float(drift_threshold)),
    )
    vote_reasons.extend(name for name, triggered in checks if triggered)

    calibrated_probability = None
    calibrated_trigger = False
    calibrated_reasons: List[str] = []
    if calibrator is not None:
        method = str(calibrator["method"]).strip().lower()
        if method == "source_a_preupdate_quantile_rpr":
            thresholds = np.asarray(calibrator["feature_thresholds"], dtype=np.float64)
            x = np.asarray([float(features[name]) for name in PREUPDATE_RPR_FEATURES], dtype=np.float64)
            exceeded_features = [
                name
                for name, value, threshold in zip(PREUPDATE_RPR_FEATURES, x, thresholds)
                if value > threshold
            ]
            calibrated_reasons = [
                f"{group_name}({','.join(name for name in group_features if name in exceeded_features)})"
                for group_name, group_features in calibrator["feature_groups"].items()
                if any(name in exceeded_features for name in group_features)
            ]
            calibrated_probability = float(len(calibrated_reasons))
            calibrated_trigger = len(calibrated_reasons) >= int(calibrator["vote_threshold"])
        else:
            x = np.asarray([float(features[name]) for name in PREUPDATE_RPR_FEATURES], dtype=np.float64)
            mean = np.asarray(calibrator["feature_mean"], dtype=np.float64)
            scale = np.asarray(calibrator["feature_scale"], dtype=np.float64)
            coefficients = np.asarray(calibrator["coefficients"], dtype=np.float64)
            scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
            logit = float(calibrator["intercept"]) + float(np.dot((x - mean) / scale, coefficients))
            calibrated_probability = float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit)))))
            calibrated_trigger = calibrated_probability >= float(calibrator["decision_threshold"])

    risky = bool(
        hard_reasons
        or calibrated_trigger
        or (calibrator is None and len(vote_reasons) >= int(vote_threshold))
    )
    if hard_reasons:
        decision_source = "hard_guard"
    elif calibrator is not None:
        decision_source = str(calibrator["method"])
    else:
        decision_source = "predeclared_vote"
    return {
        "risky": risky,
        "decision_source": decision_source,
        "decision_score": calibrated_probability,
        "vote_count": len(vote_reasons),
        "vote_reasons": vote_reasons,
        "calibrated_reasons": calibrated_reasons,
        "hard_reasons": hard_reasons,
    }


def source_calibrated_mre_score(
    name: str,
    info: Dict[str, float],
    baseline_info: Dict[str, float],
    calibrator: Dict[str, object],
) -> float:
    features = [
        _finite_info_value(info, key) - _finite_info_value(baseline_info, key)
        for key in calibrator["continuous_features"]
    ]
    features.extend(float(name == candidate_name) for candidate_name in calibrator["candidate_features"])
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(calibrator["feature_mean"], dtype=np.float64)
    scale = np.asarray(calibrator["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(calibrator["coefficients"], dtype=np.float64)
    scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
    return float(calibrator["intercept"]) + float(np.dot((x - mean) / scale, coefficients))


def select_source_calibrated_candidate(
    candidate_logits: Dict[str, torch.Tensor],
    candidate_infos: Dict[str, Dict[str, float]],
    calibrator: Dict[str, object],
) -> Tuple[str, torch.Tensor, Dict[str, float], str]:
    baseline_name = str(calibrator["baseline_candidate"])
    if baseline_name not in candidate_logits or baseline_name not in candidate_infos:
        fallback = "final" if "final" in candidate_logits else next(iter(candidate_logits))
        return (
            fallback,
            candidate_logits[fallback],
            candidate_infos[fallback],
            f"source_calibrated_mre_missing_baseline={baseline_name}",
        )

    allowed = [str(name) for name in calibrator["candidate_names"]]
    pool = [name for name in allowed if name in candidate_logits and name in candidate_infos]
    if baseline_name not in pool:
        pool.insert(0, baseline_name)
    baseline_info = candidate_infos[baseline_name]
    predicted_gain = {baseline_name: 0.0}
    for name in pool:
        if name == baseline_name:
            continue
        predicted_gain[name] = source_calibrated_mre_score(
            name,
            candidate_infos[name],
            baseline_info,
            calibrator,
        )

    selected = max(predicted_gain, key=predicted_gain.get)
    threshold = float(calibrator["selection_threshold"])
    if selected != baseline_name and predicted_gain[selected] < threshold:
        selected = baseline_name
    score_text = {name: round(score, 5) for name, score in predicted_gain.items()}
    reason = (
        f"source_calibrated_mre={selected},threshold={threshold:.5f},"
        f"predicted_dice_gain={score_text}"
    )
    return selected, candidate_logits[selected], candidate_infos[selected], reason


def load_source_nonlinear_rv_selector(path: Optional[str]) -> Optional[Dict[str, object]]:
    if path is None or str(path).strip() == "":
        return None
    model_path = Path(path).expanduser().resolve()
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    required = {
        "model_type",
        "baseline_candidate",
        "candidate_names",
        "feature_spec",
        "initial_prediction",
        "learning_rate",
        "trees",
        "selection_threshold",
        "source",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"RV selector {model_path} is missing fields: {missing}")
    if payload["model_type"] != "source_rv_gradient_boosting_v1":
        raise ValueError(f"Unsupported RV selector type: {payload['model_type']!r}")
    if payload.get("source", {}).get("domain") != "A":
        raise ValueError("RV selector must be calibrated exclusively on Source-A")
    feature_count = len(payload["feature_spec"])
    for tree in payload["trees"]:
        nodes = tree.get("nodes", [])
        if not nodes:
            raise ValueError(f"RV selector {model_path} contains an empty tree")
        for node in nodes:
            feature = int(node.get("feature", -2))
            if feature >= feature_count:
                raise ValueError(f"RV selector tree feature {feature} exceeds {feature_count}")
    payload["path"] = str(model_path)
    return payload


def _rv_selector_features(
    name: str,
    info: Dict[str, float],
    baseline_info: Dict[str, float],
    feature_spec: Iterable[Dict[str, object]],
) -> np.ndarray:
    values: List[float] = []
    for spec in feature_spec:
        kind = str(spec.get("kind", ""))
        key = str(spec.get("key", ""))
        if kind == "delta":
            value = _finite_info_value(info, key) - _finite_info_value(baseline_info, key)
        elif kind == "candidate":
            value = _finite_info_value(info, key)
        elif kind == "baseline":
            value = _finite_info_value(baseline_info, key)
        elif kind == "ratio":
            numerator = _finite_info_value(info, key)
            denominator = _finite_info_value(baseline_info, key)
            value = numerator / denominator if abs(denominator) > 1e-8 else 1.0
            value = max(-10.0, min(10.0, value))
        elif kind == "identity":
            value = float(name == str(spec.get("name", "")))
        else:
            raise ValueError(f"Unknown RV selector feature kind: {kind!r}")
        values.append(float(value) if math.isfinite(float(value)) else 0.0)
    return np.asarray(values, dtype=np.float64)


def _evaluate_exported_tree(tree: Dict[str, object], features: np.ndarray) -> float:
    nodes = tree["nodes"]
    index = 0
    while True:
        node = nodes[index]
        left = int(node["left"])
        right = int(node["right"])
        feature = int(node["feature"])
        if left < 0 or right < 0 or feature < 0:
            return float(node["value"])
        index = left if float(features[feature]) <= float(node["threshold"]) else right


def source_nonlinear_rv_gain(
    name: str,
    info: Dict[str, float],
    baseline_info: Dict[str, float],
    selector: Dict[str, object],
) -> float:
    features = _rv_selector_features(name, info, baseline_info, selector["feature_spec"])
    prediction = float(selector["initial_prediction"])
    learning_rate = float(selector["learning_rate"])
    for tree in selector["trees"]:
        prediction += learning_rate * _evaluate_exported_tree(tree, features)
    return prediction


def select_source_nonlinear_rv_candidate(
    candidate_logits: Dict[str, torch.Tensor],
    candidate_infos: Dict[str, Dict[str, float]],
    selector: Dict[str, object],
) -> Tuple[str, torch.Tensor, Dict[str, float], str]:
    baseline_name = str(selector["baseline_candidate"])
    if baseline_name not in candidate_logits or baseline_name not in candidate_infos:
        fallback = "final" if "final" in candidate_logits else next(iter(candidate_logits))
        return (
            fallback,
            candidate_logits[fallback],
            candidate_infos[fallback],
            f"source_nonlinear_rv_missing_baseline={baseline_name}",
        )
    baseline_info = candidate_infos[baseline_name]
    predicted_gain = {baseline_name: 0.0}
    for name in selector["candidate_names"]:
        name = str(name)
        if name == baseline_name or name not in candidate_logits or name not in candidate_infos:
            continue
        predicted_gain[name] = source_nonlinear_rv_gain(
            name,
            candidate_infos[name],
            baseline_info,
            selector,
        )
    selected = max(predicted_gain, key=predicted_gain.get)
    threshold = float(selector["selection_threshold"])
    if selected != baseline_name and predicted_gain[selected] < threshold:
        selected = baseline_name
    gains_text = {name: round(value, 4) for name, value in predicted_gain.items()}
    reason = (
        f"source_nonlinear_rv={selected},threshold_points={threshold:.4f},"
        f"predicted_rv_gain_points={gains_text}"
    )
    return selected, candidate_logits[selected], candidate_infos[selected], reason


def _batched_prior_tensor(value: torch.Tensor, batch: int) -> torch.Tensor:
    if value.shape[0] == 1 and batch > 1:
        return value.repeat(batch, *([1] * (value.dim() - 1)))
    return value


def _binary_center_score(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    if bool((mask_a.sum() <= 0).detach().cpu().item()) and bool((mask_b.sum() <= 0).detach().cpu().item()):
        return mask_a.new_tensor(1.0, dtype=torch.float32)
    if bool((mask_a.sum() <= 0).detach().cpu().item()) or bool((mask_b.sum() <= 0).detach().cpu().item()):
        return mask_a.new_tensor(0.0, dtype=torch.float32)
    ctr_a = _class_centroid(mask_a)
    ctr_b = _class_centroid(mask_b)
    if ctr_a is None or ctr_b is None:
        return mask_a.new_tensor(0.0, dtype=torch.float32)
    h, w = mask_a.shape[-2:]
    dist = torch.norm(ctr_a - ctr_b) / math.sqrt(float(max(1, h * w)))
    return (1.0 - 3.0 * dist).clamp(0.0, 1.0)


def classwise_candidate_info(
    logits: torch.Tensor,
    source_logits: torch.Tensor,
    prior: Dict[str, torch.Tensor],
    cls_idx: int,
    name: str,
) -> Dict[str, float]:
    prob = torch.softmax(logits, dim=1).clamp(1e-6, 1.0 - 1e-6)
    source_prob = torch.softmax(source_logits, dim=1).clamp(1e-6, 1.0 - 1e-6)
    pred = logits.argmax(dim=1)
    src_pred = source_logits.argmax(dim=1)
    cls_prob = prob[:, cls_idx]
    src_cls_prob = source_prob[:, cls_idx]
    cls_mask = pred == cls_idx
    src_mask = src_pred == cls_idx
    batch = int(logits.shape[0])
    prior_prob = _batched_prior_tensor(prior["prob"], batch).to(device=logits.device, dtype=prob.dtype)
    prior_cls = prior_prob[:, cls_idx].clamp(0.0, 1.0)

    cls_area = cls_mask.float().mean(dim=(1, 2))
    src_area = src_mask.float().mean(dim=(1, 2))
    eps = logits.new_tensor(1e-7)
    both_absent = (cls_area <= eps) & (src_area <= eps)
    both_present = (cls_area > eps) & (src_area > eps)
    area_ratio = torch.minimum(cls_area / src_area.clamp_min(1e-7), src_area / cls_area.clamp_min(1e-7)).clamp(0.0, 1.0)
    area_score = torch.where(both_absent, torch.ones_like(area_ratio), torch.where(both_present, area_ratio, torch.zeros_like(area_ratio)))

    center_scores = []
    for bi in range(batch):
        center_scores.append(_binary_center_score(cls_mask[bi].float(), src_mask[bi].float()))
    center_score = torch.stack(center_scores).mean() if center_scores else logits.new_tensor(0.0)

    mask_count = cls_mask.float().sum(dim=(1, 2))
    masked_conf = (cls_prob * cls_mask.float()).sum(dim=(1, 2)) / mask_count.clamp_min(1.0)
    peak_conf = cls_prob.amax(dim=(1, 2))
    confidence = torch.where(mask_count > 0, masked_conf, 0.35 * peak_conf).mean()

    entropy = -(
        cls_prob * cls_prob.log() + (1.0 - cls_prob) * (1.0 - cls_prob).log()
    ) / math.log(2.0)
    prior_weight = prior_cls.clamp(0.05, 1.0)
    entropy_mean = (entropy * prior_weight).sum(dim=(1, 2)) / prior_weight.sum(dim=(1, 2)).clamp_min(1e-6)
    entropy_v = entropy_mean.mean()

    prior_align = (cls_prob * prior_cls).mean(dim=(1, 2)) - 0.50 * (cls_prob * (1.0 - prior_cls)).mean(dim=(1, 2))
    prior_align_v = prior_align.mean()
    class_drift = torch.abs(cls_prob - src_cls_prob).mean()

    score = (
        1.20 * area_score.mean()
        + 0.85 * center_score
        + 1.10 * confidence
        + 8.00 * prior_align_v
        - 0.35 * entropy_v
        - 0.70 * class_drift
    )
    return {
        "branch": name,
        "class": int(cls_idx),
        "score": float(score.detach().cpu().item()),
        "area_score": float(area_score.mean().detach().cpu().item()),
        "center_score": float(center_score.detach().cpu().item()),
        "confidence": float(confidence.detach().cpu().item()),
        "prior_align": float(prior_align_v.detach().cpu().item()),
        "entropy": float(entropy_v.detach().cpu().item()),
        "class_drift": float(class_drift.detach().cpu().item()),
        "pred_area": float(cls_area.mean().detach().cpu().item()),
        "source_area": float(src_area.mean().detach().cpu().item()),
    }


def compose_classwise_candidate_logits(
    base_logits: torch.Tensor,
    candidates: Dict[str, torch.Tensor],
    selected: Dict[int, str],
    *,
    compose: str,
    logit_boost: float,
) -> torch.Tensor:
    mode = str(compose or "logit").strip().lower()
    if mode == "mask":
        out = base_logits.new_full(base_logits.shape, -8.0)
        out[:, 0] = 4.0
        for cls_idx, cand_name in selected.items():
            cand = candidates[cand_name]
            cand_prob = torch.softmax(cand, dim=1)[:, cls_idx]
            cand_mask = cand.argmax(dim=1) == int(cls_idx)
            out[:, int(cls_idx)] = torch.where(cand_mask, 4.0 + cand_prob + float(logit_boost), out[:, int(cls_idx)])
            out[:, 0] = torch.where(cand_mask, out[:, 0].new_tensor(-4.0), out[:, 0])
        return out

    out = base_logits.clone()
    for cls_idx, cand_name in selected.items():
        out[:, int(cls_idx)] = candidates[cand_name][:, int(cls_idx)] + float(logit_boost)
    return out


@torch.no_grad()
def compose_rv_margin_candidate(
    base_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    rv_class: int = 3,
) -> torch.Tensor:
    """Keep base non-RV decisions while reproducing the donor's RV hard mask.

    The donor RV-vs-best-non-RV margin is transferred into the base logit
    system. Consequently, RV wins at exactly the donor's RV pixels, while the
    relative LV/MYO/BG ordering remains that of the guarded base candidate.
    """
    if base_logits.shape != donor_logits.shape:
        raise ValueError(
            f"RV composition shape mismatch: base={tuple(base_logits.shape)}, "
            f"donor={tuple(donor_logits.shape)}"
        )
    class_count = int(base_logits.shape[1])
    rv_class = int(rv_class)
    if rv_class <= 0 or rv_class >= class_count:
        return base_logits
    non_rv = [idx for idx in range(class_count) if idx != rv_class]
    donor_non_rv = donor_logits[:, non_rv].amax(dim=1)
    base_non_rv = base_logits[:, non_rv].amax(dim=1)
    donor_margin = donor_logits[:, rv_class] - donor_non_rv
    out = base_logits.clone()
    out[:, rv_class] = base_non_rv + donor_margin
    return out


@torch.no_grad()
def compose_consensus_rv_margin_candidate(
    base_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    donor_logits: Dict[str, torch.Tensor],
    prior_prob: Optional[torch.Tensor],
    *,
    rv_class: int = 3,
    min_agreement: int = 3,
    donor_confidence: float = 0.50,
    base_margin_floor: float = -0.35,
    prior_threshold: float = 0.10,
    protect_confidence: float = 0.72,
    support_kernel: int = 11,
    max_extra_ratio: float = 0.35,
    strength: float = 0.80,
) -> torch.Tensor:
    """Add RV only where several independent APSA candidates agree.

    ``base_logits`` provides the guarded LV/MYO/BG decisions.  The softer
    ``reference_logits`` is used for uncertainty gating because cleaned logits
    are intentionally near one-hot and therefore unsuitable for calibration.
    Existing base RV pixels are never removed.
    """
    if base_logits.shape != reference_logits.shape or base_logits.ndim != 4:
        return base_logits
    class_count = int(base_logits.shape[1])
    rv_class = int(rv_class)
    if rv_class <= 0 or rv_class >= class_count:
        return base_logits

    donors = [
        logits
        for logits in donor_logits.values()
        if logits.shape == base_logits.shape
    ]
    required = max(1, int(min_agreement))
    if len(donors) < required:
        return base_logits

    non_rv = [idx for idx in range(class_count) if idx != rv_class]
    donor_votes = []
    donor_positive_margins = []
    conf_threshold = max(0.0, min(1.0, float(donor_confidence)))
    for logits in donors:
        prob = torch.softmax(logits, dim=1)
        non_rv_logit = logits[:, non_rv].amax(dim=1)
        margin = logits[:, rv_class] - non_rv_logit
        vote = (logits.argmax(dim=1) == rv_class) & (prob[:, rv_class] >= conf_threshold)
        donor_votes.append(vote)
        donor_positive_margins.append(torch.where(vote, margin.clamp_min(0.0), torch.zeros_like(margin)))

    votes = torch.stack(donor_votes, dim=0)
    vote_count = votes.sum(dim=0)
    positive_margin = torch.stack(donor_positive_margins, dim=0).sum(dim=0)
    positive_margin = positive_margin / vote_count.clamp_min(1).to(positive_margin.dtype)

    reference_prob = torch.softmax(reference_logits, dim=1)
    reference_non_rv = reference_logits[:, non_rv].amax(dim=1)
    reference_margin = reference_logits[:, rv_class] - reference_non_rv
    base_pred = base_logits.argmax(dim=1)

    if prior_prob is not None and prior_prob.shape == reference_prob.shape:
        structural_prob = prior_prob.to(device=reference_prob.device, dtype=reference_prob.dtype)
    else:
        structural_prob = reference_prob
    cardiac_seed = torch.maximum(
        reference_prob[:, 1:3].sum(dim=1, keepdim=True),
        structural_prob[:, 1:3].sum(dim=1, keepdim=True),
    ).clamp(0.0, 1.0)
    cardiac_support = _soft_dilate(cardiac_seed, int(support_kernel))[:, 0] >= 0.10
    prior_rv = structural_prob[:, rv_class]
    stronger_consensus = vote_count >= min(len(donors), required + 1)
    anatomy_support = cardiac_support & (
        (prior_rv >= float(prior_threshold)) | stronger_consensus
    )

    protected_structure = reference_prob[:, 1:3].amax(dim=1) >= float(protect_confidence)
    eligible = (
        (vote_count >= required)
        & (base_pred != rv_class)
        & (reference_margin >= float(base_margin_floor))
        & (~protected_structure)
        & anatomy_support
    )

    # Limit per-slice expansion relative to the existing RV or LV extent.
    selected = torch.zeros_like(eligible)
    agreement_score = vote_count.to(reference_prob.dtype) / float(len(donors))
    rank_score = agreement_score + torch.sigmoid(positive_margin) + 0.5 * prior_rv
    base_rv = base_pred == rv_class
    base_lv = base_pred == 1
    for sample_idx in range(int(base_logits.shape[0])):
        eligible_flat = eligible[sample_idx].reshape(-1)
        candidate_count = int(eligible_flat.sum().item())
        if candidate_count <= 0:
            continue
        base_rv_area = int(base_rv[sample_idx].sum().item())
        base_lv_area = int(base_lv[sample_idx].sum().item())
        area_reference = max(base_rv_area, base_lv_area, 16)
        max_extra = max(1, int(round(float(max_extra_ratio) * float(area_reference))))
        keep_count = min(candidate_count, max_extra)
        eligible_indices = torch.nonzero(eligible_flat, as_tuple=False).flatten()
        eligible_scores = rank_score[sample_idx].reshape(-1)[eligible_indices]
        keep_indices = eligible_indices[torch.topk(eligible_scores, k=keep_count, largest=True).indices]
        selected[sample_idx].view(-1)[keep_indices] = True

    if not bool(selected.any()):
        return base_logits
    base_non_rv = base_logits[:, non_rv].amax(dim=1)
    target_margin = (0.02 + float(strength) * positive_margin).clamp(min=0.02, max=2.0)
    out = base_logits.clone()
    out[:, rv_class] = torch.where(
        selected,
        base_non_rv + target_margin,
        out[:, rv_class],
    )
    return out


def select_classwise_candidate_logits(
    base_logits: torch.Tensor,
    candidates: Dict[str, torch.Tensor],
    source_logits: torch.Tensor,
    prior: Dict[str, torch.Tensor],
    candidate_names: Iterable[str],
    *,
    name_bias: Optional[Dict[str, float]],
    min_gain: float,
    compose: str,
    logit_boost: float,
) -> Tuple[torch.Tensor, Dict[int, str], Dict[int, Dict[str, float]]]:
    names = []
    for name in candidate_names:
        key = str(name).strip()
        if key and key in candidates and key not in names:
            names.append(key)
    if "state" in candidates and "state" not in names:
        names.insert(0, "state")
    if not names:
        return base_logits, {}, {}

    selected: Dict[int, str] = {}
    selected_scores: Dict[int, Dict[str, float]] = {}
    class_count = int(base_logits.shape[1])
    for cls_idx in range(1, class_count):
        infos = {
            name: classwise_candidate_info(candidates[name], source_logits, prior, cls_idx, name)
            for name in names
        }
        adjusted = {
            name: float(info["score"]) + float((name_bias or {}).get(name, 0.0))
            for name, info in infos.items()
        }
        best_name = max(adjusted, key=adjusted.get)
        if "state" in adjusted and best_name != "state" and adjusted[best_name] < adjusted["state"] + float(min_gain):
            best_name = "state"
        selected[cls_idx] = best_name
        selected_scores[cls_idx] = {
            "best_score": adjusted[best_name],
            "state_score": adjusted.get("state"),
            **{f"{name}_score": score for name, score in adjusted.items()},
        }
    composed = compose_classwise_candidate_logits(
        base_logits,
        candidates,
        selected,
        compose=compose,
        logit_boost=logit_boost,
    )
    return composed, selected, selected_scores


def select_candidate_v3(
    candidate_logits: Dict[str, torch.Tensor],
    candidate_infos: Dict[str, Dict[str, float]],
    *,
    source_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    source_info: Dict[str, float],
    adapted_info: Dict[str, float],
    refined_info: Optional[Dict[str, float]],
    candidate_names: Iterable[str],
    name_bias: Optional[Dict[str, float]],
    min_gain: float,
    geometry_weight: float,
    align_weight: float,
    center_weight: float,
    rv_weight: float,
    clean_final_rv_min_ratio: float,
    clean_final_area_min_ratio: float,
    clean_final_penalty: float,
    tta_geometry_guard: float,
    tta_align_guard: float,
    tta_penalty: float,
    allow_source: bool,
) -> Tuple[str, torch.Tensor, Dict[str, float], str]:
    pool_logits: Dict[str, torch.Tensor] = dict(candidate_logits)
    pool_infos: Dict[str, Dict[str, float]] = dict(candidate_infos)
    pool_logits["adapted"] = adapted_logits
    pool_infos["adapted"] = adapted_info
    if refined_info is not None:
        pool_logits["refined"] = refined_logits
        pool_infos["refined"] = refined_info
    if allow_source:
        pool_logits["source"] = source_logits
        pool_infos["source"] = source_info

    names: List[str] = []
    for item in candidate_names:
        name = str(item).strip()
        if name and name in pool_logits and name in pool_infos and name not in names:
            names.append(name)
    if not names:
        names = [name for name in pool_logits if name in pool_infos]

    ref_name = "clean_final" if "clean_final" in pool_infos else "final" if "final" in pool_infos else names[0]
    ref_info = pool_infos[ref_name]
    ref_score = candidate_bank_adjusted_score(ref_name, ref_info, name_bias)
    adapted_rv = float(adapted_info.get("rv_pred_area", 0.0))
    adapted_area = float(adapted_info.get("fg_area", 0.0))

    scored: Dict[str, float] = {}
    notes: List[str] = []
    for name in names:
        info = pool_infos[name]
        score = candidate_bank_adjusted_score(name, info, name_bias)
        score += float(geometry_weight) * float(info.get("geometry", 0.0))
        score += float(align_weight) * float(info.get("prior_align_mean", info.get("prior_align", 0.0)) or 0.0)
        center_dist = info.get("center_dist")
        if center_dist is not None:
            score -= float(center_weight) * float(center_dist)
        score += float(rv_weight) * float(info.get("rv_pred_area", 0.0))

        if name == "clean_final":
            rv_pred = float(info.get("rv_pred_area", 0.0))
            fg_area = float(info.get("fg_area", 0.0))
            rv_bad = adapted_rv > 1e-5 and rv_pred < adapted_rv * float(clean_final_rv_min_ratio)
            area_bad = adapted_area > 1e-5 and fg_area < adapted_area * float(clean_final_area_min_ratio)
            if rv_bad or area_bad:
                score -= float(clean_final_penalty)
                notes.append(f"clean_final_penalty:rv_bad={rv_bad}/area_bad={area_bad}")

        if name.startswith("tta_"):
            geom_ok = float(info.get("geometry", 0.0)) >= float(ref_info.get("geometry", 0.0)) - float(tta_geometry_guard)
            align_ok = float(info.get("prior_align_mean", 0.0) or 0.0) >= float(ref_info.get("prior_align_mean", 0.0) or 0.0) - float(tta_align_guard)
            if not (geom_ok and align_ok):
                score -= float(tta_penalty)
                notes.append(f"{name}_penalty:geom_ok={geom_ok}/align_ok={align_ok}")
        scored[name] = float(score)

    best_name = max(scored, key=scored.get)
    if best_name != ref_name and scored[best_name] < ref_score + float(min_gain):
        notes.append(f"v3_min_gain_reject:{best_name}->{ref_name}")
        best_name = ref_name

    score_brief = {name: round(value, 3) for name, value in sorted(scored.items())}
    reason = f"selector_v3={best_name},v3_scores={score_brief}"
    if notes:
        reason += ",v3_notes=" + "|".join(notes[:6])
    return best_name, pool_logits[best_name], pool_infos[best_name], reason


def select_guarded_clean_candidate(
    candidate_logits: Dict[str, torch.Tensor],
    candidate_infos: Dict[str, Dict[str, float]],
    *,
    required_votes: int,
    entropy_margin: float,
    geometry_margin: float,
    align_margin: float,
    area_min_ratio: float,
    rv_min_ratio: float,
    rv_conf_margin: float,
) -> Tuple[str, torch.Tensor, Dict[str, float], str]:
    """Prefer clean P_final unless several independent MRE cues reject it."""
    final_name = "clean_final" if "clean_final" in candidate_infos else "final"
    adapted_name = "clean_adapted" if "clean_adapted" in candidate_infos else "adapted"
    if final_name not in candidate_logits or adapted_name not in candidate_logits:
        fallback = final_name if final_name in candidate_logits else next(iter(candidate_logits))
        return fallback, candidate_logits[fallback], candidate_infos[fallback], "guarded_mre_missing_pair"

    final_info = candidate_infos[final_name]
    adapted_info = candidate_infos[adapted_name]
    votes: List[str] = []
    if float(final_info.get("entropy", 0.0)) > float(adapted_info.get("entropy", 0.0)) + float(entropy_margin):
        votes.append("entropy")
    if float(final_info.get("geometry", 0.0)) < float(adapted_info.get("geometry", 0.0)) - float(geometry_margin):
        votes.append("geometry")
    final_align = float(final_info.get("prior_align_mean", final_info.get("prior_align", 0.0)) or 0.0)
    adapted_align = float(adapted_info.get("prior_align_mean", adapted_info.get("prior_align", 0.0)) or 0.0)
    if final_align < adapted_align - float(align_margin):
        votes.append("prior_align")

    final_area = float(final_info.get("fg_area", 0.0))
    adapted_area = float(adapted_info.get("fg_area", 0.0))
    if adapted_area > 1e-6 and final_area < adapted_area * float(area_min_ratio):
        votes.append("fg_area")
    final_rv = float(final_info.get("rv_pred_area", 0.0))
    adapted_rv = float(adapted_info.get("rv_pred_area", 0.0))
    if adapted_rv > 1e-6 and final_rv < adapted_rv * float(rv_min_ratio):
        votes.append("rv_area")
    if float(final_info.get("rv_confidence", 0.0)) < float(adapted_info.get("rv_confidence", 0.0)) - float(rv_conf_margin):
        votes.append("rv_confidence")

    needed = max(1, int(required_votes))
    selected = adapted_name if len(votes) >= needed else final_name
    reason = f"guarded_mre={selected},votes={len(votes)}/{needed},cues={'+'.join(votes) if votes else 'none'}"
    return selected, candidate_logits[selected], candidate_infos[selected], reason


def build_models(
    device: torch.device,
    adapter_name: str,
    *,
    tent_learning_rate: Optional[float] = None,
    tent_train_scope: str = "bn",
    disable_adaptation_dropout: bool = False,
    safe_ema_candidate: bool = False,
    temporal_consistency: bool = False,
    safe_ema_alpha: float = 0.999,
):
    base_model = load_model(cfg.MODEL.NETWORK, cfg.MODEL.CKPT_DIR, cfg.MODEL.DATASET, cfg.MODEL.METHOD)
    base_model = base_model.to(device)
    source_model = copy.deepcopy(base_model).to(device).eval()
    for param in source_model.parameters():
        param.requires_grad_(False)

    adapter_key = str(adapter_name).strip().lower()
    if adapter_key == "source":
        adapter_model = SourceAdapter(base_model)
    elif adapter_key == "tent":
        adapter_model = TentAdapter(
            base_model,
            learning_rate=tent_learning_rate,
            train_scope=tent_train_scope,
            disable_dropout=disable_adaptation_dropout,
        )
        if safe_ema_candidate or temporal_consistency:
            adapter_model.enable_safe_ema(safe_ema_alpha)
    elif adapter_key == "a3-lite":
        adapter_model = A3LiteAdapter(base_model)
    elif adapter_key == "a3-no-anchor":
        adapter_model = setup_a3_tta(base_model, ablation_mode="no-anchor")
        adapter_model.adapter_name = "a3-no-anchor"
    elif adapter_key == "a3-no-ema":
        adapter_model = setup_a3_tta(base_model, ablation_mode="no-ema")
        adapter_model.adapter_name = "a3-no-ema"
    elif adapter_key == "a3-no-anchor-no-ema":
        adapter_model = setup_a3_tta(base_model, ablation_mode="no-anchor-no-ema")
        adapter_model.adapter_name = "a3-no-anchor-no-ema"
    elif adapter_key == "a3-full":
        adapter_model = setup_a3_tta(base_model)
        adapter_model.adapter_name = "a3-full"
    else:
        raise ValueError(f"Unknown adapter {adapter_name!r}; expected one of {ADAPTER_NAMES}")
    adapter_model = adapter_model.to(device)
    return adapter_model, source_model


def evaluate_domain(
    *,
    a3_model,
    source_model,
    adapter_name: str,
    use_apssa: bool,
    domain: str,
    output_label: Optional[str],
    save_dir: Path,
    batch_size: int,
    num_workers: int,
    max_cases: Optional[int],
    calibration_corruption: str,
    surface_assd_protocol: str,
    entropy_margin: float,
    drift_threshold: float,
    geometry_margin: float,
    source_fusion: float,
    source_fusion_classes: Optional[Iterable[float]],
    adaptive_class_fusion: bool,
    adaptive_fusion_strength: float,
    adaptive_fusion_max: float,
    candidate_bank_v2: bool,
    candidate_bank_fusion_scales: Optional[Iterable[float]],
    candidate_bank_include_refined: bool,
    candidate_bank_tta_flips: Optional[Iterable[str]],
    candidate_bank_min_gain: float,
    candidate_bank_name_bias: Optional[Dict[str, float]],
    rv_composed_candidate_bank: bool,
    rv_composed_candidate_names: Optional[Iterable[str]],
    rv_consensus_candidate_bank: bool,
    rv_consensus_candidate_names: Optional[Iterable[str]],
    rv_consensus_min_agreement: int,
    rv_consensus_confidence: float,
    rv_consensus_base_margin_floor: float,
    rv_consensus_prior_threshold: float,
    rv_consensus_protect_confidence: float,
    rv_consensus_support_kernel: int,
    rv_consensus_max_extra_ratio: float,
    rv_consensus_strength: float,
    mre_calibrator: Optional[Dict[str, object]],
    rv_gain_selector: Optional[Dict[str, object]],
    candidate_cleanup_v2: bool,
    candidate_cleanup_sources: Optional[Iterable[str]],
    cleanup_min_area_abs: int,
    cleanup_min_area_ratio: float,
    cleanup_logit_bias: float,
    cleanup_keep_largest_classes: Optional[Iterable[int]],
    cleanup_fill_hole_classes: Optional[Iterable[int]],
    anatomy_protected_cleanup: bool,
    cleanup_protected_sources: Optional[Iterable[str]],
    cleanup_protected_classes: Optional[Iterable[int]],
    cleanup_protection_min_area_abs: int,
    cleanup_protection_candidate_confidence: float,
    cleanup_protection_reference_overlap: float,
    cleanup_protection_consensus_overlap: float,
    cleanup_protection_prior_mean: float,
    cleanup_protection_presence_margin: float,
    cleanup_protection_max_components: int,
    volume_continuity_blend: float,
    final_bias: float,
    final_min_gain: float,
    final_geometry_guard: float,
    source_fallback_margin: float,
    source_output_penalty: float,
    rollback_state: bool,
    mre_selection_enabled: bool,
    risk_aware_reversion: bool,
    preupdate_rpr: bool,
    rpr_calibrator: Optional[Dict[str, object]],
    rpr_vote_threshold: int,
    rpr_checkpoint_interval: int,
    rpr_entropy_margin: float,
    rpr_prior_align_margin: float,
    rpr_center_margin: float,
    rpr_topology_margin: float,
    rpr_confidence_margin: float,
    rpr_pre_adapt_drift_threshold: float,
    rpr_fg_min_ratio: float,
    rpr_fg_max_ratio: float,
    rpr_paired_probe: bool,
    rpr_probe_horizons: Iterable[int],
    rpr_probe_max_events: Optional[int],
    anatomical_prior_enabled: bool,
    prior_guided: bool,
    apl_online_weight: float,
    prior_source_blend: float,
    prior_smooth_sigma: float,
    prior_generic_sigma: float,
    prior_min: float,
    prior_presence_preserving: bool,
    prior_presence_classes: Optional[Iterable[int]],
    prior_presence_min_pixels: int,
    prior_presence_area_reference: int,
    prior_presence_confidence_low: float,
    prior_presence_confidence_high: float,
    prior_presence_floor: float,
    topology_prior_strength: float,
    topology_rv_strength: float,
    topology_ring_kernel: int,
    soft_myo_ring_strength: float,
    soft_rv_adjacency_strength: float,
    soft_anatomy_ring_kernel: int,
    soft_anatomy_rv_kernel: int,
    soft_anatomy_gate_low: float,
    soft_anatomy_gate_high: float,
    soft_anatomy_max_delta: float,
    soft_anatomy_candidate_bank: bool,
    soft_anatomy_candidate_min_gain: float,
    rv_image_candidate: bool,
    rv_image_probability_cap: float,
    rv_image_adjacency_kernel: int,
    rv_image_trigger_ratio: float,
    rv_image_max_area_ratio: float,
    rv_image_background_margin: float,
    rv_image_min_lv_pixels: int,
    rv_image_lv_confidence: float,
    rv_image_myo_confidence: float,
    rv_image_intensity_sigma_scale: float,
    rv_atlas_candidate: bool,
    rv_atlas: Optional[Dict[str, object]],
    rv_atlas_probability_cap: float,
    rv_atlas_support_threshold: float,
    rv_atlas_max_area_scale: float,
    rv_atlas_background_margin: float,
    rv_atlas_min_lv_pixels: int,
    rv_atlas_lv_confidence: float,
    rv_atlas_myo_confidence: float,
    rv_atlas_competitor_margin: float,
    rv_slice_candidate: bool,
    rv_slice_probability_cap: float,
    rv_slice_max_gap: int,
    rv_slice_require_two_sided: bool,
    rv_slice_seed_confidence: float,
    rv_slice_min_seed_pixels: int,
    rv_slice_min_lv_pixels: int,
    rv_slice_trigger_neighbor_ratio: float,
    rv_slice_max_area_scale: float,
    rv_slice_support_kernel: int,
    rv_slice_direction_sigma_scale: float,
    rv_slice_direction_consistency: float,
    rv_slice_background_margin: float,
    rv_slice_competitor_ratio: float,
    rv_slice_target_rv_hint: float,
    safe_ema_candidate: bool,
    safe_ema_apl_refine: bool,
    safe_ema_candidate_min_gain: float,
    temporal_consistency: bool,
    temporal_consistency_lambda: float,
    temporal_consistency_class_weights: Optional[Iterable[float]],
    temporal_consistency_conf_threshold: float,
    temporal_consistency_rv_conf_threshold: float,
    temporal_consistency_prior_threshold: float,
    temporal_consistency_temperature: float,
    topology_score_weight: float,
    topology_guard: float,
    prior_entropy_lambda: float,
    prior_loc_lambda: float,
    prior_bce_lambda: float,
    prior_ts_lambda: float,
    prior_source_consistency_lambda: float,
    prior_weight_gamma: float,
    prior_weight_min: float,
    prior_ema_alpha: float,
    prior_refine_steps: int,
    prior_refine_lr: float,
    prior_class_weights: Optional[Iterable[float]],
    prior_consistency_class_weights: Optional[Iterable[float]],
    boundary_prior_lambda: float,
    boundary_prior_classes: Optional[Iterable[int]],
    boundary_prior_kernel: int,
    boundary_uncertainty_low: float,
    boundary_uncertainty_high: float,
    mre_lambda_ap: float,
    mre_area_min: float,
    mre_area_max: float,
    mre_center_max: float,
    mre_use_entropy: bool,
    mre_use_prior_align: bool,
    mre_use_area: bool,
    mre_use_center: bool,
    mre_use_prior_conf: bool,
    mre_use_topology: bool,
    rv_guard: bool,
    rv_guard_min_area: float,
    rv_guard_min_ratio: float,
    rv_guard_conf_drop: float,
    final_takeover_v2: bool,
    final_takeover_score_gain: float,
    final_takeover_geometry_gain: float,
    final_takeover_align_gain: float,
    final_takeover_entropy_increase: float,
    source_rescue_v2: bool,
    source_rescue_margin: float,
    source_rescue_min_geometry: float,
    source_rescue_rv_min_ratio: float,
    source_rescue_rv_max_ratio: float,
    source_rescue_align_margin: float,
    source_rescue_entropy_margin: float,
    samplewise_mre_output: bool,
    guarded_mre_clean: bool,
    guarded_mre_required_votes: int,
    guarded_mre_entropy_margin: float,
    guarded_mre_geometry_margin: float,
    guarded_mre_align_margin: float,
    guarded_mre_area_min_ratio: float,
    guarded_mre_rv_min_ratio: float,
    guarded_mre_rv_conf_margin: float,
    candidate_selector_v3: bool,
    candidate_selector_v3_names: Optional[Iterable[str]],
    candidate_selector_v3_name_bias: Optional[Dict[str, float]],
    candidate_selector_v3_min_gain: float,
    candidate_selector_v3_geometry_weight: float,
    candidate_selector_v3_align_weight: float,
    candidate_selector_v3_center_weight: float,
    candidate_selector_v3_rv_weight: float,
    candidate_selector_v3_clean_final_rv_min_ratio: float,
    candidate_selector_v3_clean_final_area_min_ratio: float,
    candidate_selector_v3_clean_final_penalty: float,
    candidate_selector_v3_tta_geometry_guard: float,
    candidate_selector_v3_tta_align_guard: float,
    candidate_selector_v3_tta_penalty: float,
    candidate_selector_v3_allow_source: bool,
    classwise_candidate_select: bool,
    classwise_candidate_names: Optional[Iterable[str]],
    classwise_candidate_min_gain: float,
    classwise_candidate_name_bias: Optional[Dict[str, float]],
    classwise_candidate_compose: str,
    classwise_candidate_logit_boost: float,
    force_output_candidate: Optional[str],
    candidate_oracle_audit: bool,
    candidate_oracle_assd: bool,
    oracle_audit: bool,
    safe_branch_selector_conservative: bool,
    safe_branch_selector_adaptive: bool,
    use_legacy_selector: bool,
    keep_prior_state: bool,
    allow_source_output: bool,
    prior_refine_clone: bool,
    save_pred_masks_dir: Optional[Path],
    save_pred_masks_cases: Optional[Iterable[str]],
    save_pred_masks_all: bool,
    save_stage_masks: bool,
) -> Dict[str, object]:
    metric_fn = get_multi_class_evaluation_score
    dataset, _, _ = get_dataset(dataset=cfg.MODEL.DATASET, domain=domain, online=True)
    slice_positions = dataset_slice_position_map(dataset)
    if max_cases is not None:
        subset_size = min(max(0, int(max_cases)), len(dataset))
        dataset = Subset(dataset, range(subset_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    domain_label = str(output_label or domain)
    domain_dir = save_dir / domain_label
    domain_dir.mkdir(parents=True, exist_ok=True)

    dice_rows: List[List[object]] = []
    assd_rows: List[List[object]] = []
    hd95_rows: List[List[object]] = []
    audit_rows: List[List[object]] = []
    case_summary_rows: List[List[object]] = []
    candidate_trace_rows: List[List[object]] = []
    candidate_oracle_rows: List[List[object]] = []
    paired_probe_rows: List[List[object]] = []
    dice_all: List[List[float]] = []
    assd_all: List[List[float]] = []
    hd95_all: List[List[float]] = []
    branch_counter: Counter = Counter()
    risky_count = 0
    risk_reverted_count = 0
    rpr_checkpoint_commit_count = 0
    rpr_checkpoint_rollback_span_total = 0
    accepted_count = 0
    total_batches = 0
    paired_probe_events = 0
    probe_horizons = sorted({int(x) for x in rpr_probe_horizons if int(x) > 0})
    probe_max_horizon = max(probe_horizons) if probe_horizons else 0
    seen = 0
    save_case_names = {os.path.basename(str(x)) for x in (save_pred_masks_cases or [])}
    mask_out_dir: Optional[Path] = None
    if save_pred_masks_dir is not None:
        mask_out_dir = Path(save_pred_masks_dir) / domain
        mask_out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_interval = max(1, int(rpr_checkpoint_interval))
    checkpoint_controller: Optional[RPRCheckpointController] = None
    if (
        use_apssa
        and preupdate_rpr
        and risk_aware_reversion
        and rollback_state
        and checkpoint_interval > 1
    ):
        checkpoint_controller = getattr(a3_model, "_rpr_checkpoint_controller", None)
        if checkpoint_controller is None:
            checkpoint_controller = RPRCheckpointController(checkpoint_interval)
            setattr(a3_model, "_rpr_checkpoint_controller", checkpoint_controller)
        elif int(checkpoint_controller.interval) != checkpoint_interval:
            raise RuntimeError(
                "The persistent RPR checkpoint interval changed within one stream"
            )

    device = next(source_model.parameters()).device
    progress_name = f"{adapter_name}-{'APSA' if use_apssa else 'raw'}-MMS-{domain_label}"
    for batch_idx, sampled in enumerate(tqdm(loader, desc=progress_name, ncols=96)):
        volume = sampled["image"].to(device)
        label = sampled["label"].to(device)
        names = sampled["names"]
        volume, label = convert_2d(volume, label)
        volume = apply_source_calibration_corruption(
            volume,
            calibration_corruption,
            batch_index=batch_idx,
        )
        label_np = label.detach().cpu().numpy().squeeze(1)

        with torch.no_grad():
            source_logits = source_model(volume)
            ema_logits = (
                a3_model.safe_ema_forward(volume)
                if safe_ema_candidate and hasattr(a3_model, "safe_ema_forward")
                else None
            )
        if use_apssa:
            if anatomical_prior_enabled:
                prior = build_cardiac_prior(
                    source_logits,
                    source_blend=prior_source_blend,
                    smooth_sigma=prior_smooth_sigma,
                    generic_sigma=prior_generic_sigma,
                    min_prior=prior_min,
                    topology_strength=topology_prior_strength,
                    topology_rv_strength=topology_rv_strength,
                    topology_ring_kernel=topology_ring_kernel,
                    presence_preserving=prior_presence_preserving,
                    presence_classes=prior_presence_classes,
                    presence_min_pixels=prior_presence_min_pixels,
                    presence_area_reference=prior_presence_area_reference,
                    presence_confidence_low=prior_presence_confidence_low,
                    presence_confidence_high=prior_presence_confidence_high,
                    presence_floor=prior_presence_floor,
                )
            else:
                prior = build_neutral_prior(source_logits)
        else:
            prior = None
        snap_before_a3 = snapshot_a3_state(a3_model) if use_apssa else None
        if checkpoint_controller is not None and snap_before_a3 is not None:
            checkpoint_controller.ensure_initialized(snap_before_a3)
        pre_update_logits: Optional[torch.Tensor] = None
        if use_apssa and preupdate_rpr:
            with torch.no_grad():
                pre_update_logits = a3_model.model(volume)
            # A pre-update observation must not advance any mutable runtime
            # state. This is a no-op for the clean BN configuration, whose
            # running statistics are disabled, and protects future carriers.
            restore_a3_state(a3_model, snap_before_a3)
        if use_apssa and adapter_name == "tent" and float(apl_online_weight) > 0.0:
            adapted_logits = a3_model.forward_apl(
                volume,
                prior,
                source_logits,
                apl_weight=apl_online_weight,
                entropy_lambda=prior_entropy_lambda,
                loc_lambda=prior_loc_lambda,
                bce_lambda=prior_bce_lambda,
                ts_lambda=prior_ts_lambda,
                source_consistency_lambda=prior_source_consistency_lambda,
                weight_gamma=prior_weight_gamma,
                weight_min=prior_weight_min,
                class_weights=prior_class_weights,
                consistency_class_weights=prior_consistency_class_weights,
                temporal_consistency_lambda=(
                    temporal_consistency_lambda if temporal_consistency else 0.0
                ),
                temporal_consistency_class_weights=temporal_consistency_class_weights,
                temporal_consistency_conf_threshold=temporal_consistency_conf_threshold,
                temporal_consistency_rv_conf_threshold=temporal_consistency_rv_conf_threshold,
                temporal_consistency_prior_threshold=temporal_consistency_prior_threshold,
                temporal_consistency_temperature=temporal_consistency_temperature,
            )
        else:
            adapted_logits = a3_model(volume, label, names)
        snap_after_a3 = snapshot_a3_state(a3_model) if use_apssa else None

        if not use_apssa:
            chosen_logits = source_logits if adapter_name == "source" else adapted_logits
            selected_branch = "source" if adapter_name == "source" else "adapted"
            selected_candidate = "P_source" if adapter_name == "source" else "P_adapt"
            branch_counter[selected_branch] += int(chosen_logits.shape[0])
            total_batches += 1
            pred = chosen_logits.argmax(dim=1).detach().cpu().numpy()
            for i, name in enumerate(names):
                image_name = os.path.basename(str(name))
                if mask_out_dir is not None and (save_pred_masks_all or image_name in save_case_names):
                    mask_img = sitk.GetImageFromArray(pred[i].astype(np.uint8))
                    sitk.WriteImage(mask_img, str(mask_out_dir / image_name))
                dice = [float(x) for x in metric_fn(pred[i], label_np[i], int(cfg.MODEL.NUMBER_CLASS), "dice")]
                assd, hd95 = surface_metrics_per_class(
                    pred[i],
                    label_np[i],
                    int(cfg.MODEL.NUMBER_CLASS),
                    assd_protocol=surface_assd_protocol,
                    metric_fn=metric_fn,
                )
                dice_all.append(dice)
                assd_all.append(assd)
                hd95_all.append(hd95)
                dice_rows.append([image_name] + dice + [float(np.mean(dice))])
                assd_rows.append([image_name] + assd + [float(np.mean(assd))])
                hd95_rows.append([image_name] + hd95 + [nanmean_or_nan(hd95)])
                case_summary_rows.append([domain_label, image_name, batch_idx, adapter_name, 0, selected_candidate, 0])
                for candidate_name, candidate_logits in (("P_source", source_logits), ("P_adapt", adapted_logits)):
                    candidate_trace_rows.append(
                        [
                            domain_label,
                            image_name,
                            batch_idx,
                            adapter_name,
                            0,
                            candidate_name,
                            None,
                            mean_normalized_entropy(candidate_logits[i : i + 1]),
                            None,
                            None,
                            selected_candidate,
                            int(candidate_name == selected_candidate),
                            0,
                        ]
                    )
                seen += 1
            continue

        tta_candidate_logits: Dict[str, torch.Tensor] = {}
        if candidate_bank_v2:
            for flip_name in candidate_bank_tta_flips or []:
                dims = _tta_flip_dims(flip_name)
                if dims is None:
                    continue
                with torch.no_grad():
                    flipped_volume = torch.flip(volume, dims=dims)
                    flipped_logits = a3_model.model(flipped_volume)
                    unflipped_logits = torch.flip(flipped_logits, dims=dims)
                restore_a3_state(a3_model, snap_after_a3)
                tta_candidate_logits[f"tta_{str(flip_name).lower()}"] = 0.5 * (adapted_logits + unflipped_logits)
            if len(tta_candidate_logits) > 1:
                stacked_tta = torch.stack(list(tta_candidate_logits.values()), dim=0)
                tta_candidate_logits["tta_avg"] = stacked_tta.mean(dim=0)
        refine_stats = {"loss": 0.0, "ent": 0.0, "loc": 0.0, "ce": 0.0, "boundary": 0.0, "ts": 0.0, "src_cons": 0.0}
        refined_logits = adapted_logits
        if prior_guided:
            if prior_refine_clone:
                refined_logits, refine_stats = prior_guided_refine_candidate(
                    a3_model,
                    volume,
                    source_logits,
                    adapted_logits,
                    prior,
                    entropy_lambda=prior_entropy_lambda,
                    loc_lambda=prior_loc_lambda,
                    bce_lambda=prior_bce_lambda,
                    ts_lambda=prior_ts_lambda,
                    source_consistency_lambda=prior_source_consistency_lambda,
                    weight_gamma=prior_weight_gamma,
                    weight_min=prior_weight_min,
                    refine_steps=prior_refine_steps,
                    refine_learning_rate=prior_refine_lr,
                    class_weights=prior_class_weights,
                    consistency_class_weights=prior_consistency_class_weights,
                    boundary_prior_lambda=boundary_prior_lambda,
                    boundary_prior_classes=boundary_prior_classes,
                    boundary_prior_kernel=boundary_prior_kernel,
                    boundary_uncertainty_low=boundary_uncertainty_low,
                    boundary_uncertainty_high=boundary_uncertainty_high,
                )
            else:
                refined_logits, refine_stats = prior_guided_refine(
                    a3_model,
                    volume,
                    source_logits,
                    adapted_logits,
                    prior,
                    entropy_lambda=prior_entropy_lambda,
                    loc_lambda=prior_loc_lambda,
                    bce_lambda=prior_bce_lambda,
                    ts_lambda=prior_ts_lambda,
                    source_consistency_lambda=prior_source_consistency_lambda,
                    weight_gamma=prior_weight_gamma,
                    weight_min=prior_weight_min,
                    ema_alpha=prior_ema_alpha,
                    refine_steps=prior_refine_steps,
                    refine_learning_rate=prior_refine_lr,
                    class_weights=prior_class_weights,
                    consistency_class_weights=prior_consistency_class_weights,
                    boundary_prior_lambda=boundary_prior_lambda,
                    boundary_prior_classes=boundary_prior_classes,
                    boundary_prior_kernel=boundary_prior_kernel,
                    boundary_uncertainty_low=boundary_uncertainty_low,
                    boundary_uncertainty_high=boundary_uncertainty_high,
                )
        anatomy_refined_logits: Optional[torch.Tensor] = None
        if float(soft_myo_ring_strength) > 0.0 or float(soft_rv_adjacency_strength) > 0.0:
            projected_logits = confidence_gated_soft_anatomy_refine(
                refined_logits,
                prior.get("prob"),
                myo_ring_strength=soft_myo_ring_strength,
                rv_adjacency_strength=soft_rv_adjacency_strength,
                ring_kernel=soft_anatomy_ring_kernel,
                rv_kernel=soft_anatomy_rv_kernel,
                gate_low=soft_anatomy_gate_low,
                gate_high=soft_anatomy_gate_high,
                max_delta=soft_anatomy_max_delta,
            )
            if soft_anatomy_candidate_bank:
                anatomy_refined_logits = projected_logits
            else:
                refined_logits = projected_logits
        final_logits = confidence_aware_fusion(
            refined_logits,
            source_logits,
            source_weight=source_fusion,
            source_weight_by_class=source_fusion_classes,
            adaptive_class_fusion=adaptive_class_fusion,
            adaptive_fusion_strength=adaptive_fusion_strength,
            adaptive_fusion_max=adaptive_fusion_max,
        )
        candidate_logits: Dict[str, torch.Tensor] = {"final": final_logits}
        rv_image_stats = {
            "trigger_fraction": 0.0,
            "support_fraction": 0.0,
            "changed_fraction": 0.0,
            "mean_probability_delta": 0.0,
        }
        if rv_image_candidate:
            rv_image_logits, rv_image_stats = rv_image_conditioned_candidate(
                volume,
                final_logits,
                refined_logits,
                adapted_logits,
                probability_cap=rv_image_probability_cap,
                adjacency_kernel=rv_image_adjacency_kernel,
                trigger_ratio=rv_image_trigger_ratio,
                max_area_ratio=rv_image_max_area_ratio,
                background_margin=rv_image_background_margin,
                min_lv_pixels=rv_image_min_lv_pixels,
                lv_confidence=rv_image_lv_confidence,
                myo_confidence=rv_image_myo_confidence,
                intensity_sigma_scale=rv_image_intensity_sigma_scale,
            )
            candidate_logits["rv_image_final"] = rv_image_logits
        rv_atlas_stats = {
            "trigger_fraction": 0.0,
            "support_fraction": 0.0,
            "changed_fraction": 0.0,
            "hard_changed_fraction": 0.0,
            "mean_probability_delta": 0.0,
            "mean_atlas_presence": 0.0,
        }
        if rv_atlas_candidate:
            if rv_atlas is None:
                raise RuntimeError("RV atlas candidate is enabled but no source atlas was loaded")
            rv_atlas_logits, rv_atlas_stats = rv_atlas_guided_candidate(
                names,
                slice_positions,
                rv_atlas,
                final_logits,
                refined_logits,
                adapted_logits,
                probability_cap=rv_atlas_probability_cap,
                support_threshold=rv_atlas_support_threshold,
                max_area_scale=rv_atlas_max_area_scale,
                background_margin=rv_atlas_background_margin,
                min_lv_pixels=rv_atlas_min_lv_pixels,
                lv_confidence=rv_atlas_lv_confidence,
                myo_confidence=rv_atlas_myo_confidence,
                competitor_margin=rv_atlas_competitor_margin,
            )
            candidate_logits["rv_atlas_final"] = rv_atlas_logits
        rv_slice_stats = {
            "neighbor_seed_fraction": 0.0,
            "trigger_fraction": 0.0,
            "mean_neighbor_count": 0.0,
            "support_fraction": 0.0,
            "changed_fraction": 0.0,
            "mean_probability_delta": 0.0,
        }
        if rv_slice_candidate:
            rv_slice_logits, rv_slice_stats = rv_directional_slice_candidate(
                names,
                final_logits,
                refined_logits,
                adapted_logits,
                probability_cap=rv_slice_probability_cap,
                max_slice_gap=rv_slice_max_gap,
                require_two_sided=rv_slice_require_two_sided,
                seed_confidence=rv_slice_seed_confidence,
                min_seed_pixels=rv_slice_min_seed_pixels,
                min_lv_pixels=rv_slice_min_lv_pixels,
                trigger_neighbor_ratio=rv_slice_trigger_neighbor_ratio,
                max_area_scale=rv_slice_max_area_scale,
                support_kernel=rv_slice_support_kernel,
                direction_sigma_scale=rv_slice_direction_sigma_scale,
                direction_consistency=rv_slice_direction_consistency,
                background_margin=rv_slice_background_margin,
                competitor_ratio=rv_slice_competitor_ratio,
                target_rv_hint=rv_slice_target_rv_hint,
            )
            candidate_logits["rv_slice_final"] = rv_slice_logits
        ema_final_logits: Optional[torch.Tensor] = None
        if candidate_bank_v2 and ema_logits is not None:
            ema_refined_logits = ema_logits
            if safe_ema_apl_refine and prior_guided:
                ema_refined_logits, _ = prior_guided_refine_candidate(
                    a3_model,
                    volume,
                    source_logits,
                    ema_logits,
                    prior,
                    entropy_lambda=prior_entropy_lambda,
                    loc_lambda=prior_loc_lambda,
                    bce_lambda=prior_bce_lambda,
                    ts_lambda=prior_ts_lambda,
                    source_consistency_lambda=prior_source_consistency_lambda,
                    weight_gamma=prior_weight_gamma,
                    weight_min=prior_weight_min,
                    refine_steps=prior_refine_steps,
                    refine_learning_rate=prior_refine_lr,
                    class_weights=prior_class_weights,
                    consistency_class_weights=prior_consistency_class_weights,
                    boundary_prior_lambda=boundary_prior_lambda,
                    boundary_prior_classes=boundary_prior_classes,
                    boundary_prior_kernel=boundary_prior_kernel,
                    boundary_uncertainty_low=boundary_uncertainty_low,
                    boundary_uncertainty_high=boundary_uncertainty_high,
                    template_model=a3_model.model_ema,
                )
                candidate_logits["ema_refined"] = ema_refined_logits
            candidate_logits["ema"] = ema_logits
            ema_final_logits = confidence_aware_fusion(
                ema_refined_logits,
                source_logits,
                source_weight=source_fusion,
                source_weight_by_class=source_fusion_classes,
                adaptive_class_fusion=adaptive_class_fusion,
                adaptive_fusion_strength=adaptive_fusion_strength,
                adaptive_fusion_max=adaptive_fusion_max,
            )
            candidate_logits["ema_final"] = ema_final_logits
        anatomy_final_logits: Optional[torch.Tensor] = None
        if candidate_bank_v2 and anatomy_refined_logits is not None:
            anatomy_final_logits = confidence_aware_fusion(
                anatomy_refined_logits,
                source_logits,
                source_weight=source_fusion,
                source_weight_by_class=source_fusion_classes,
                adaptive_class_fusion=adaptive_class_fusion,
                adaptive_fusion_strength=adaptive_fusion_strength,
                adaptive_fusion_max=adaptive_fusion_max,
            )
            candidate_logits["anatomy_final"] = anatomy_final_logits
        if candidate_bank_v2:
            if candidate_bank_include_refined:
                candidate_logits["refined"] = refined_logits
            for scale in candidate_bank_fusion_scales or []:
                scaled_classes = _scaled_fusion_weights(source_fusion_classes, scale)
                candidate_logits[f"fusion_x{float(scale):.2f}"] = confidence_aware_fusion(
                    refined_logits,
                    source_logits,
                    source_weight=float(source_fusion) * float(scale),
                    source_weight_by_class=scaled_classes,
                    adaptive_class_fusion=adaptive_class_fusion,
                    adaptive_fusion_strength=adaptive_fusion_strength,
                    adaptive_fusion_max=adaptive_fusion_max,
                )
            candidate_logits.update(tta_candidate_logits)
        if candidate_cleanup_v2:
            cleanup_source_map = {
                "final": final_logits,
                "adapted": adapted_logits,
                "refined": refined_logits,
            }
            cleanup_consensus_map = {
                "final": refined_logits,
                "adapted": adapted_logits,
                "refined": refined_logits,
            }
            if "rv_image_final" in candidate_logits:
                cleanup_source_map["rv_image_final"] = candidate_logits["rv_image_final"]
                cleanup_consensus_map["rv_image_final"] = refined_logits
            if "rv_slice_final" in candidate_logits:
                cleanup_source_map["rv_slice_final"] = candidate_logits["rv_slice_final"]
                cleanup_consensus_map["rv_slice_final"] = refined_logits
            if "rv_atlas_final" in candidate_logits:
                cleanup_source_map["rv_atlas_final"] = candidate_logits["rv_atlas_final"]
                cleanup_consensus_map["rv_atlas_final"] = refined_logits
            if anatomy_final_logits is not None:
                cleanup_source_map["anatomy_final"] = anatomy_final_logits
                cleanup_consensus_map["anatomy_final"] = anatomy_refined_logits
            if ema_final_logits is not None:
                cleanup_source_map["ema_final"] = ema_final_logits
                cleanup_consensus_map["ema_final"] = ema_refined_logits
            for source_name in candidate_cleanup_sources or []:
                key = str(source_name).strip().lower()
                if key not in cleanup_source_map:
                    continue
                candidate_logits[f"clean_{key}"] = cleanup_cardiac_logits(
                    cleanup_source_map[key],
                    min_area_abs=cleanup_min_area_abs,
                    min_area_ratio=cleanup_min_area_ratio,
                    logit_bias=cleanup_logit_bias,
                    keep_largest_classes=cleanup_keep_largest_classes or [],
                    fill_hole_classes=cleanup_fill_hole_classes or [],
                )
            if anatomy_protected_cleanup:
                for source_name in cleanup_protected_sources or []:
                    key = str(source_name).strip().lower()
                    if key not in cleanup_source_map:
                        continue
                    candidate_logits[f"protected_clean_{key}"] = cleanup_cardiac_logits(
                        cleanup_source_map[key],
                        min_area_abs=cleanup_min_area_abs,
                        min_area_ratio=cleanup_min_area_ratio,
                        logit_bias=cleanup_logit_bias,
                        keep_largest_classes=cleanup_keep_largest_classes or [],
                        fill_hole_classes=cleanup_fill_hole_classes or [],
                        reference_logits=source_logits,
                        consensus_logits=cleanup_consensus_map.get(key),
                        prior_prob=prior.get("prob"),
                        protected_classes=cleanup_protected_classes or [],
                        protection_min_area_abs=cleanup_protection_min_area_abs,
                        protection_candidate_confidence=cleanup_protection_candidate_confidence,
                        protection_reference_overlap=cleanup_protection_reference_overlap,
                        protection_consensus_overlap=cleanup_protection_consensus_overlap,
                        protection_prior_mean=cleanup_protection_prior_mean,
                        protection_presence_margin=cleanup_protection_presence_margin,
                        protection_max_components=cleanup_protection_max_components,
                    )
        if float(volume_continuity_blend) > 0.0 and "clean_final" in candidate_logits:
            candidate_logits["volume_clean_final"] = volume_continuity_logits(
                candidate_logits["clean_final"],
                names,
                blend=volume_continuity_blend,
            )
        if rv_composed_candidate_bank:
            rv_base_name = "clean_final" if "clean_final" in candidate_logits else "final"
            rv_base_logits = candidate_logits[rv_base_name]
            rv_donor_logits = dict(candidate_logits)
            rv_donor_logits.setdefault("adapted", adapted_logits)
            rv_donor_logits.setdefault("refined", refined_logits)
            donor_names = []
            for donor_name in rv_composed_candidate_names or []:
                donor_key = str(donor_name).strip()
                if (
                    donor_key
                    and donor_key != rv_base_name
                    and donor_key in rv_donor_logits
                    and donor_key not in donor_names
                ):
                    donor_names.append(donor_key)
            for donor_name in donor_names:
                candidate_logits[f"rv_from_{donor_name}"] = compose_rv_margin_candidate(
                    rv_base_logits,
                    rv_donor_logits[donor_name],
                )
        if rv_consensus_candidate_bank:
            rv_base_name = "clean_final" if "clean_final" in candidate_logits else "final"
            available_donors = dict(candidate_logits)
            available_donors.setdefault("adapted", adapted_logits)
            available_donors.setdefault("refined", refined_logits)
            consensus_donors = {}
            for donor_name in rv_consensus_candidate_names or []:
                donor_key = str(donor_name).strip()
                if (
                    donor_key
                    and donor_key != rv_base_name
                    and donor_key in available_donors
                    and donor_key not in consensus_donors
                ):
                    consensus_donors[donor_key] = available_donors[donor_key]
            candidate_logits["rv_consensus_final"] = compose_consensus_rv_margin_candidate(
                candidate_logits[rv_base_name],
                final_logits,
                consensus_donors,
                prior.get("prob"),
                min_agreement=rv_consensus_min_agreement,
                donor_confidence=rv_consensus_confidence,
                base_margin_floor=rv_consensus_base_margin_floor,
                prior_threshold=rv_consensus_prior_threshold,
                protect_confidence=rv_consensus_protect_confidence,
                support_kernel=rv_consensus_support_kernel,
                max_extra_ratio=rv_consensus_max_extra_ratio,
                strength=rv_consensus_strength,
            )

        candidate_infos: Dict[str, Dict[str, float]] = {}
        if use_legacy_selector:
            source_info = reliability_info(source_logits, source_logits, "source")
            adapted_info = reliability_info(adapted_logits, source_logits, "adapted")
            if candidate_bank_v2:
                candidate_infos = {
                    name: reliability_info(logits, source_logits, name) for name, logits in candidate_logits.items()
                }
                bank_name = max(
                    candidate_infos,
                    key=lambda key: candidate_bank_adjusted_score(key, candidate_infos[key], candidate_bank_name_bias),
                )
                bank_score = candidate_bank_adjusted_score(bank_name, candidate_infos[bank_name], candidate_bank_name_bias)
                final_score = candidate_bank_adjusted_score("final", candidate_infos["final"], candidate_bank_name_bias)
                if safe_ema_candidate:
                    bank_name, bank_score = guard_safe_ema_candidate(
                        bank_name,
                        candidate_infos,
                        candidate_bank_name_bias,
                        safe_ema_candidate_min_gain,
                    )
                if soft_anatomy_candidate_bank and bank_name in ("anatomy_final", "clean_anatomy_final"):
                    reference_name = "clean_final" if bank_name == "clean_anatomy_final" else "final"
                    reference_score = candidate_bank_adjusted_score(
                        reference_name, candidate_infos[reference_name], candidate_bank_name_bias
                    )
                    if bank_score < reference_score + float(soft_anatomy_candidate_min_gain):
                        bank_name = reference_name
                        bank_score = reference_score
                if bank_name != "final" and bank_score < final_score + float(candidate_bank_min_gain):
                    bank_name = "final"
                final_logits = candidate_logits[bank_name]
                final_info = candidate_infos[bank_name]
            else:
                bank_name = "final"
                final_info = reliability_info(final_logits, source_logits, "final")
            risky = is_risky_update(source_info, adapted_info, entropy_margin, drift_threshold, geometry_margin)
            best, reason = select_branch(
                source_info,
                adapted_info,
                final_info,
                risky,
                allow_source_output,
                final_bias,
                final_min_gain,
                source_fallback_margin,
            )
            if candidate_bank_v2:
                reason = f"candidate_bank={bank_name}," + reason
        else:
            source_info = compute_mre_info(
                source_logits,
                source_logits,
                prior,
                "source",
                mre_area_min,
                mre_area_max,
                mre_center_max,
                mre_lambda_ap,
                topology_score_weight,
                topology_ring_kernel,
                mre_use_entropy,
                mre_use_prior_align,
                mre_use_area,
                mre_use_center,
                mre_use_prior_conf,
                mre_use_topology,
            )
            adapted_info = compute_mre_info(
                adapted_logits,
                source_logits,
                prior,
                "adapted",
                mre_area_min,
                mre_area_max,
                mre_center_max,
                mre_lambda_ap,
                topology_score_weight,
                topology_ring_kernel,
                mre_use_entropy,
                mre_use_prior_align,
                mre_use_area,
                mre_use_center,
                mre_use_prior_conf,
                mre_use_topology,
            )
            if candidate_bank_v2:
                for name, logits in candidate_logits.items():
                    candidate_infos[name] = compute_mre_info(
                        logits,
                        source_logits,
                        prior,
                        name,
                        mre_area_min,
                        mre_area_max,
                        mre_center_max,
                        mre_lambda_ap,
                        topology_score_weight,
                        topology_ring_kernel,
                        mre_use_entropy,
                        mre_use_prior_align,
                        mre_use_area,
                        mre_use_center,
                        mre_use_prior_conf,
                        mre_use_topology,
                    )
                refined_info_for_v3: Optional[Dict[str, float]] = candidate_infos.get("refined")
                if candidate_selector_v3 and refined_info_for_v3 is None:
                    refined_info_for_v3 = compute_mre_info(
                        refined_logits,
                        source_logits,
                        prior,
                        "refined",
                        mre_area_min,
                        mre_area_max,
                        mre_center_max,
                        mre_lambda_ap,
                        topology_score_weight,
                        topology_ring_kernel,
                        mre_use_entropy,
                        mre_use_prior_align,
                        mre_use_area,
                        mre_use_center,
                        mre_use_prior_conf,
                        mre_use_topology,
                    )
                if rv_gain_selector is not None:
                    bank_name, final_logits, final_info, v3_reason = select_source_nonlinear_rv_candidate(
                        candidate_logits,
                        candidate_infos,
                        rv_gain_selector,
                    )
                elif mre_calibrator is not None:
                    bank_name, final_logits, final_info, v3_reason = select_source_calibrated_candidate(
                        candidate_logits,
                        candidate_infos,
                        mre_calibrator,
                    )
                elif guarded_mre_clean:
                    bank_name, final_logits, final_info, v3_reason = select_guarded_clean_candidate(
                        candidate_logits,
                        candidate_infos,
                        required_votes=guarded_mre_required_votes,
                        entropy_margin=guarded_mre_entropy_margin,
                        geometry_margin=guarded_mre_geometry_margin,
                        align_margin=guarded_mre_align_margin,
                        area_min_ratio=guarded_mre_area_min_ratio,
                        rv_min_ratio=guarded_mre_rv_min_ratio,
                        rv_conf_margin=guarded_mre_rv_conf_margin,
                    )
                elif candidate_selector_v3:
                    bank_name, final_logits, final_info, v3_reason = select_candidate_v3(
                        candidate_logits,
                        candidate_infos,
                        source_logits=source_logits,
                        adapted_logits=adapted_logits,
                        refined_logits=refined_logits,
                        source_info=source_info,
                        adapted_info=adapted_info,
                        refined_info=refined_info_for_v3,
                        candidate_names=candidate_selector_v3_names or [],
                        name_bias=candidate_selector_v3_name_bias,
                        min_gain=candidate_selector_v3_min_gain,
                        geometry_weight=candidate_selector_v3_geometry_weight,
                        align_weight=candidate_selector_v3_align_weight,
                        center_weight=candidate_selector_v3_center_weight,
                        rv_weight=candidate_selector_v3_rv_weight,
                        clean_final_rv_min_ratio=candidate_selector_v3_clean_final_rv_min_ratio,
                        clean_final_area_min_ratio=candidate_selector_v3_clean_final_area_min_ratio,
                        clean_final_penalty=candidate_selector_v3_clean_final_penalty,
                        tta_geometry_guard=candidate_selector_v3_tta_geometry_guard,
                        tta_align_guard=candidate_selector_v3_tta_align_guard,
                        tta_penalty=candidate_selector_v3_tta_penalty,
                        allow_source=candidate_selector_v3_allow_source,
                    )
                else:
                    v3_reason = ""
                    bank_name = max(
                        candidate_infos,
                        key=lambda key: candidate_bank_adjusted_score(key, candidate_infos[key], candidate_bank_name_bias),
                    )
                    bank_score = candidate_bank_adjusted_score(bank_name, candidate_infos[bank_name], candidate_bank_name_bias)
                    final_score = candidate_bank_adjusted_score("final", candidate_infos["final"], candidate_bank_name_bias)
                    if safe_ema_candidate:
                        bank_name, bank_score = guard_safe_ema_candidate(
                            bank_name,
                            candidate_infos,
                            candidate_bank_name_bias,
                            safe_ema_candidate_min_gain,
                        )
                    if soft_anatomy_candidate_bank and bank_name in ("anatomy_final", "clean_anatomy_final"):
                        reference_name = "clean_final" if bank_name == "clean_anatomy_final" else "final"
                        reference_score = candidate_bank_adjusted_score(
                            reference_name, candidate_infos[reference_name], candidate_bank_name_bias
                        )
                        if bank_score < reference_score + float(soft_anatomy_candidate_min_gain):
                            bank_name = reference_name
                            bank_score = reference_score
                    if bank_name != "final" and bank_score < final_score + float(candidate_bank_min_gain):
                        bank_name = "final"
                    final_logits = candidate_logits[bank_name]
                    final_info = candidate_infos[bank_name]
            else:
                v3_reason = ""
                bank_name = "final"
                final_info = compute_mre_info(
                    final_logits,
                    source_logits,
                    prior,
                    "final",
                    mre_area_min,
                    mre_area_max,
                    mre_center_max,
                    mre_lambda_ap,
                    topology_score_weight,
                    topology_ring_kernel,
                    mre_use_entropy,
                    mre_use_prior_align,
                    mre_use_area,
                    mre_use_center,
                    mre_use_prior_conf,
                    mre_use_topology,
                )
            risky = is_risky_update(source_info, adapted_info, entropy_margin, drift_threshold, geometry_margin)
            best, reason = safe_branch_select_mre(
                source_info,
                adapted_info,
                final_info,
                conservative=safe_branch_selector_conservative,
                adaptive=safe_branch_selector_adaptive,
                allow_source_output=allow_source_output,
                source_output_penalty=source_output_penalty,
                final_min_gain=final_min_gain,
                final_bias=final_bias,
                final_geometry_guard=final_geometry_guard,
                use_entropy=mre_use_entropy,
                use_prior_align=mre_use_prior_align,
                use_area=mre_use_area,
                use_topology=mre_use_topology,
                topology_guard=topology_guard,
                use_rv_guard=rv_guard,
                rv_guard_min_area=rv_guard_min_area,
                rv_guard_min_ratio=rv_guard_min_ratio,
                rv_guard_conf_drop=rv_guard_conf_drop,
                final_takeover_v2=final_takeover_v2,
                final_takeover_score_gain=final_takeover_score_gain,
                final_takeover_geometry_gain=final_takeover_geometry_gain,
                final_takeover_align_gain=final_takeover_align_gain,
                final_takeover_entropy_increase=final_takeover_entropy_increase,
                source_rescue_v2=source_rescue_v2,
                source_rescue_margin=source_rescue_margin,
                source_rescue_min_geometry=source_rescue_min_geometry,
                source_rescue_rv_min_ratio=source_rescue_rv_min_ratio,
                source_rescue_rv_max_ratio=source_rescue_rv_max_ratio,
                source_rescue_align_margin=source_rescue_align_margin,
                source_rescue_entropy_margin=source_rescue_entropy_margin,
            )
            if mre_calibrator is not None or rv_gain_selector is not None:
                # Source-calibrated MRE already performed output selection. RPR
                # below remains responsible for accepting/reverting online state.
                best = "final"
                reason = f"{v3_reason},source_calibrated_mre_output,{reason}"
            if candidate_bank_v2:
                bank_scores = {name: round(info["score"], 3) for name, info in candidate_infos.items()}
                bank_adj_scores = {
                    name: round(candidate_bank_adjusted_score(name, info, candidate_bank_name_bias), 3)
                    for name, info in candidate_infos.items()
                }
                reason = f"candidate_bank={bank_name},bank_scores={bank_scores},bank_adj_scores={bank_adj_scores}," + reason
                if guarded_mre_clean or candidate_selector_v3:
                    reason = f"{v3_reason}," + reason

        pre_update_info: Optional[Dict[str, float]] = None
        rpr_evidence: Dict[str, object] = {"features": {}}
        rpr_decision: Dict[str, object] = {
            "risky": bool(risky),
            "decision_source": "legacy_source_relative",
            "decision_score": None,
            "vote_count": 0,
            "vote_reasons": [],
            "calibrated_reasons": [],
            "hard_reasons": [],
        }
        if preupdate_rpr:
            if pre_update_logits is None:
                raise RuntimeError("Pre-update RPR was enabled but P_pre was not generated")
            pre_update_info = compute_mre_info(
                pre_update_logits,
                source_logits,
                prior,
                "pre_update",
                mre_area_min,
                mre_area_max,
                mre_center_max,
                mre_lambda_ap,
                topology_score_weight,
                topology_ring_kernel,
                True,
                True,
                True,
                True,
                True,
                True,
            )
            adapted_rpr_info = compute_mre_info(
                adapted_logits,
                source_logits,
                prior,
                "adapted",
                mre_area_min,
                mre_area_max,
                mre_center_max,
                mre_lambda_ap,
                topology_score_weight,
                topology_ring_kernel,
                True,
                True,
                True,
                True,
                True,
                True,
            )
            rpr_evidence = preupdate_rpr_evidence(
                pre_update_info,
                adapted_rpr_info,
                pre_update_logits,
                adapted_logits,
            )
            rpr_decision = decide_preupdate_rpr(
                rpr_evidence,
                calibrator=rpr_calibrator,
                vote_threshold=rpr_vote_threshold,
                entropy_margin=rpr_entropy_margin,
                prior_align_margin=rpr_prior_align_margin,
                center_margin=rpr_center_margin,
                topology_margin=rpr_topology_margin,
                confidence_margin=rpr_confidence_margin,
                drift_threshold=rpr_pre_adapt_drift_threshold,
                fg_min_ratio=rpr_fg_min_ratio,
                fg_max_ratio=rpr_fg_max_ratio,
            )
            risky = bool(rpr_decision["risky"])
            reason = (
                f"{reason},rpr={rpr_decision['decision_source']},"
                f"score={rpr_decision['decision_score']},"
                f"votes={rpr_decision['vote_reasons']},"
                f"calibrated={rpr_decision['calibrated_reasons']},"
                f"hard={rpr_decision['hard_reasons']}"
            )

        if not mre_selection_enabled:
            # Keep all candidates available for a controlled ablation, but bypass
            # reliability-based ranking and branch selection at the output.
            bank_name = "final"
            final_logits = candidate_logits["final"]
            if candidate_bank_v2:
                final_info = candidate_infos["final"]
            best = "final"
            reason = "mre_selection_disabled_fixed_final"

        oracle_values = {
            "best": None,
            "dice_source": None,
            "dice_adapted": None,
            "dice_final": None,
            "rv_dice_source": None,
            "rv_dice_adapted": None,
            "rv_dice_final": None,
            "selected_dice": None,
            "best_gain": None,
        }
        if oracle_audit:
            oracle_scores = {
                "source": candidate_dice_summary(source_logits, label_np, metric_fn, int(cfg.MODEL.NUMBER_CLASS)),
                "adapted": candidate_dice_summary(adapted_logits, label_np, metric_fn, int(cfg.MODEL.NUMBER_CLASS)),
                "final": candidate_dice_summary(final_logits, label_np, metric_fn, int(cfg.MODEL.NUMBER_CLASS)),
            }
            oracle_best = max(oracle_scores, key=lambda key: oracle_scores[key]["avg"])
            oracle_values = {
                "best": oracle_best,
                "dice_source": oracle_scores["source"]["avg"],
                "dice_adapted": oracle_scores["adapted"]["avg"],
                "dice_final": oracle_scores["final"]["avg"],
                "rv_dice_source": oracle_scores["source"]["class_3"],
                "rv_dice_adapted": oracle_scores["adapted"]["class_3"],
                "rv_dice_final": oracle_scores["final"]["class_3"],
                "selected_dice": oracle_scores[best]["avg"],
                "best_gain": oracle_scores[oracle_best]["avg"] - oracle_scores[best]["avg"],
            }

        if candidate_oracle_audit:
            oracle_candidate_logits = {
                "source": source_logits,
                "adapted": adapted_logits,
                "refined": refined_logits,
            }
            if pre_update_logits is not None:
                oracle_candidate_logits["pre_update"] = pre_update_logits
            oracle_candidate_logits.update(candidate_logits)
            oracle_candidate_infos: Dict[str, Dict[str, float]] = {
                "source": source_info,
                "adapted": adapted_info,
            }
            if pre_update_info is not None:
                oracle_candidate_infos["pre_update"] = pre_update_info
            oracle_candidate_infos.update(candidate_infos)
            if "refined" not in oracle_candidate_infos:
                if use_legacy_selector:
                    oracle_candidate_infos["refined"] = reliability_info(refined_logits, source_logits, "refined")
                else:
                    oracle_candidate_infos["refined"] = compute_mre_info(
                        refined_logits,
                        source_logits,
                        prior,
                        "refined",
                        mre_area_min,
                        mre_area_max,
                        mre_center_max,
                        mre_lambda_ap,
                        topology_score_weight,
                        topology_ring_kernel,
                        mre_use_entropy,
                        mre_use_prior_align,
                        mre_use_area,
                        mre_use_center,
                        mre_use_prior_conf,
                        mre_use_topology,
                    )
            for cand_name, cand_logits in oracle_candidate_logits.items():
                metrics = candidate_metric_summary(
                    cand_logits,
                    label_np,
                    metric_fn,
                    int(cfg.MODEL.NUMBER_CLASS),
                    include_assd=bool(candidate_oracle_assd),
                    assd_protocol=surface_assd_protocol,
                )
                info = oracle_candidate_infos.get(cand_name, {})
                candidate_oracle_rows.append(
                    [
                        batch_idx,
                        ";".join(os.path.basename(str(x)) for x in names),
                        cand_name,
                        int(cand_name == bank_name),
                        int((best == "source" and cand_name == "source") or (best == "adapted" and cand_name == "adapted") or (best == "final" and cand_name == bank_name)),
                        best,
                        bank_name,
                        info.get("score"),
                        info.get("score_generic"),
                        info.get("score_anatomical"),
                        info.get("entropy"),
                        info.get("prior_align_mean"),
                        info.get("fg_area"),
                        info.get("center_dist"),
                        info.get("geometry"),
                        info.get("rv_soft_area"),
                        info.get("rv_pred_area"),
                        info.get("rv_confidence"),
                        metrics["dice_avg"],
                        metrics["dice_class_1"],
                        metrics["dice_class_2"],
                        metrics["dice_class_3"],
                        metrics["assd_avg"],
                        metrics["assd_class_1"],
                        metrics["assd_class_2"],
                        metrics["assd_class_3"],
                    ]
                )

        if (
            rpr_paired_probe
            and risky
            and snap_before_a3 is not None
            and snap_after_a3 is not None
            and probe_max_horizon > 0
            and (
                rpr_probe_max_events is None
                or paired_probe_events < int(rpr_probe_max_events)
            )
        ):
            event_index = paired_probe_events
            event_rows = probe_future_state_pair(
                model=a3_model,
                source_model=source_model,
                pre_state=snap_before_a3,
                candidate_state=snap_after_a3,
                dataset=dataset,
                current_batch_idx=batch_idx,
                batch_size=batch_size,
                max_horizon=probe_max_horizon,
                device=device,
                calibration_corruption=calibration_corruption,
                metric_fn=metric_fn,
                number_class=int(cfg.MODEL.NUMBER_CLASS),
                surface_assd_protocol=surface_assd_protocol,
                prior_kwargs={
                    "source_blend": prior_source_blend,
                    "smooth_sigma": prior_smooth_sigma,
                    "generic_sigma": prior_generic_sigma,
                    "min_prior": prior_min,
                    "topology_strength": topology_prior_strength,
                    "topology_rv_strength": topology_rv_strength,
                    "topology_ring_kernel": topology_ring_kernel,
                    "presence_preserving": prior_presence_preserving,
                    "presence_classes": prior_presence_classes,
                    "presence_min_pixels": prior_presence_min_pixels,
                    "presence_area_reference": prior_presence_area_reference,
                    "presence_confidence_low": prior_presence_confidence_low,
                    "presence_confidence_high": prior_presence_confidence_high,
                    "presence_floor": prior_presence_floor,
                },
                apl_kwargs={
                    "apl_weight": apl_online_weight,
                    "entropy_lambda": prior_entropy_lambda,
                    "loc_lambda": prior_loc_lambda,
                    "bce_lambda": prior_bce_lambda,
                    "ts_lambda": prior_ts_lambda,
                    "source_consistency_lambda": prior_source_consistency_lambda,
                    "weight_gamma": prior_weight_gamma,
                    "weight_min": prior_weight_min,
                    "class_weights": prior_class_weights,
                    "consistency_class_weights": prior_consistency_class_weights,
                    "temporal_consistency_lambda": (
                        temporal_consistency_lambda if temporal_consistency else 0.0
                    ),
                    "temporal_consistency_class_weights": temporal_consistency_class_weights,
                    "temporal_consistency_conf_threshold": temporal_consistency_conf_threshold,
                    "temporal_consistency_rv_conf_threshold": temporal_consistency_rv_conf_threshold,
                    "temporal_consistency_prior_threshold": temporal_consistency_prior_threshold,
                    "temporal_consistency_temperature": temporal_consistency_temperature,
                },
            )
            event_names = ";".join(os.path.basename(str(x)) for x in names)
            for probe_row in event_rows:
                paired_probe_rows.append(
                    [
                        domain_label,
                        event_index,
                        batch_idx,
                        event_names,
                        rpr_decision["decision_source"],
                        rpr_decision["decision_score"],
                        rpr_decision["vote_count"],
                        ";".join(str(x) for x in rpr_decision["vote_reasons"]),
                        ";".join(str(x) for x in rpr_decision["calibrated_reasons"]),
                        ";".join(str(x) for x in rpr_decision["hard_reasons"]),
                        probe_row["future_step"],
                        int(probe_row["future_step"] in probe_horizons),
                        probe_row["future_batch_idx"],
                        probe_row["future_names"],
                        *[
                            probe_row[f"{branch}_{stage}_{metric}_{suffix}"]
                            for branch in (
                                "accept",
                                "revert",
                                "revert_gain",
                                "revert_repeat",
                                "null_repeat_delta",
                            )
                            for stage in ("pre", "post")
                            for metric in ("dice", "assd")
                            for suffix in ("avg", "class_1", "class_2", "class_3")
                        ],
                    ]
                )
            paired_probe_events += 1

        # RAR protects the online stream even when the current output remains a
        # refined candidate. This is distinct from output-branch selection.
        # Source inference has no online parameter state. Its APSA result can
        # validate APL/MRE, but RAR is explicitly not applicable.
        risk_reverted = bool(
            risk_aware_reversion
            and str(adapter_name).strip().lower() != "source"
            and rollback_state
            and risky
        )
        rpr_checkpoint_committed = False
        rpr_checkpoint_rollback_span = 0
        if risk_reverted:
            if checkpoint_controller is None:
                restore_a3_state(a3_model, snap_before_a3)
                rpr_checkpoint_rollback_span = 1
            else:
                checkpoint_state, rpr_checkpoint_rollback_span = checkpoint_controller.reject(
                    snap_before_a3
                )
                # A lagged checkpoint restores only the declared online model
                # and optimizer state. It must not rewind the global RNG or
                # data-stream progression to an earlier batch.
                restore_a3_state(a3_model, checkpoint_state, restore_rng=False)

        state_best = best
        if best == "source":
            if rollback_state and not risk_reverted:
                restore_a3_state(a3_model, snap_before_a3)
            elif prior_guided and not prior_refine_clone:
                restore_a3_state(a3_model, snap_after_a3)
            chosen_logits = source_logits
        elif best == "adapted":
            if prior_guided and not prior_refine_clone:
                restore_a3_state(a3_model, snap_after_a3)
            chosen_logits = adapted_logits
            accepted_count += 1
        else:
            chosen_logits = final_logits
            if prior_guided and not keep_prior_state and not prior_refine_clone:
                restore_a3_state(a3_model, snap_after_a3)
            accepted_count += 1

        if checkpoint_controller is not None and not risk_reverted:
            rpr_checkpoint_committed = checkpoint_controller.accept(
                snapshot_a3_state(a3_model)
            )

        if (
            (safe_ema_candidate or temporal_consistency)
            and not risk_reverted
            and best in ("adapted", "final")
            and hasattr(a3_model, "update_safe_ema")
        ):
            a3_model.update_safe_ema()

        output_best = best
        forced_output_branch: Optional[str] = None
        samplewise_counts: Counter = Counter()
        samplewise_bank_counts: Counter = Counter()
        sample_choices: List[str] = []
        sample_selected_candidates: List[str] = []
        classwise_selected: Dict[int, str] = {}
        force_key = str(force_output_candidate or "").strip()
        if force_key:
            forced_candidates = {
                "source": source_logits,
                "adapted": adapted_logits,
                "refined": refined_logits,
            }
            forced_candidates.update(candidate_logits)
            if force_key not in forced_candidates:
                available = ",".join(sorted(forced_candidates))
                raise ValueError(f"force_output_candidate={force_key!r} not available. Available: {available}")
            chosen_logits = forced_candidates[force_key]
            output_best = f"force_{force_key}"
            forced_output_branch = "source" if force_key == "source" else "adapted" if force_key == "adapted" else "final"
            reason = f"{reason},force_output_candidate={force_key}"
        elif classwise_candidate_select and not use_legacy_selector and state_best != "source":
            classwise_candidates = {
                "state": chosen_logits,
                "source": source_logits,
                "adapted": adapted_logits,
                "refined": refined_logits,
            }
            classwise_candidates.update(candidate_logits)
            chosen_logits, classwise_selected, classwise_scores = select_classwise_candidate_logits(
                chosen_logits,
                classwise_candidates,
                source_logits,
                prior,
                classwise_candidate_names or [],
                name_bias=classwise_candidate_name_bias,
                min_gain=classwise_candidate_min_gain,
                compose=classwise_candidate_compose,
                logit_boost=classwise_candidate_logit_boost,
            )
            output_best = "classwise"
            score_brief = {
                int(cls_idx): {k: round(float(v), 3) for k, v in scores.items() if v is not None and k in ("best_score", "state_score")}
                for cls_idx, scores in classwise_scores.items()
            }
            reason += f",state_best={state_best},classwise_selected={classwise_selected},classwise_scores={score_brief}"
        elif samplewise_mre_output and not use_legacy_selector:
            chosen_parts: List[torch.Tensor] = []
            for sample_idx in range(int(source_logits.shape[0])):
                prior_i = _slice_prior(prior, sample_idx)
                src_i = source_logits[sample_idx : sample_idx + 1]
                ada_i = adapted_logits[sample_idx : sample_idx + 1]
                source_info_i = compute_mre_info(
                    src_i,
                    src_i,
                    prior_i,
                    "source",
                    mre_area_min,
                    mre_area_max,
                    mre_center_max,
                    mre_lambda_ap,
                    topology_score_weight,
                    topology_ring_kernel,
                    mre_use_entropy,
                    mre_use_prior_align,
                    mre_use_area,
                    mre_use_center,
                    mre_use_prior_conf,
                    mre_use_topology,
                )
                adapted_info_i = compute_mre_info(
                    ada_i,
                    src_i,
                    prior_i,
                    "adapted",
                    mre_area_min,
                    mre_area_max,
                    mre_center_max,
                    mre_lambda_ap,
                    topology_score_weight,
                    topology_ring_kernel,
                    mre_use_entropy,
                    mre_use_prior_align,
                    mre_use_area,
                    mre_use_center,
                    mre_use_prior_conf,
                    mre_use_topology,
                )
                if candidate_bank_v2:
                    sample_candidate_infos = {}
                    for cand_name, cand_logits in candidate_logits.items():
                        sample_candidate_infos[cand_name] = compute_mre_info(
                            cand_logits[sample_idx : sample_idx + 1],
                            src_i,
                            prior_i,
                            cand_name,
                            mre_area_min,
                            mre_area_max,
                            mre_center_max,
                            mre_lambda_ap,
                            topology_score_weight,
                            topology_ring_kernel,
                            mre_use_entropy,
                            mre_use_prior_align,
                            mre_use_area,
                            mre_use_center,
                            mre_use_prior_conf,
                            mre_use_topology,
                        )
                    sample_bank_name = max(
                        sample_candidate_infos,
                        key=lambda key: candidate_bank_adjusted_score(
                            key, sample_candidate_infos[key], candidate_bank_name_bias
                        ),
                    )
                    sample_bank_score = candidate_bank_adjusted_score(
                        sample_bank_name, sample_candidate_infos[sample_bank_name], candidate_bank_name_bias
                    )
                    sample_final_score = candidate_bank_adjusted_score(
                        "final", sample_candidate_infos["final"], candidate_bank_name_bias
                    )
                    if safe_ema_candidate:
                        sample_bank_name, sample_bank_score = guard_safe_ema_candidate(
                            sample_bank_name,
                            sample_candidate_infos,
                            candidate_bank_name_bias,
                            safe_ema_candidate_min_gain,
                        )
                    if soft_anatomy_candidate_bank and sample_bank_name in ("anatomy_final", "clean_anatomy_final"):
                        reference_name = "clean_final" if sample_bank_name == "clean_anatomy_final" else "final"
                        reference_score = candidate_bank_adjusted_score(
                            reference_name, sample_candidate_infos[reference_name], candidate_bank_name_bias
                        )
                        if sample_bank_score < reference_score + float(soft_anatomy_candidate_min_gain):
                            sample_bank_name = reference_name
                            sample_bank_score = reference_score
                    if sample_bank_name != "final" and sample_bank_score < sample_final_score + float(candidate_bank_min_gain):
                        sample_bank_name = "final"
                    fin_i = candidate_logits[sample_bank_name][sample_idx : sample_idx + 1]
                    final_info_i = sample_candidate_infos[sample_bank_name]
                    samplewise_bank_counts[sample_bank_name] += 1
                else:
                    sample_bank_name = "final"
                    fin_i = final_logits[sample_idx : sample_idx + 1]
                    final_info_i = compute_mre_info(
                        fin_i,
                        src_i,
                        prior_i,
                        "final",
                        mre_area_min,
                        mre_area_max,
                        mre_center_max,
                        mre_lambda_ap,
                        topology_score_weight,
                        topology_ring_kernel,
                        mre_use_entropy,
                        mre_use_prior_align,
                        mre_use_area,
                        mre_use_center,
                        mre_use_prior_conf,
                        mre_use_topology,
                    )
                sample_best, _ = safe_branch_select_mre(
                    source_info_i,
                    adapted_info_i,
                    final_info_i,
                    conservative=safe_branch_selector_conservative,
                    adaptive=safe_branch_selector_adaptive,
                    allow_source_output=allow_source_output,
                    source_output_penalty=source_output_penalty,
                    final_min_gain=final_min_gain,
                    final_bias=final_bias,
                    final_geometry_guard=final_geometry_guard,
                    use_entropy=mre_use_entropy,
                    use_prior_align=mre_use_prior_align,
                    use_area=mre_use_area,
                    use_topology=mre_use_topology,
                    topology_guard=topology_guard,
                    use_rv_guard=rv_guard,
                    rv_guard_min_area=rv_guard_min_area,
                    rv_guard_min_ratio=rv_guard_min_ratio,
                    rv_guard_conf_drop=rv_guard_conf_drop,
                    final_takeover_v2=final_takeover_v2,
                    final_takeover_score_gain=final_takeover_score_gain,
                    final_takeover_geometry_gain=final_takeover_geometry_gain,
                    final_takeover_align_gain=final_takeover_align_gain,
                    final_takeover_entropy_increase=final_takeover_entropy_increase,
                    source_rescue_v2=source_rescue_v2,
                    source_rescue_margin=source_rescue_margin,
                    source_rescue_min_geometry=source_rescue_min_geometry,
                    source_rescue_rv_min_ratio=source_rescue_rv_min_ratio,
                    source_rescue_rv_max_ratio=source_rescue_rv_max_ratio,
                    source_rescue_align_margin=source_rescue_align_margin,
                    source_rescue_entropy_margin=source_rescue_entropy_margin,
                )
                sample_choices.append(sample_best)
                sample_selected_candidates.append(sample_bank_name if sample_best == "final" else sample_best)
                samplewise_counts[sample_best] += 1
                if sample_best == "source":
                    chosen_parts.append(src_i)
                elif sample_best == "adapted":
                    chosen_parts.append(ada_i)
                else:
                    chosen_parts.append(fin_i)
            chosen_logits = torch.cat(chosen_parts, dim=0)
            output_best = sample_choices[0] if len(samplewise_counts) == 1 else "mixed"
            reason += f",state_best={state_best},samplewise_output={dict(samplewise_counts)},samplewise_bank={dict(samplewise_bank_counts)}"
        risky_count += int(risky)
        risk_reverted_count += int(risk_reverted)
        rpr_checkpoint_commit_count += int(rpr_checkpoint_committed)
        rpr_checkpoint_rollback_span_total += int(rpr_checkpoint_rollback_span)
        total_batches += 1
        if forced_output_branch is not None:
            branch_counter[forced_output_branch] += int(source_logits.shape[0]) if samplewise_mre_output else 1
        elif samplewise_mre_output and samplewise_counts:
            branch_counter.update(samplewise_counts)
        elif classwise_selected:
            branch_counter["final"] += 1
        else:
            branch_counter[best] += 1

        trace_logits: Dict[str, torch.Tensor] = {
            "source": source_logits,
            "adapted": adapted_logits,
            "refined": refined_logits,
        }
        if pre_update_logits is not None:
            trace_logits["pre_update"] = pre_update_logits
        trace_logits.update(candidate_logits)
        pred = chosen_logits.argmax(dim=1).detach().cpu().numpy()
        for i, name in enumerate(names):
            image_name = os.path.basename(str(name))
            if force_key:
                selected_internal = force_key
            elif classwise_selected:
                selected_internal = "classwise:" + json.dumps(classwise_selected, sort_keys=True)
            elif sample_selected_candidates:
                selected_internal = sample_selected_candidates[i]
            elif best == "final":
                selected_internal = bank_name
            else:
                selected_internal = best
            selected_candidate = trace_candidate_name(selected_internal) if not selected_internal.startswith("classwise:") else selected_internal
            case_summary_rows.append(
                [domain_label, image_name, batch_idx, adapter_name, 1, selected_candidate, int(risk_reverted)]
            )
            prior_i = _slice_prior(prior, i)
            src_i = source_logits[i : i + 1]
            for candidate_name, candidate_batch_logits in trace_logits.items():
                candidate_i = candidate_batch_logits[i : i + 1]
                info_i = compute_mre_info(
                    candidate_i,
                    src_i,
                    prior_i,
                    candidate_name,
                    mre_area_min,
                    mre_area_max,
                    mre_center_max,
                    mre_lambda_ap,
                    topology_score_weight,
                    topology_ring_kernel,
                    mre_use_entropy,
                    mre_use_prior_align,
                    mre_use_area,
                    mre_use_center,
                    mre_use_prior_conf,
                    mre_use_topology,
                )
                trace_name = trace_candidate_name(candidate_name)
                candidate_trace_rows.append(
                    [
                        domain_label,
                        image_name,
                        batch_idx,
                        adapter_name,
                        1,
                        trace_name,
                        info_i.get("score"),
                        info_i.get("entropy"),
                        info_i.get("geometry"),
                        info_i.get("prior_align_mean"),
                        selected_candidate,
                        int(trace_name == selected_candidate),
                        int(risk_reverted),
                    ]
                )
            if mask_out_dir is not None and (save_pred_masks_all or image_name in save_case_names):
                mask_img = sitk.GetImageFromArray(pred[i].astype(np.uint8))
                sitk.WriteImage(mask_img, str(mask_out_dir / image_name))
                if save_stage_masks:
                    stage_logits = {
                        "P_src": source_logits,
                        "P_adapt": adapted_logits,
                        "P_ref": refined_logits,
                        "P_final": chosen_logits,
                    }
                    if pre_update_logits is not None:
                        stage_logits["P_pre"] = pre_update_logits
                    for stage_name, stage_batch_logits in stage_logits.items():
                        stage_dir = mask_out_dir / "stages" / stage_name
                        stage_dir.mkdir(parents=True, exist_ok=True)
                        stage_pred = stage_batch_logits[i].argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
                        sitk.WriteImage(sitk.GetImageFromArray(stage_pred), str(stage_dir / image_name))
            dice = metric_fn(pred[i], label_np[i], int(cfg.MODEL.NUMBER_CLASS), "dice")
            assd, hd95 = surface_metrics_per_class(
                pred[i],
                label_np[i],
                int(cfg.MODEL.NUMBER_CLASS),
                assd_protocol=surface_assd_protocol,
                metric_fn=metric_fn,
            )
            dice = [float(x) for x in dice]
            dice_all.append(dice)
            assd_all.append(assd)
            hd95_all.append(hd95)
            dice_rows.append([image_name] + dice + [float(np.mean(dice))])
            assd_rows.append([image_name] + assd + [float(np.mean(assd))])
            hd95_rows.append([image_name] + hd95 + [nanmean_or_nan(hd95)])
            seen += 1

        audit_rows.append(
            [
                batch_idx,
                ";".join(os.path.basename(str(x)) for x in names),
                output_best,
                reason,
                int(risky),
                int(risk_reverted),
                source_info["score"],
                adapted_info["score"],
                final_info["score"],
                source_info["entropy"],
                adapted_info["entropy"],
                final_info["entropy"],
                adapted_info["drift"],
                adapted_info["geometry"],
                final_info["geometry"],
                source_info["fg_area"],
                adapted_info["fg_area"],
                final_info["fg_area"],
                source_info.get("score_generic"),
                adapted_info.get("score_generic"),
                final_info.get("score_generic"),
                source_info.get("score_anatomical"),
                adapted_info.get("score_anatomical"),
                final_info.get("score_anatomical"),
                source_info.get("prior_align_mean"),
                adapted_info.get("prior_align_mean"),
                final_info.get("prior_align_mean"),
                source_info.get("area_ok"),
                adapted_info.get("area_ok"),
                final_info.get("area_ok"),
                source_info.get("center_dist"),
                adapted_info.get("center_dist"),
                final_info.get("center_dist"),
                source_info.get("prior_q_mean"),
                adapted_info.get("prior_q_mean"),
                final_info.get("prior_q_mean"),
                source_info.get("topology_score"),
                adapted_info.get("topology_score"),
                final_info.get("topology_score"),
                source_info.get("rv_soft_area"),
                adapted_info.get("rv_soft_area"),
                final_info.get("rv_soft_area"),
                source_info.get("rv_pred_area"),
                adapted_info.get("rv_pred_area"),
                final_info.get("rv_pred_area"),
                source_info.get("rv_confidence"),
                adapted_info.get("rv_confidence"),
                final_info.get("rv_confidence"),
                oracle_values["best"],
                oracle_values["dice_source"],
                oracle_values["dice_adapted"],
                oracle_values["dice_final"],
                oracle_values["rv_dice_source"],
                oracle_values["rv_dice_adapted"],
                oracle_values["rv_dice_final"],
                oracle_values["selected_dice"],
                oracle_values["best_gain"],
                refine_stats["loss"],
                refine_stats["ent"],
                refine_stats["loc"],
                refine_stats["ce"],
                refine_stats["boundary"],
                refine_stats["ts"],
                refine_stats["src_cons"],
                getattr(a3_model, "last_temporal_consistency_loss", 0.0),
                getattr(a3_model, "last_temporal_reliability_fraction", 0.0),
                rv_image_stats["trigger_fraction"],
                rv_image_stats["support_fraction"],
                rv_image_stats["changed_fraction"],
                rv_image_stats["mean_probability_delta"],
                rv_atlas_stats["trigger_fraction"],
                rv_atlas_stats["support_fraction"],
                rv_atlas_stats["changed_fraction"],
                rv_atlas_stats["hard_changed_fraction"],
                rv_atlas_stats["mean_probability_delta"],
                rv_atlas_stats["mean_atlas_presence"],
                rv_slice_stats["neighbor_seed_fraction"],
                rv_slice_stats["trigger_fraction"],
                rv_slice_stats["mean_neighbor_count"],
                rv_slice_stats["support_fraction"],
                rv_slice_stats["changed_fraction"],
                rv_slice_stats["mean_probability_delta"],
                rpr_decision["decision_source"],
                rpr_decision["decision_score"],
                rpr_decision["vote_count"],
                ";".join(str(x) for x in rpr_decision["vote_reasons"]),
                ";".join(str(x) for x in rpr_decision["calibrated_reasons"]),
                ";".join(str(x) for x in rpr_decision["hard_reasons"]),
                *[
                    rpr_evidence.get("features", {}).get(feature_name)
                    for feature_name in PREUPDATE_RPR_FEATURES
                ],
                checkpoint_interval,
                int(rpr_checkpoint_committed),
                int(rpr_checkpoint_rollback_span),
            ]
        )

    dice_np = np.asarray(dice_all, dtype=np.float32)
    assd_np = np.asarray(assd_all, dtype=np.float32)
    hd95_np = np.asarray(hd95_all, dtype=np.float32)
    if dice_np.size == 0:
        raise RuntimeError(f"No cases were evaluated for domain {domain_label}")
    dice_mean, dice_std = dice_np.mean(axis=0), dice_np.std(axis=0)
    assd_mean, assd_std = assd_np.mean(axis=0), assd_np.std(axis=0)
    hd95_mean = np.asarray([nanmean_or_nan(hd95_np[:, i]) for i in range(hd95_np.shape[1])], dtype=np.float32)
    hd95_std = np.asarray(
        [float(np.std(hd95_np[np.isfinite(hd95_np[:, i]), i])) if np.isfinite(hd95_np[:, i]).any() else float("nan") for i in range(hd95_np.shape[1])],
        dtype=np.float32,
    )
    hd95_valid = np.isfinite(hd95_np).sum(axis=0).astype(np.int64)

    dice_rows.append(["mean"] + list(dice_mean) + [float(dice_mean.mean())])
    dice_rows.append(["std"] + list(dice_std) + [float(dice_std.mean())])
    assd_rows.append(["mean"] + list(assd_mean) + [float(assd_mean.mean())])
    assd_rows.append(["std"] + list(assd_std) + [float(assd_std.mean())])
    hd95_rows.append(["mean"] + list(hd95_mean) + [nanmean_or_nan(hd95_mean)])
    hd95_rows.append(["std"] + list(hd95_std) + [nanmean_or_nan(hd95_std)])
    hd95_rows.append(["valid_count"] + list(hd95_valid) + [int(hd95_valid.sum())])

    metric_head = ["image"] + CLASS_NAMES + ["average"]
    _write_rows(domain_dir / "dice_all.csv", metric_head, dice_rows)
    _write_rows(domain_dir / "assd_all.csv", metric_head, assd_rows)
    _write_rows(domain_dir / "hd95_all.csv", metric_head, hd95_rows)
    _write_rows(
        domain_dir / "case_summary.csv",
        ["domain", "case_id", "batch_idx", "adapter", "use_apssa", "selected_candidate", "rollback_flag"],
        case_summary_rows,
    )
    _write_rows(
        domain_dir / "candidate_trace.csv",
        [
            "domain",
            "case_id",
            "batch_idx",
            "adapter",
            "use_apssa",
            "candidate_name",
            "mre_score",
            "entropy",
            "geometry",
            "prior_align",
            "selected_candidate",
            "is_selected",
            "rollback_flag",
        ],
        candidate_trace_rows,
    )
    _write_rows(
        domain_dir / "branch_audit.csv",
        [
            "batch_idx",
            "names",
            "selected",
            "reason",
            "risky",
            "risk_reverted",
            "score_source",
            "score_adapted",
            "score_final",
            "entropy_source",
            "entropy_adapted",
            "entropy_final",
            "drift_adapted",
            "geometry_adapted",
            "geometry_final",
            "fg_area_source",
            "fg_area_adapted",
            "fg_area_final",
            "score_generic_source",
            "score_generic_adapted",
            "score_generic_final",
            "score_anatomical_source",
            "score_anatomical_adapted",
            "score_anatomical_final",
            "prior_align_source",
            "prior_align_adapted",
            "prior_align_final",
            "area_ok_source",
            "area_ok_adapted",
            "area_ok_final",
            "center_dist_source",
            "center_dist_adapted",
            "center_dist_final",
            "prior_q_source",
            "prior_q_adapted",
            "prior_q_final",
            "topology_score_source",
            "topology_score_adapted",
            "topology_score_final",
            "rv_soft_area_source",
            "rv_soft_area_adapted",
            "rv_soft_area_final",
            "rv_pred_area_source",
            "rv_pred_area_adapted",
            "rv_pred_area_final",
            "rv_confidence_source",
            "rv_confidence_adapted",
            "rv_confidence_final",
            "oracle_best",
            "oracle_dice_source",
            "oracle_dice_adapted",
            "oracle_dice_final",
            "oracle_rv_dice_source",
            "oracle_rv_dice_adapted",
            "oracle_rv_dice_final",
            "oracle_selected_dice",
            "oracle_best_gain",
            "prior_refine_loss",
            "prior_refine_entropy",
            "prior_refine_loc",
            "prior_refine_ce",
            "prior_refine_boundary",
            "prior_refine_ts",
            "prior_refine_src_cons",
            "online_temporal_consistency_loss",
            "online_temporal_reliability_fraction",
            "rv_image_trigger_fraction",
            "rv_image_support_fraction",
            "rv_image_changed_fraction",
            "rv_image_mean_probability_delta",
            "rv_atlas_trigger_fraction",
            "rv_atlas_support_fraction",
            "rv_atlas_changed_fraction",
            "rv_atlas_hard_changed_fraction",
            "rv_atlas_mean_probability_delta",
            "rv_atlas_mean_presence",
            "rv_slice_neighbor_seed_fraction",
            "rv_slice_trigger_fraction",
            "rv_slice_mean_neighbor_count",
            "rv_slice_support_fraction",
            "rv_slice_changed_fraction",
            "rv_slice_mean_probability_delta",
            "rpr_decision_source",
            "rpr_decision_score",
            "rpr_vote_count",
            "rpr_vote_reasons",
            "rpr_calibrated_reasons",
            "rpr_hard_reasons",
            *[f"rpr_{feature_name}" for feature_name in PREUPDATE_RPR_FEATURES],
            "rpr_checkpoint_interval",
            "rpr_checkpoint_committed",
            "rpr_checkpoint_rollback_span",
        ],
        audit_rows,
    )
    if candidate_oracle_audit:
        _write_rows(
            domain_dir / "candidate_oracle.csv",
            [
                "batch_idx",
                "names",
                "candidate",
                "is_bank_candidate",
                "is_batch_selected_output",
                "batch_selected_branch",
                "bank_candidate",
                "mre_score",
                "mre_score_generic",
                "mre_score_anatomical",
                "entropy",
                "prior_align",
                "fg_area",
                "center_dist",
                "geometry",
                "rv_soft_area",
                "rv_pred_area",
                "rv_confidence",
                "dice_avg",
                "dice_class_1",
                "dice_class_2",
                "dice_class_3",
                "assd_avg",
                "assd_class_1",
                "assd_class_2",
                "assd_class_3",
            ],
            candidate_oracle_rows,
        )
    if rpr_paired_probe:
        _write_rows(
            domain_dir / "paired_future_probe.csv",
            [
                "domain",
                "event_index",
                "event_batch_idx",
                "event_names",
                "rpr_decision_source",
                "rpr_decision_score",
                "rpr_vote_count",
                "rpr_vote_reasons",
                "rpr_calibrated_reasons",
                "rpr_hard_reasons",
                "future_step",
                "is_requested_horizon",
                "future_batch_idx",
                "future_names",
                *[
                    f"{branch}_{stage}_{metric}_{suffix}"
                    for branch in (
                        "accept",
                        "revert",
                        "revert_gain",
                        "revert_repeat",
                        "null_repeat_delta",
                    )
                    for stage in ("pre", "post")
                    for metric in ("dice", "assd")
                    for suffix in ("avg", "class_1", "class_2", "class_3")
                ],
            ],
            paired_probe_rows,
        )

    return {
        "domain": domain_label,
        "dice_mean": dice_mean,
        "dice_std": dice_std,
        "assd_mean": assd_mean,
        "assd_std": assd_std,
        "hd95_mean": hd95_mean,
        "hd95_std": hd95_std,
        "hd95_valid": hd95_valid,
        "selected_source": branch_counter["source"],
        "selected_adapted": branch_counter["adapted"],
        "selected_final": branch_counter["final"],
        "accepted": accepted_count,
        "risky": risky_count,
        "risk_reverted": risk_reverted_count,
        "rpr_checkpoint_commits": rpr_checkpoint_commit_count,
        "rpr_checkpoint_rollback_span": rpr_checkpoint_rollback_span_total,
        "paired_probe_events": paired_probe_events,
        "paired_probe_rows": len(paired_probe_rows),
        "total_batches": total_batches,
        "total_cases": seen,
    }


def write_summary(results: List[Dict[str, object]], save_dir: Path) -> None:
    rows = []
    head = ["Domain"]
    for cls in CLASS_NAMES:
        head.extend([f"{cls}_Dice", f"{cls}_ASSD", f"{cls}_HD95", f"{cls}_HD95_Valid"])
    head.extend([
        "Avg_Dice",
        "Avg_ASSD",
        "Avg_HD95",
        "Selected_Source",
        "Selected_Adapted",
        "Selected_Final",
        "Accepted",
        "Risky",
        "Risk_Reverted",
        "RPR_Checkpoint_Commits",
        "RPR_Rollback_Span",
        "Batches",
        "Cases",
    ])

    for item in results:
        row = [item["domain"]]
        for i in range(len(CLASS_NAMES)):
            row.append(_format_pm(float(item["dice_mean"][i]), float(item["dice_std"][i])))
            row.append(_format_pm(float(item["assd_mean"][i]), float(item["assd_std"][i])))
            row.append(_format_pm(float(item["hd95_mean"][i]), float(item["hd95_std"][i])))
            row.append(int(item["hd95_valid"][i]))
        row.append(_format_pm(float(np.mean(item["dice_mean"])), float(np.mean(item["dice_std"]))))
        row.append(_format_pm(float(np.mean(item["assd_mean"])), float(np.mean(item["assd_std"]))))
        row.append(_format_pm(nanmean_or_nan(item["hd95_mean"]), nanmean_or_nan(item["hd95_std"])))
        row.extend([
            item["selected_source"],
            item["selected_adapted"],
            item["selected_final"],
            item["accepted"],
            item["risky"],
            item["risk_reverted"],
            item["rpr_checkpoint_commits"],
            item["rpr_checkpoint_rollback_span"],
            item["total_batches"],
            item["total_cases"],
        ])
        rows.append(row)

    mean_dice = np.mean([x["dice_mean"] for x in results], axis=0)
    std_dice = np.mean([x["dice_std"] for x in results], axis=0)
    mean_assd = np.mean([x["assd_mean"] for x in results], axis=0)
    std_assd = np.mean([x["assd_std"] for x in results], axis=0)
    all_hd95_mean = np.asarray([x["hd95_mean"] for x in results], dtype=np.float64)
    all_hd95_std = np.asarray([x["hd95_std"] for x in results], dtype=np.float64)
    mean_hd95 = np.asarray([nanmean_or_nan(all_hd95_mean[:, i]) for i in range(len(CLASS_NAMES))])
    std_hd95 = np.asarray([nanmean_or_nan(all_hd95_std[:, i]) for i in range(len(CLASS_NAMES))])
    valid_hd95 = np.sum([x["hd95_valid"] for x in results], axis=0)
    avg = ["Average"]
    for i in range(len(CLASS_NAMES)):
        avg.append(_format_pm(float(mean_dice[i]), float(std_dice[i])))
        avg.append(_format_pm(float(mean_assd[i]), float(std_assd[i])))
        avg.append(_format_pm(float(mean_hd95[i]), float(std_hd95[i])))
        avg.append(int(valid_hd95[i]))
    avg.append(_format_pm(float(np.mean(mean_dice)), float(np.mean(std_dice))))
    avg.append(_format_pm(float(np.mean(mean_assd)), float(np.mean(std_assd))))
    avg.append(_format_pm(nanmean_or_nan(mean_hd95), nanmean_or_nan(std_hd95)))
    for key in [
        "selected_source",
        "selected_adapted",
        "selected_final",
        "accepted",
        "risky",
        "risk_reverted",
        "rpr_checkpoint_commits",
        "rpr_checkpoint_rollback_span",
        "total_batches",
        "total_cases",
    ]:
        avg.append(int(sum(int(x[key]) for x in results)))
    rows.append(avg)

    _write_rows(save_dir / "summary_table.csv", head, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfgs/mms/a3-tta-mms2d.yaml")
    parser.add_argument("--adapter", choices=ADAPTER_NAMES, default="a3-full")
    parser.add_argument(
        "--use-apssa",
        type=int,
        choices=(0, 1),
        default=1,
        help="Run the shared APSA candidate/refinement/selection pipeline (1) or output the raw adapter prediction (0).",
    )
    parser.add_argument(
        "--strict_ablation_variant",
        choices=("", "base_bn_only", "plus_apl", "plus_apl_mre"),
        default="",
        help=(
            "Validate a clean incremental APSA ablation and record its effective "
            "modules. Any residual module or experimental candidate causes an error."
        ),
    )
    parser.add_argument(
        "--longstream_safety_variant",
        choices=("", "without_rpr", "with_rpr"),
        default="",
        help=(
            "Validate the paired B-C-D repeated-stream RPR safety experiment. "
            "The two variants may differ only in state rollback/RPR."
        ),
    )
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--stream_rounds", type=int, default=1, help="Repeat the specified target-domain stream this many times without model reset.")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument(
        "--source_calibration_corruption",
        choices=SOURCE_CALIBRATION_CORRUPTIONS,
        default="none",
        help="Deterministic source-A shift for MRE calibration data generation; rejected on target domains.",
    )
    parser.add_argument(
        "--surface_assd_protocol",
        choices=("scipy_exact", "sictta_capped"),
        default="scipy_exact",
        help=(
            "Cardiac ASSD evaluator. 'sictta_capped' reproduces the public "
            "A3-TTA/SiCTTA per-slice protocol (10-pixel cap; one-empty=10); "
            "'scipy_exact' preserves earlier clean-run outputs."
        ),
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tent_lr",
        type=float,
        default=None,
        help="Optional BN-only TENT learning rate.",
    )
    parser.add_argument(
        "--tent_train_scope",
        choices=TENT_TRAIN_SCOPES,
        default="bn",
        help="Generic U-Net parameters updated by entropy/APL adaptation; the default is BN affine only.",
    )
    parser.add_argument(
        "--disable_adaptation_dropout",
        action="store_true",
        help="Keep Dropout in eval mode while BN and APL parameters remain adaptive.",
    )
    parser.add_argument(
        "--safe_ema_candidate",
        action="store_true",
        help="Add a rollback-protected temporal EMA teacher as an APSA candidate; no A3 memory or anchor is used.",
    )
    parser.add_argument("--safe_ema_alpha", type=float, default=0.999)
    parser.add_argument("--safe_ema_apl_refine", action="store_true")
    parser.add_argument("--safe_ema_candidate_min_gain", type=float, default=0.0)
    parser.add_argument(
        "--temporal_consistency",
        action="store_true",
        help="Add MRE-style reliability-gated EMA consistency to BN-only APL adaptation.",
    )
    parser.add_argument("--temporal_consistency_lambda", type=float, default=0.0)
    parser.add_argument(
        "--temporal_consistency_class_weights",
        default="0,1,1,1",
        help="BG,LV,MYO,RV teacher-class gates for temporal consistency.",
    )
    parser.add_argument("--temporal_consistency_conf_threshold", type=float, default=0.70)
    parser.add_argument("--temporal_consistency_rv_conf_threshold", type=float, default=0.85)
    parser.add_argument("--temporal_consistency_prior_threshold", type=float, default=0.25)
    parser.add_argument("--temporal_consistency_temperature", type=float, default=0.50)
    parser.add_argument(
        "--prior_refine_lr",
        type=float,
        default=None,
        help="Optional APL candidate-refinement learning rate. Defaults to cfg.OPTIM.LR.",
    )
    parser.add_argument("--entropy_margin", type=float, default=0.02)
    parser.add_argument("--drift_threshold", type=float, default=0.18)
    parser.add_argument("--geometry_margin", type=float, default=0.45)
    parser.add_argument(
        "--preupdate_rpr",
        action="store_true",
        help=(
            "Evaluate the inherited online predictor before adaptation and base "
            "RPR on P_pre-to-P_adapt evidence instead of source-relative geometry."
        ),
    )
    parser.add_argument(
        "--rpr_calibrator",
        default="",
        help="Frozen Source-A pre-update RPR calibration JSON. Target labels are never loaded by it.",
    )
    parser.add_argument("--rpr_vote_threshold", type=int, default=2)
    parser.add_argument(
        "--rpr_checkpoint_interval",
        type=int,
        default=1,
        help=(
            "Commit a trusted RPR recovery checkpoint after this many consecutive "
            "safe updates. One preserves ordinary pre-update rollback; values above "
            "one enable checkpointed RPR+."
        ),
    )
    parser.add_argument("--rpr_entropy_margin", type=float, default=0.005)
    parser.add_argument("--rpr_prior_align_margin", type=float, default=0.005)
    parser.add_argument("--rpr_center_margin", type=float, default=0.015)
    parser.add_argument("--rpr_topology_margin", type=float, default=0.05)
    parser.add_argument("--rpr_confidence_margin", type=float, default=0.01)
    parser.add_argument("--rpr_pre_adapt_drift_threshold", type=float, default=0.04)
    parser.add_argument("--rpr_fg_min_ratio", type=float, default=0.50)
    parser.add_argument("--rpr_fg_max_ratio", type=float, default=2.50)
    parser.add_argument(
        "--rpr_paired_probe",
        action="store_true",
        help=(
            "At every natural RPR trigger, evaluate forced-accept and "
            "forced-revert states on identical future batches. Diagnostic "
            "only: target labels are used after the gate decision and the "
            "main online trajectory is unchanged."
        ),
    )
    parser.add_argument(
        "--rpr_probe_horizons",
        default="1,3,5",
        help="Comma-separated future-batch horizons reported by the paired RPR audit.",
    )
    parser.add_argument(
        "--rpr_probe_max_events",
        type=int,
        default=None,
        help="Optional smoke-test cap on probed natural RPR triggers.",
    )
    parser.add_argument("--source_fusion", type=float, default=0.35)
    parser.add_argument(
        "--source_fusion_classes",
        default=None,
        help="Optional comma-separated source fusion weights. Use LV,MYO,RV or BG,LV,MYO,RV.",
    )
    parser.add_argument(
        "--adaptive_class_fusion",
        action="store_true",
        help="Increase per-class source fusion when adapted class area/confidence drifts from source.",
    )
    parser.add_argument("--adaptive_fusion_strength", type=float, default=0.0)
    parser.add_argument("--adaptive_fusion_max", type=float, default=0.75)
    parser.add_argument("--candidate_bank_v2", action="store_true", help="Rank multiple APSA-generated final candidates with MRE before safe branch selection.")
    parser.add_argument(
        "--candidate_bank_fusion_scales",
        default="",
        help="Comma-separated multipliers for source_fusion/source_fusion_classes used to build extra fusion candidates.",
    )
    parser.add_argument("--candidate_bank_no_refined", action="store_true", help="Do not include the prior-refined candidate in candidate_bank_v2.")
    parser.add_argument(
        "--candidate_bank_tta_flips",
        default="",
        help="Comma-separated flip candidates for adapted logits: h, v, hv.",
    )
    parser.add_argument("--candidate_bank_min_gain", type=float, default=0.005)
    parser.add_argument(
        "--candidate_bank_name_bias",
        default="",
        help="Comma-separated candidate score calibration, e.g. clean_final:0.02,refined:-0.02.",
    )
    parser.add_argument(
        "--rv_composed_candidate_bank",
        action="store_true",
        help=(
            "Add candidates that keep clean-final non-RV decisions and transfer only each "
            "donor's RV-vs-non-RV margin. Disabled by default; requires --candidate_bank_v2."
        ),
    )
    parser.add_argument(
        "--rv_composed_candidate_names",
        default="clean_adapted,adapted,refined,final,fusion_x0.50,fusion_x0.70,fusion_x0.85,fusion_x1.00,fusion_x1.20,rv_atlas_final",
        help="Comma-separated donor candidates used by --rv_composed_candidate_bank.",
    )
    parser.add_argument(
        "--rv_consensus_candidate_bank",
        action="store_true",
        help=(
            "Add one guarded RV candidate from agreement among existing APSA candidates. "
            "Disabled by default; requires --candidate_bank_v2."
        ),
    )
    parser.add_argument(
        "--rv_consensus_candidate_names",
        default="adapted,refined,final,tta_v",
        help="Comma-separated donor candidates used by --rv_consensus_candidate_bank.",
    )
    parser.add_argument("--rv_consensus_min_agreement", type=int, default=3)
    parser.add_argument("--rv_consensus_confidence", type=float, default=0.50)
    parser.add_argument("--rv_consensus_base_margin_floor", type=float, default=-0.35)
    parser.add_argument("--rv_consensus_prior_threshold", type=float, default=0.10)
    parser.add_argument("--rv_consensus_protect_confidence", type=float, default=0.72)
    parser.add_argument("--rv_consensus_support_kernel", type=int, default=11)
    parser.add_argument("--rv_consensus_max_extra_ratio", type=float, default=0.35)
    parser.add_argument("--rv_consensus_strength", type=float, default=0.80)
    parser.add_argument(
        "--mre_calibrator",
        default="",
        help="Frozen source-domain MRE calibration JSON. Disabled by default; target labels are never loaded by it.",
    )
    parser.add_argument(
        "--rv_gain_selector",
        default="",
        help=(
            "Frozen Source-A nonlinear RV gain selector JSON. Disabled by default; requires "
            "RV-composed candidates and never loads target labels."
        ),
    )
    parser.add_argument("--candidate_cleanup_v2", action="store_true", help="Add mask-cleaned cardiac candidates to the APSA candidate bank.")
    parser.add_argument(
        "--disable_mre_selection",
        action="store_true",
        help="Ablation: bypass MRE candidate ranking and safe branch selection, and output the fixed P_final candidate.",
    )
    parser.add_argument("--candidate_cleanup_sources", default="final", help="Comma-separated cleanup sources: final, adapted, refined.")
    parser.add_argument("--cleanup_min_area_abs", type=int, default=16)
    parser.add_argument("--cleanup_min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--cleanup_logit_bias", type=float, default=4.0)
    parser.add_argument("--cleanup_keep_largest_classes", default="", help="Comma-separated class ids to keep only the largest component.")
    parser.add_argument("--cleanup_fill_hole_classes", default="1,2", help="Comma-separated class ids for binary hole filling.")
    parser.add_argument(
        "--anatomy_protected_cleanup",
        action="store_true",
        help="Add a source/prior-supported small-component cleanup candidate; the legacy clean candidates are retained.",
    )
    parser.add_argument("--cleanup_protected_sources", default="final")
    parser.add_argument("--cleanup_protected_classes", default="3")
    parser.add_argument("--cleanup_protection_min_area_abs", type=int, default=8)
    parser.add_argument("--cleanup_protection_candidate_confidence", type=float, default=0.85)
    parser.add_argument("--cleanup_protection_reference_overlap", type=float, default=0.20)
    parser.add_argument("--cleanup_protection_consensus_overlap", type=float, default=0.50)
    parser.add_argument("--cleanup_protection_prior_mean", type=float, default=0.20)
    parser.add_argument("--cleanup_protection_presence_margin", type=float, default=0.02)
    parser.add_argument("--cleanup_protection_max_components", type=int, default=1)
    parser.add_argument(
        "--volume_continuity_blend",
        type=float,
        default=0.0,
        help="Blend clean-final probabilities across adjacent slices of the same volume (0 disables).",
    )
    parser.add_argument("--final_bias", type=float, default=0.02)
    parser.add_argument("--final_min_gain", type=float, default=0.01)
    parser.add_argument(
        "--final_geometry_guard",
        type=float,
        default=0.0,
        help="Block final candidate when its source-relative geometry drops below adapted by this margin.",
    )
    parser.add_argument("--source_fallback_margin", type=float, default=0.75)
    parser.add_argument("--source_output_penalty", type=float, default=0.04)
    parser.add_argument(
        "--disable_anatomical_prior",
        action="store_true",
        help="Ablation: remove the spatial cardiac prior from both APL refinement and MRE prior-based cues.",
    )
    parser.add_argument("--no_prior_guidance", action="store_true", help="Disable cardiac prior-guided refinement update.")
    parser.add_argument("--prior_source_blend", type=float, default=0.65)
    parser.add_argument("--prior_smooth_sigma", type=float, default=2.0)
    parser.add_argument("--prior_generic_sigma", type=float, default=0.12)
    parser.add_argument("--prior_min", type=float, default=0.02)
    parser.add_argument(
        "--prior_presence_preserving",
        action="store_true",
        help="Preserve source class presence in APL priors and make prior foreground mass consistent across losses/cues.",
    )
    parser.add_argument("--prior_presence_classes", default="3")
    parser.add_argument("--prior_presence_min_pixels", type=int, default=4)
    parser.add_argument("--prior_presence_area_reference", type=int, default=32)
    parser.add_argument("--prior_presence_confidence_low", type=float, default=0.50)
    parser.add_argument("--prior_presence_confidence_high", type=float, default=0.90)
    parser.add_argument("--prior_presence_floor", type=float, default=0.02)
    parser.add_argument("--topology_prior_strength", type=float, default=0.55)
    parser.add_argument("--topology_rv_strength", type=float, default=0.0)
    parser.add_argument("--topology_ring_kernel", type=int, default=9)
    parser.add_argument("--soft_myo_ring_strength", type=float, default=0.0)
    parser.add_argument("--soft_rv_adjacency_strength", type=float, default=0.0)
    parser.add_argument("--soft_anatomy_ring_kernel", type=int, default=9)
    parser.add_argument("--soft_anatomy_rv_kernel", type=int, default=9)
    parser.add_argument("--soft_anatomy_gate_low", type=float, default=0.03)
    parser.add_argument("--soft_anatomy_gate_high", type=float, default=0.25)
    parser.add_argument("--soft_anatomy_max_delta", type=float, default=0.40)
    parser.add_argument(
        "--soft_anatomy_candidate_bank",
        action="store_true",
        help="Keep the original final branch and add soft-anatomy outputs as guarded MRE candidates.",
    )
    parser.add_argument("--soft_anatomy_candidate_min_gain", type=float, default=0.0)
    parser.add_argument(
        "--rv_image_candidate",
        action="store_true",
        help=(
            "Add a target-image-conditioned, LV/MYO-adjacent RV recovery candidate to the guarded "
            "MRE bank. Disabled by default; requires --candidate_bank_v2."
        ),
    )
    parser.add_argument("--rv_image_probability_cap", type=float, default=0.10)
    parser.add_argument("--rv_image_adjacency_kernel", type=int, default=31)
    parser.add_argument("--rv_image_trigger_ratio", type=float, default=0.30)
    parser.add_argument("--rv_image_max_area_ratio", type=float, default=1.50)
    parser.add_argument("--rv_image_background_margin", type=float, default=0.50)
    parser.add_argument("--rv_image_min_lv_pixels", type=int, default=32)
    parser.add_argument("--rv_image_lv_confidence", type=float, default=0.70)
    parser.add_argument("--rv_image_myo_confidence", type=float, default=0.45)
    parser.add_argument("--rv_image_intensity_sigma_scale", type=float, default=1.50)
    parser.add_argument(
        "--rv_atlas_candidate",
        action="store_true",
        help=(
            "Add a frozen Source-A RV atlas candidate aligned by predicted LV/MYO geometry. "
            "Disabled by default; requires --candidate_bank_v2 and --rv_atlas_path."
        ),
    )
    parser.add_argument("--rv_atlas_path", default="")
    parser.add_argument("--rv_atlas_probability_cap", type=float, default=0.18)
    parser.add_argument("--rv_atlas_support_threshold", type=float, default=0.10)
    parser.add_argument("--rv_atlas_max_area_scale", type=float, default=1.00)
    parser.add_argument("--rv_atlas_background_margin", type=float, default=0.45)
    parser.add_argument("--rv_atlas_min_lv_pixels", type=int, default=32)
    parser.add_argument("--rv_atlas_lv_confidence", type=float, default=0.55)
    parser.add_argument("--rv_atlas_myo_confidence", type=float, default=0.40)
    parser.add_argument("--rv_atlas_competitor_margin", type=float, default=0.005)
    parser.add_argument(
        "--rv_slice_candidate",
        action="store_true",
        help=(
            "Add a current-batch, same-case directional slice-consistency RV candidate to the "
            "guarded MRE bank. Disabled by default; requires --candidate_bank_v2."
        ),
    )
    parser.add_argument("--rv_slice_probability_cap", type=float, default=0.10)
    parser.add_argument("--rv_slice_max_gap", type=int, default=2)
    parser.add_argument("--rv_slice_require_two_sided", action="store_true")
    parser.add_argument("--rv_slice_seed_confidence", type=float, default=0.70)
    parser.add_argument("--rv_slice_min_seed_pixels", type=int, default=16)
    parser.add_argument("--rv_slice_min_lv_pixels", type=int, default=32)
    parser.add_argument("--rv_slice_trigger_neighbor_ratio", type=float, default=0.50)
    parser.add_argument("--rv_slice_max_area_scale", type=float, default=1.15)
    parser.add_argument("--rv_slice_support_kernel", type=int, default=11)
    parser.add_argument("--rv_slice_direction_sigma_scale", type=float, default=1.50)
    parser.add_argument("--rv_slice_direction_consistency", type=float, default=0.70)
    parser.add_argument("--rv_slice_background_margin", type=float, default=0.35)
    parser.add_argument("--rv_slice_competitor_ratio", type=float, default=0.80)
    parser.add_argument("--rv_slice_target_rv_hint", type=float, default=0.08)
    parser.add_argument("--topology_score_weight", type=float, default=1.25)
    parser.add_argument("--topology_guard", type=float, default=0.08)
    parser.add_argument("--prior_entropy_lambda", type=float, default=1.0)
    parser.add_argument("--prior_loc_lambda", type=float, default=0.15)
    parser.add_argument("--prior_bce_lambda", type=float, default=0.05)
    parser.add_argument("--prior_ts_lambda", type=float, default=0.10)
    parser.add_argument("--prior_source_consistency_lambda", type=float, default=0.03)
    parser.add_argument("--prior_weight_gamma", type=float, default=1.0)
    parser.add_argument("--prior_weight_min", type=float, default=0.20)
    parser.add_argument("--prior_ema_alpha", type=float, default=0.99)
    parser.add_argument("--prior_refine_steps", type=int, default=1)
    parser.add_argument(
        "--apl_online_weight",
        type=float,
        default=0.0,
        help="Blend APSA prior guidance into BN-only online adaptation (0 disables; range 0-1).",
    )
    parser.add_argument(
        "--prior_class_weights",
        default="",
        help="Optional comma-separated soft-prior CE weights. Use LV,MYO,RV or BG,LV,MYO,RV.",
    )
    parser.add_argument(
        "--prior_consistency_class_weights",
        default="",
        help="Optional class weights for teacher/source probability consistency. Use LV,MYO,RV or BG,LV,MYO,RV.",
    )
    parser.add_argument("--boundary_prior_lambda", type=float, default=0.0)
    parser.add_argument(
        "--boundary_prior_classes",
        default="2,3",
        help="Comma-separated cardiac class ids used by uncertainty-gated boundary APL.",
    )
    parser.add_argument("--boundary_prior_kernel", type=int, default=5)
    parser.add_argument("--boundary_uncertainty_low", type=float, default=0.02)
    parser.add_argument("--boundary_uncertainty_high", type=float, default=0.20)
    parser.add_argument("--inplace_prior_refine", action="store_true", help="Apply prior refinement on the online A3 model instead of a temporary candidate clone.")
    parser.add_argument("--mre_lambda_ap", type=float, default=1.0)
    parser.add_argument("--mre_area_min", type=float, default=0.0001)
    parser.add_argument("--mre_area_max", type=float, default=0.45)
    parser.add_argument("--mre_center_max", type=float, default=0.35)
    parser.add_argument("--mre_use_entropy", type=str2bool, default=True)
    parser.add_argument("--mre_use_prior_align", type=str2bool, default=True)
    parser.add_argument("--mre_use_area", type=str2bool, default=True)
    parser.add_argument("--mre_use_center", type=str2bool, default=True)
    parser.add_argument("--mre_use_prior_conf", type=str2bool, default=True)
    parser.add_argument("--mre_use_topology", type=str2bool, default=True)
    parser.add_argument("--rv_guard", type=str2bool, default=False)
    parser.add_argument("--rv_guard_min_area", type=float, default=0.001)
    parser.add_argument("--rv_guard_min_ratio", type=float, default=0.70)
    parser.add_argument("--rv_guard_conf_drop", type=float, default=0.08)
    parser.add_argument("--final_takeover_v2", action="store_true", help="Require the fused final candidate to show enough geometry/anatomy gain before overriding adapted.")
    parser.add_argument("--final_takeover_score_gain", type=float, default=0.002)
    parser.add_argument("--final_takeover_geometry_gain", type=float, default=0.015)
    parser.add_argument("--final_takeover_align_gain", type=float, default=0.0005)
    parser.add_argument("--final_takeover_entropy_increase", type=float, default=0.0005)
    parser.add_argument("--source_rescue_v2", action="store_true", help="Allow a guarded source rollback when adapted/final are close but anatomy reliability favors source.")
    parser.add_argument("--source_rescue_margin", type=float, default=0.03)
    parser.add_argument("--source_rescue_min_geometry", type=float, default=0.85)
    parser.add_argument("--source_rescue_rv_min_ratio", type=float, default=0.95)
    parser.add_argument("--source_rescue_rv_max_ratio", type=float, default=1.20)
    parser.add_argument("--source_rescue_align_margin", type=float, default=0.001)
    parser.add_argument("--source_rescue_entropy_margin", type=float, default=0.008)
    parser.add_argument(
        "--samplewise_mre_output",
        action="store_true",
        help="Use per-slice MRE/candidate selection for the final output while keeping batch-level rollback.",
    )
    parser.add_argument(
        "--guarded_mre_clean",
        action="store_true",
        help="Default to clean_final and use multi-cue MRE voting to guardingly fall back to clean_adapted.",
    )
    parser.add_argument("--guarded_mre_required_votes", type=int, default=3)
    parser.add_argument("--guarded_mre_entropy_margin", type=float, default=0.001)
    parser.add_argument("--guarded_mre_geometry_margin", type=float, default=0.02)
    parser.add_argument("--guarded_mre_align_margin", type=float, default=0.001)
    parser.add_argument("--guarded_mre_area_min_ratio", type=float, default=0.98)
    parser.add_argument("--guarded_mre_rv_min_ratio", type=float, default=0.95)
    parser.add_argument("--guarded_mre_rv_conf_margin", type=float, default=0.002)
    parser.add_argument(
        "--candidate_selector_v3",
        action="store_true",
        help="Use guarded whole-mask candidate selection instead of plain candidate-bank score max.",
    )
    parser.add_argument(
        "--candidate_selector_v3_names",
        default="clean_final,clean_adapted,adapted,refined,final,fusion_x0.50,fusion_x0.70,fusion_x0.85,fusion_x1.20",
        help="Comma-separated whole-mask candidates considered by candidate_selector_v3.",
    )
    parser.add_argument(
        "--candidate_selector_v3_name_bias",
        default="",
        help="Comma-separated selector-v3 score calibration, e.g. clean_final:0.02,adapted:0.01.",
    )
    parser.add_argument("--candidate_selector_v3_min_gain", type=float, default=0.0)
    parser.add_argument("--candidate_selector_v3_geometry_weight", type=float, default=0.0)
    parser.add_argument("--candidate_selector_v3_align_weight", type=float, default=0.0)
    parser.add_argument("--candidate_selector_v3_center_weight", type=float, default=0.0)
    parser.add_argument("--candidate_selector_v3_rv_weight", type=float, default=0.0)
    parser.add_argument("--candidate_selector_v3_clean_final_rv_min_ratio", type=float, default=0.85)
    parser.add_argument("--candidate_selector_v3_clean_final_area_min_ratio", type=float, default=0.80)
    parser.add_argument("--candidate_selector_v3_clean_final_penalty", type=float, default=0.02)
    parser.add_argument("--candidate_selector_v3_tta_geometry_guard", type=float, default=0.04)
    parser.add_argument("--candidate_selector_v3_tta_align_guard", type=float, default=0.004)
    parser.add_argument("--candidate_selector_v3_tta_penalty", type=float, default=1.0)
    parser.add_argument("--candidate_selector_v3_allow_source", action="store_true")
    parser.add_argument(
        "--classwise_candidate_select",
        action="store_true",
        help="Select LV/MYO/RV foreground channels independently from the APSA candidate bank using class-specific MRE cues.",
    )
    parser.add_argument(
        "--classwise_candidate_names",
        default="state,final,clean_final,clean_adapted,fusion_x0.50,fusion_x0.70,fusion_x0.85,fusion_x1.00,fusion_x1.20",
        help="Comma-separated candidates considered by classwise_candidate_select.",
    )
    parser.add_argument("--classwise_candidate_min_gain", type=float, default=0.0)
    parser.add_argument(
        "--classwise_candidate_name_bias",
        default="",
        help="Comma-separated classwise candidate score calibration, e.g. clean_final:0.02,adapted:-0.01.",
    )
    parser.add_argument(
        "--classwise_candidate_compose",
        default="logit",
        choices=["logit", "mask"],
        help="Compose per-class candidates by foreground logit replacement or hard candidate masks.",
    )
    parser.add_argument("--classwise_candidate_logit_boost", type=float, default=0.0)
    parser.add_argument(
        "--force_output_candidate",
        default="",
        help="Diagnostic only: force final output from a named candidate, e.g. adapted, final, clean_final.",
    )
    parser.add_argument(
        "--candidate_oracle_audit",
        action="store_true",
        help="Write per-candidate oracle Dice/ASSD diagnostics for every batch. Diagnostic only; uses target labels.",
    )
    parser.add_argument(
        "--candidate_oracle_assd",
        action="store_true",
        help="Also compute per-candidate ASSD in candidate_oracle.csv. Slow; Dice-only is the default.",
    )
    parser.add_argument("--oracle_audit", type=str2bool, default=False)
    parser.add_argument("--safe_branch_selector_conservative", action="store_true")
    parser.add_argument("--safe_branch_selector_adaptive", dest="safe_branch_selector_adaptive", action="store_true", default=True)
    parser.add_argument("--no_safe_branch_selector_adaptive", dest="safe_branch_selector_adaptive", action="store_false")
    parser.add_argument("--legacy_selector", action="store_true", help="Use the earlier cardiac wrapper selector instead of MRE.")
    parser.add_argument("--keep_prior_state", action="store_true", help="Keep prior-guided refinement weights in the online stream when final is selected.")
    parser.add_argument(
        "--no_source_output",
        action="store_true",
        help="Use source predictions for priors/scoring but never as the final cardiac output.",
    )
    parser.add_argument(
        "--reset_each_domain",
        action="store_true",
        help="Reload source checkpoint for every target domain. Default follows A3's continuous target-domain stream.",
    )
    parser.add_argument(
        "--no_state_rollback",
        action="store_true",
        help="Disable all restoration of the online A3 state, including risk-aware parameter reversion.",
    )
    parser.add_argument(
        "--risk_aware_reversion",
        action="store_true",
        help="Restore theta_pre when the risk detector flags the current online update as unsafe.",
    )
    parser.add_argument(
        "--save_pred_masks_dir",
        default="",
        help="Optional directory for saving selected prediction masks as NIfTI files. Default is disabled.",
    )
    parser.add_argument(
        "--save_pred_masks_cases",
        default="",
        help="Comma-separated image basenames to save when --save_pred_masks_dir is set.",
    )
    parser.add_argument(
        "--save_pred_masks_all",
        action="store_true",
        help="Save prediction masks for every evaluated case when --save_pred_masks_dir is set.",
    )
    parser.add_argument(
        "--save_stage_masks",
        action="store_true",
        help="Also save P_pre/P_adapt/P_ref/P_final masks for requested visualization cases.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_config(args.cfg, args.opts)
    set_seed(args.seed)
    if bool(args.rpr_paired_probe):
        configure_paired_audit_runtime()

    use_apssa = bool(args.use_apssa)
    if use_apssa and bool(args.inplace_prior_refine):
        raise ValueError("Adapter comparison requires clone-based APL; remove --inplace_prior_refine.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    continuous_domains = not bool(args.reset_each_domain)
    rollback_state = not bool(args.no_state_rollback)
    anatomical_prior_enabled = not bool(args.disable_anatomical_prior)
    prior_guided = not bool(args.no_prior_guidance) and anatomical_prior_enabled
    mre_use_prior_align = bool(args.mre_use_prior_align) and anatomical_prior_enabled
    mre_use_prior_conf = bool(args.mre_use_prior_conf) and anatomical_prior_enabled
    mre_use_center = bool(args.mre_use_center) and anatomical_prior_enabled
    mre_use_topology = bool(args.mre_use_topology) and anatomical_prior_enabled
    source_fusion_classes = parse_float_list(args.source_fusion_classes)
    rpr_probe_horizons = sorted(set(parse_int_list(args.rpr_probe_horizons)))
    candidate_bank_fusion_scales = parse_float_list(args.candidate_bank_fusion_scales) or []
    candidate_bank_tta_flips = parse_string_list(args.candidate_bank_tta_flips)
    candidate_bank_name_bias = parse_name_float_map(args.candidate_bank_name_bias)
    rv_composed_candidate_names = parse_string_list(args.rv_composed_candidate_names)
    rv_consensus_candidate_names = parse_string_list(args.rv_consensus_candidate_names)
    if bool(args.rv_composed_candidate_bank) and not bool(args.candidate_bank_v2):
        raise ValueError("--rv_composed_candidate_bank requires --candidate_bank_v2")
    if bool(args.rv_consensus_candidate_bank) and not bool(args.candidate_bank_v2):
        raise ValueError("--rv_consensus_candidate_bank requires --candidate_bank_v2")
    if int(args.rv_consensus_min_agreement) < 1:
        raise ValueError("--rv_consensus_min_agreement must be at least 1")
    if not 0.0 <= float(args.rv_consensus_confidence) <= 1.0:
        raise ValueError("--rv_consensus_confidence must be in [0, 1]")
    if not 0.0 <= float(args.rv_consensus_prior_threshold) <= 1.0:
        raise ValueError("--rv_consensus_prior_threshold must be in [0, 1]")
    if not 0.0 <= float(args.rv_consensus_protect_confidence) <= 1.0:
        raise ValueError("--rv_consensus_protect_confidence must be in [0, 1]")
    if int(args.rv_consensus_support_kernel) < 3:
        raise ValueError("--rv_consensus_support_kernel must be at least 3")
    if float(args.rv_consensus_max_extra_ratio) <= 0.0:
        raise ValueError("--rv_consensus_max_extra_ratio must be positive")
    if not 0.0 < float(args.rv_consensus_strength) <= 1.0:
        raise ValueError("--rv_consensus_strength must be in (0, 1]")
    if bool(args.rv_image_candidate) and not bool(args.candidate_bank_v2):
        raise ValueError("--rv_image_candidate requires --candidate_bank_v2")
    if not 0.0 < float(args.rv_image_probability_cap) <= 0.49:
        raise ValueError("--rv_image_probability_cap must be in (0, 0.49]")
    if float(args.rv_image_max_area_ratio) <= 0.0:
        raise ValueError("--rv_image_max_area_ratio must be positive")
    if bool(args.rv_atlas_candidate) and not bool(args.candidate_bank_v2):
        raise ValueError("--rv_atlas_candidate requires --candidate_bank_v2")
    if bool(args.rv_atlas_candidate) and not str(args.rv_atlas_path or "").strip():
        raise ValueError("--rv_atlas_candidate requires --rv_atlas_path")
    if not 0.0 < float(args.rv_atlas_probability_cap) <= 0.49:
        raise ValueError("--rv_atlas_probability_cap must be in (0, 0.49]")
    if not 0.0 <= float(args.rv_atlas_support_threshold) < 1.0:
        raise ValueError("--rv_atlas_support_threshold must be in [0, 1)")
    if float(args.rv_atlas_max_area_scale) <= 0.0:
        raise ValueError("--rv_atlas_max_area_scale must be positive")
    rv_atlas = load_source_rv_atlas(args.rv_atlas_path) if bool(args.rv_atlas_candidate) else None
    if bool(args.rv_slice_candidate) and not bool(args.candidate_bank_v2):
        raise ValueError("--rv_slice_candidate requires --candidate_bank_v2")
    if not 0.0 < float(args.rv_slice_probability_cap) <= 0.49:
        raise ValueError("--rv_slice_probability_cap must be in (0, 0.49]")
    if int(args.rv_slice_max_gap) < 1:
        raise ValueError("--rv_slice_max_gap must be at least 1")
    if float(args.rv_slice_max_area_scale) <= 0.0:
        raise ValueError("--rv_slice_max_area_scale must be positive")
    mre_calibrator = load_source_calibrated_mre(args.mre_calibrator)
    rpr_calibrator = load_source_calibrated_rpr(args.rpr_calibrator)
    if mre_calibrator is not None and not bool(args.candidate_bank_v2):
        raise ValueError("--mre_calibrator requires --candidate_bank_v2")
    if rpr_calibrator is not None and not bool(args.preupdate_rpr):
        raise ValueError("--rpr_calibrator requires --preupdate_rpr")
    if bool(args.preupdate_rpr) and str(args.adapter).strip().lower() != "tent":
        raise ValueError("--preupdate_rpr currently supports only the clean TentAdapter")
    if int(args.rpr_vote_threshold) < 1:
        raise ValueError("--rpr_vote_threshold must be at least 1")
    if int(args.rpr_checkpoint_interval) < 1:
        raise ValueError("--rpr_checkpoint_interval must be at least 1")
    if int(args.rpr_checkpoint_interval) > 1 and not bool(args.preupdate_rpr):
        raise ValueError("Checkpointed RPR+ requires --preupdate_rpr")
    for name in (
        "rpr_entropy_margin",
        "rpr_prior_align_margin",
        "rpr_center_margin",
        "rpr_topology_margin",
        "rpr_confidence_margin",
        "rpr_pre_adapt_drift_threshold",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    if not (0.0 < float(args.rpr_fg_min_ratio) <= 1.0):
        raise ValueError("--rpr_fg_min_ratio must be in (0, 1]")
    if float(args.rpr_fg_max_ratio) < 1.0:
        raise ValueError("--rpr_fg_max_ratio must be at least 1")
    if any(horizon < 1 for horizon in rpr_probe_horizons):
        raise ValueError("--rpr_probe_horizons must contain only positive integers")
    if bool(args.rpr_paired_probe):
        probe_errors = []
        if not rpr_probe_horizons:
            probe_errors.append("at least one positive probe horizon is required")
        if not bool(args.preupdate_rpr):
            probe_errors.append("--preupdate_rpr is required")
        if not bool(args.risk_aware_reversion):
            probe_errors.append("--risk_aware_reversion is required")
        if not rollback_state:
            probe_errors.append("state rollback must be enabled")
        if str(args.adapter).strip().lower() != "tent":
            probe_errors.append("the clean TentAdapter is required")
        if not bool(args.disable_adaptation_dropout):
            probe_errors.append("deterministic dropout-disabled adaptation is required")
        if args.rpr_probe_max_events is not None and int(args.rpr_probe_max_events) < 1:
            probe_errors.append("--rpr_probe_max_events must be positive")
        if probe_errors:
            raise ValueError("Paired RPR probe integrity failed: " + "; ".join(probe_errors))
    rv_gain_selector = load_source_nonlinear_rv_selector(args.rv_gain_selector)
    if rv_gain_selector is not None and not bool(args.candidate_bank_v2):
        raise ValueError("--rv_gain_selector requires --candidate_bank_v2")
    if rv_gain_selector is not None and not bool(args.rv_composed_candidate_bank):
        raise ValueError("--rv_gain_selector requires --rv_composed_candidate_bank")
    if rv_gain_selector is not None and mre_calibrator is not None:
        raise ValueError("Use either --rv_gain_selector or --mre_calibrator, not both")
    candidate_cleanup_sources = parse_string_list(args.candidate_cleanup_sources)
    cleanup_protected_sources = parse_string_list(args.cleanup_protected_sources)
    cleanup_protected_classes = parse_int_list(args.cleanup_protected_classes)
    candidate_selector_v3_names = parse_string_list(args.candidate_selector_v3_names)
    candidate_selector_v3_name_bias = parse_name_float_map(args.candidate_selector_v3_name_bias)
    classwise_candidate_names = parse_string_list(args.classwise_candidate_names)
    classwise_candidate_name_bias = parse_name_float_map(args.classwise_candidate_name_bias)
    prior_class_weights = parse_float_list(args.prior_class_weights)
    prior_consistency_class_weights = parse_float_list(args.prior_consistency_class_weights)
    temporal_consistency_class_weights = parse_float_list(args.temporal_consistency_class_weights)
    boundary_prior_classes = parse_int_list(args.boundary_prior_classes)
    cleanup_keep_largest_classes = parse_int_list(args.cleanup_keep_largest_classes)
    cleanup_fill_hole_classes = parse_int_list(args.cleanup_fill_hole_classes)
    prior_presence_classes = parse_int_list(args.prior_presence_classes)
    save_pred_masks_dir = Path(args.save_pred_masks_dir) if str(args.save_pred_masks_dir or "").strip() else None
    save_pred_masks_cases = parse_string_list(args.save_pred_masks_cases)

    effective_modules = {
        "generic_bn_adaptation": bool(str(args.adapter).strip().lower() == "tent"),
        "anatomical_prior": bool(use_apssa and anatomical_prior_enabled),
        "apl": bool(
            use_apssa
            and anatomical_prior_enabled
            and prior_guided
            and float(args.apl_online_weight) > 0.0
            and int(args.prior_refine_steps) > 0
        ),
        "candidate_expansion": bool(use_apssa and args.candidate_bank_v2),
        "mre": bool(use_apssa and not args.disable_mre_selection),
        "rpr": bool(
            use_apssa
            and args.risk_aware_reversion
            and rollback_state
            and str(args.adapter).strip().lower() != "source"
        ),
        "a3_prototype_memory": bool(str(args.adapter).strip().lower() == "a3-full"),
        "a3_anchor_selection": bool(str(args.adapter).strip().lower() == "a3-full"),
        "a3_feature_replacement": bool(str(args.adapter).strip().lower() == "a3-full"),
        "a3_ema_history": bool(str(args.adapter).strip().lower() in ("a3-full", "a3-lite")),
    }
    effective_online_losses = ["entropy_minimization"]
    if effective_modules["apl"]:
        effective_online_losses = [
            "apl_weighted_entropy",
            "apl_localization",
            "apl_soft_prior_ce",
            "apl_teacher_source_consistency",
        ]

    strict_ablation_variant = str(args.strict_ablation_variant or "")
    strict_ablation_checks: List[str] = []
    longstream_safety_variant = str(args.longstream_safety_variant or "")
    longstream_safety_checks: List[str] = []
    if strict_ablation_variant and longstream_safety_variant:
        raise ValueError(
            "--strict_ablation_variant and --longstream_safety_variant are mutually exclusive"
        )
    if strict_ablation_variant:
        expected_modules = {
            "base_bn_only": {
                "anatomical_prior": False,
                "apl": False,
                "candidate_expansion": False,
                "mre": False,
                "rpr": False,
            },
            "plus_apl": {
                "anatomical_prior": True,
                "apl": True,
                "candidate_expansion": False,
                "mre": False,
                "rpr": False,
            },
            "plus_apl_mre": {
                "anatomical_prior": True,
                "apl": True,
                "candidate_expansion": True,
                "mre": True,
                "rpr": False,
            },
        }[strict_ablation_variant]
        errors = []
        if str(args.adapter).strip().lower() != "tent":
            errors.append("strict ablations require the generic TentAdapter")
        if str(args.tent_train_scope).strip().lower() != "bn":
            errors.append("strict ablations require BN-affine train scope")
        if not bool(args.disable_adaptation_dropout):
            errors.append("strict ablations require deterministic dropout-disabled adaptation")
        for module_name, expected in expected_modules.items():
            actual = bool(effective_modules[module_name])
            if actual != bool(expected):
                errors.append(f"{module_name} expected {expected}, got {actual}")
        if not bool(args.no_state_rollback):
            errors.append("Base/APL/MRE rows must disable rollback; Full APSA reuses the frozen run")
        if bool(args.risk_aware_reversion):
            errors.append("RPR must be absent from Base/APL/MRE rows")
        forbidden_switches = {
            "preupdate_rpr": bool(args.preupdate_rpr),
            "safe_ema_candidate": bool(args.safe_ema_candidate),
            "temporal_consistency": bool(args.temporal_consistency),
            "rv_composed_candidate_bank": bool(args.rv_composed_candidate_bank),
            "rv_consensus_candidate_bank": bool(args.rv_consensus_candidate_bank),
            "soft_anatomy_candidate_bank": bool(args.soft_anatomy_candidate_bank),
            "rv_image_candidate": bool(args.rv_image_candidate),
            "rv_atlas_candidate": bool(args.rv_atlas_candidate),
            "rv_slice_candidate": bool(args.rv_slice_candidate),
            "anatomy_protected_cleanup": bool(args.anatomy_protected_cleanup),
            "source_rescue_v2": bool(args.source_rescue_v2),
            "samplewise_mre_output": bool(args.samplewise_mre_output),
            "guarded_mre_clean": bool(args.guarded_mre_clean),
            "candidate_selector_v3": bool(args.candidate_selector_v3),
            "classwise_candidate_select": bool(args.classwise_candidate_select),
            "mre_calibrator": mre_calibrator is not None,
            "rv_gain_selector": rv_gain_selector is not None,
            "boundary_prior": float(args.boundary_prior_lambda) > 0.0,
            "soft_myo_ring": float(args.soft_myo_ring_strength) > 0.0,
            "soft_rv_adjacency": float(args.soft_rv_adjacency_strength) > 0.0,
            "topology_prior": float(args.topology_prior_strength) > 0.0,
            "topology_rv_prior": float(args.topology_rv_strength) > 0.0,
        }
        enabled_forbidden = [name for name, enabled in forbidden_switches.items() if enabled]
        if enabled_forbidden:
            errors.append("forbidden residual switches: " + ",".join(enabled_forbidden))
        if strict_ablation_variant in ("base_bn_only", "plus_apl"):
            if bool(args.candidate_bank_v2) or bool(args.candidate_cleanup_v2):
                errors.append("candidate expansion/cleanup must be absent before the MRE row")
            if not bool(args.disable_mre_selection):
                errors.append("MRE selection must be explicitly disabled")
        if strict_ablation_variant == "plus_apl_mre":
            if not bool(args.candidate_bank_v2) or not bool(args.candidate_cleanup_v2):
                errors.append("the MRE row must use the frozen candidate bank and cleanup")
            if bool(args.disable_mre_selection):
                errors.append("the MRE row must enable reliability selection")
        if errors:
            raise ValueError(
                f"Strict ablation integrity failed for {strict_ablation_variant}: "
                + "; ".join(errors)
            )
        strict_ablation_checks = [
            "generic TentAdapter only",
            "BN-affine parameters only",
            "A3 prototype/anchor/feature-replacement/history disabled",
            "experimental EMA/RV/selector branches disabled",
            "module sequence matches strict incremental definition",
            "RPR disabled; frozen Full APSA is the only RPR-enabled row",
        ]

    if longstream_safety_variant:
        expected_rpr = longstream_safety_variant == "with_rpr"
        errors = []
        if str(args.adapter).strip().lower() != "tent":
            errors.append("long-stream safety comparison requires TentAdapter")
        if str(args.tent_train_scope).strip().lower() != "bn":
            errors.append("long-stream safety comparison requires BN-affine scope")
        if not bool(args.disable_adaptation_dropout):
            errors.append("long-stream safety comparison requires deterministic dropout-disabled adaptation")
        if int(args.stream_rounds) != 5:
            errors.append(f"stream_rounds expected 5, got {args.stream_rounds}")
        expected_common_modules = {
            "anatomical_prior": True,
            "apl": True,
            "candidate_expansion": True,
            "mre": True,
        }
        for module_name, expected in expected_common_modules.items():
            actual = bool(effective_modules[module_name])
            if actual != expected:
                errors.append(f"{module_name} expected {expected}, got {actual}")
        if bool(effective_modules["rpr"]) != expected_rpr:
            errors.append(
                f"rpr expected {expected_rpr}, got {effective_modules['rpr']}"
            )
        if expected_rpr:
            if bool(args.no_state_rollback) or not bool(args.risk_aware_reversion):
                errors.append("with_rpr requires rollback state and risk-aware reversion")
        else:
            if not bool(args.no_state_rollback) or bool(args.risk_aware_reversion):
                errors.append("without_rpr must disable all state rollback and RPR")
        if not bool(args.candidate_bank_v2) or not bool(args.candidate_cleanup_v2):
            errors.append("long-stream safety comparison requires the frozen MRE candidate bank")
        if bool(args.disable_mre_selection):
            errors.append("long-stream safety comparison requires MRE selection")
        if not bool(args.no_source_output):
            errors.append("long-stream safety comparison must keep source output disabled")
        if not bool(args.oracle_audit) or not bool(args.candidate_oracle_audit):
            errors.append("paired safety diagnostics require both oracle audit outputs")
        forbidden_switches = {
            "safe_ema_candidate": bool(args.safe_ema_candidate),
            "temporal_consistency": bool(args.temporal_consistency),
            "rv_composed_candidate_bank": bool(args.rv_composed_candidate_bank),
            "rv_consensus_candidate_bank": bool(args.rv_consensus_candidate_bank),
            "soft_anatomy_candidate_bank": bool(args.soft_anatomy_candidate_bank),
            "rv_image_candidate": bool(args.rv_image_candidate),
            "rv_atlas_candidate": bool(args.rv_atlas_candidate),
            "rv_slice_candidate": bool(args.rv_slice_candidate),
            "anatomy_protected_cleanup": bool(args.anatomy_protected_cleanup),
            "source_rescue_v2": bool(args.source_rescue_v2),
            "samplewise_mre_output": bool(args.samplewise_mre_output),
            "guarded_mre_clean": bool(args.guarded_mre_clean),
            "candidate_selector_v3": bool(args.candidate_selector_v3),
            "classwise_candidate_select": bool(args.classwise_candidate_select),
            "mre_calibrator": mre_calibrator is not None,
            "rv_gain_selector": rv_gain_selector is not None,
            "boundary_prior": float(args.boundary_prior_lambda) > 0.0,
            "soft_myo_ring": float(args.soft_myo_ring_strength) > 0.0,
            "soft_rv_adjacency": float(args.soft_rv_adjacency_strength) > 0.0,
            "topology_prior": float(args.topology_prior_strength) > 0.0,
            "topology_rv_prior": float(args.topology_rv_strength) > 0.0,
        }
        enabled_forbidden = [name for name, enabled in forbidden_switches.items() if enabled]
        if enabled_forbidden:
            errors.append("forbidden residual switches: " + ",".join(enabled_forbidden))
        if errors:
            raise ValueError(
                f"Long-stream safety integrity failed for {longstream_safety_variant}: "
                + "; ".join(errors)
            )
        longstream_safety_checks = [
            "B-C-D stream repeated exactly five rounds without reset",
            "generic BN-affine TentAdapter only",
            "APL and MRE match the frozen clean Full APSA method",
            "variants differ only in RPR/state rollback",
            "A3 and experimental candidate branches disabled",
            "target labels used only for post-hoc oracle audit",
        ]

    shared_a3_model = None
    shared_source_model = None
    if continuous_domains:
        shared_a3_model, shared_source_model = build_models(
            device,
            args.adapter,
            tent_learning_rate=args.tent_lr,
            tent_train_scope=args.tent_train_scope,
            disable_adaptation_dropout=bool(args.disable_adaptation_dropout),
            safe_ema_candidate=bool(args.safe_ema_candidate),
            temporal_consistency=bool(args.temporal_consistency),
            safe_ema_alpha=args.safe_ema_alpha,
        )
    trainable_parameter_names = list(getattr(shared_a3_model, "trainable_names", []))
    trainable_parameter_count = int(getattr(shared_a3_model, "trainable_parameter_count", 0))

    base_domains = args.domains if args.domains is not None else list(cfg.ADAPTATION.TARGET_DOMAIN)
    if args.source_calibration_corruption != "none" and any(
        str(domain) != str(cfg.SOURCE.SOURCE_DOMAIN) for domain in base_domains
    ):
        raise ValueError(
            "--source_calibration_corruption is restricted to the source domain and cannot be used on B/C/D"
        )
    stream_rounds = max(1, int(args.stream_rounds))
    stream_items = []
    for round_idx in range(stream_rounds):
        for domain in base_domains:
            output_label = str(domain) if stream_rounds == 1 else f"round{round_idx + 1}_{domain}"
            stream_items.append((str(domain), output_label))
    domains = [domain for domain, _ in stream_items]
    batch_size = int(args.batch_size or cfg.ADAPTATION.BATCH_SIZE)
    if args.save_dir is None:
        day = time.strftime("%Y%m%d")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = (
            Path(__file__).resolve().parent
            / f"analysis_mms_heart_apssa_adapters_{day}"
            / str(args.adapter)
            / f"apssa_{int(use_apssa)}"
            / f"run_{stamp}"
        )
    else:
        save_dir = Path(args.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty result directory: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "cfg": args.cfg,
        "adapter": str(args.adapter),
        "use_apssa": bool(use_apssa),
        "strict_ablation_variant": strict_ablation_variant,
        "strict_ablation_checks": strict_ablation_checks,
        "longstream_safety_variant": longstream_safety_variant,
        "longstream_safety_checks": longstream_safety_checks,
        "effective_modules": effective_modules,
        "effective_online_losses": effective_online_losses,
        "trainable_parameter_names": trainable_parameter_names,
        "trainable_parameter_count": trainable_parameter_count,
        "domains": domains,
        "base_domains": base_domains,
        "stream_rounds": stream_rounds,
        "stream_labels": [label for _, label in stream_items],
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "max_cases": args.max_cases,
        "source_calibration_corruption": str(args.source_calibration_corruption),
        "seed": args.seed,
        "deterministic_runtime": False,
        "paired_audit_runtime_controls": {
            "tf32": False,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "same_state_null_replay": bool(args.rpr_paired_probe),
        },
        "tent_lr": float(cfg.OPTIM.LR) if args.tent_lr is None else float(args.tent_lr),
        "tent_train_scope": str(args.tent_train_scope),
        "disable_adaptation_dropout": bool(args.disable_adaptation_dropout),
        "safe_ema_candidate": bool(args.safe_ema_candidate),
        "safe_ema_alpha": args.safe_ema_alpha,
        "safe_ema_apl_refine": bool(args.safe_ema_apl_refine),
        "safe_ema_candidate_min_gain": args.safe_ema_candidate_min_gain,
        "temporal_consistency": bool(args.temporal_consistency),
        "temporal_consistency_lambda": args.temporal_consistency_lambda,
        "temporal_consistency_class_weights": temporal_consistency_class_weights,
        "temporal_consistency_conf_threshold": args.temporal_consistency_conf_threshold,
        "temporal_consistency_rv_conf_threshold": args.temporal_consistency_rv_conf_threshold,
        "temporal_consistency_prior_threshold": args.temporal_consistency_prior_threshold,
        "temporal_consistency_temperature": args.temporal_consistency_temperature,
        "prior_refine_lr": float(cfg.OPTIM.LR) if args.prior_refine_lr is None else float(args.prior_refine_lr),
        "checkpoint": str(cfg.MODEL.CKPT_DIR),
        "method_base": str(cfg.MODEL.METHOD),
        "prediction_timing": "post_update",
        "apssa_teacher_target": "P_adapt_detached",
        "surface_assd_protocol": str(args.surface_assd_protocol),
        "surface_metric_impl": (
            "public_sictta_capped_2d_pixel_assd"
            if args.surface_assd_protocol == "sictta_capped"
            else "scipy_exact_euclidean_distance_transform_2d"
        ),
        "surface_metric_unit": "pixel",
        "surface_empty_policy": (
            "both_empty:ASSD=0,HD95=NaN;one_empty:ASSD=10,HD95=image_diagonal"
            if args.surface_assd_protocol == "sictta_capped"
            else "both_empty:ASSD=0,HD95=NaN;one_empty:ASSD=0,HD95=image_diagonal"
        ),
        "dataset": str(cfg.MODEL.DATASET),
        "source_domain": str(cfg.SOURCE.SOURCE_DOMAIN),
        "entropy_margin": args.entropy_margin,
        "drift_threshold": args.drift_threshold,
        "geometry_margin": args.geometry_margin,
        "preupdate_rpr": bool(args.preupdate_rpr),
        "rpr_reference_prediction": "inherited_online_pre_update" if args.preupdate_rpr else "frozen_source",
        "rpr_calibrator": "" if rpr_calibrator is None else str(rpr_calibrator["path"]),
        "rpr_calibrator_source": None if rpr_calibrator is None else rpr_calibrator.get("source"),
        "rpr_vote_threshold": args.rpr_vote_threshold,
        "rpr_entropy_margin": args.rpr_entropy_margin,
        "rpr_prior_align_margin": args.rpr_prior_align_margin,
        "rpr_center_margin": args.rpr_center_margin,
        "rpr_topology_margin": args.rpr_topology_margin,
        "rpr_confidence_margin": args.rpr_confidence_margin,
        "rpr_pre_adapt_drift_threshold": args.rpr_pre_adapt_drift_threshold,
        "rpr_fg_min_ratio": args.rpr_fg_min_ratio,
        "rpr_fg_max_ratio": args.rpr_fg_max_ratio,
        "rpr_paired_probe": bool(args.rpr_paired_probe),
        "rpr_probe_horizons": rpr_probe_horizons,
        "rpr_probe_max_events": args.rpr_probe_max_events,
        "source_fusion": args.source_fusion,
        "source_fusion_classes": source_fusion_classes,
        "adaptive_class_fusion": bool(args.adaptive_class_fusion),
        "adaptive_fusion_strength": args.adaptive_fusion_strength,
        "adaptive_fusion_max": args.adaptive_fusion_max,
        "candidate_bank_v2": bool(args.candidate_bank_v2),
        "candidate_bank_fusion_scales": candidate_bank_fusion_scales,
        "candidate_bank_include_refined": not bool(args.candidate_bank_no_refined),
        "candidate_bank_tta_flips": candidate_bank_tta_flips,
        "candidate_bank_min_gain": args.candidate_bank_min_gain,
        "candidate_bank_name_bias": candidate_bank_name_bias,
        "rv_composed_candidate_bank": bool(args.rv_composed_candidate_bank),
        "rv_composed_candidate_names": rv_composed_candidate_names,
        "rv_consensus_candidate_bank": bool(args.rv_consensus_candidate_bank),
        "rv_consensus_candidate_names": rv_consensus_candidate_names,
        "rv_consensus_min_agreement": args.rv_consensus_min_agreement,
        "rv_consensus_confidence": args.rv_consensus_confidence,
        "rv_consensus_base_margin_floor": args.rv_consensus_base_margin_floor,
        "rv_consensus_prior_threshold": args.rv_consensus_prior_threshold,
        "rv_consensus_protect_confidence": args.rv_consensus_protect_confidence,
        "rv_consensus_support_kernel": args.rv_consensus_support_kernel,
        "rv_consensus_max_extra_ratio": args.rv_consensus_max_extra_ratio,
        "rv_consensus_strength": args.rv_consensus_strength,
        "mre_calibrator": "" if mre_calibrator is None else str(mre_calibrator["path"]),
        "mre_calibrator_source": None if mre_calibrator is None else mre_calibrator.get("source"),
        "rv_gain_selector": "" if rv_gain_selector is None else str(rv_gain_selector["path"]),
        "rv_gain_selector_source": None if rv_gain_selector is None else rv_gain_selector.get("source"),
        "candidate_cleanup_v2": bool(args.candidate_cleanup_v2),
        "mre_selection_enabled": not bool(args.disable_mre_selection),
        "candidate_cleanup_sources": candidate_cleanup_sources,
        "cleanup_min_area_abs": args.cleanup_min_area_abs,
        "cleanup_min_area_ratio": args.cleanup_min_area_ratio,
        "cleanup_logit_bias": args.cleanup_logit_bias,
        "cleanup_keep_largest_classes": cleanup_keep_largest_classes,
        "cleanup_fill_hole_classes": cleanup_fill_hole_classes,
        "anatomy_protected_cleanup": bool(args.anatomy_protected_cleanup),
        "cleanup_protected_sources": cleanup_protected_sources,
        "cleanup_protected_classes": cleanup_protected_classes,
        "cleanup_protection_min_area_abs": args.cleanup_protection_min_area_abs,
        "cleanup_protection_candidate_confidence": args.cleanup_protection_candidate_confidence,
        "cleanup_protection_reference_overlap": args.cleanup_protection_reference_overlap,
        "cleanup_protection_consensus_overlap": args.cleanup_protection_consensus_overlap,
        "cleanup_protection_prior_mean": args.cleanup_protection_prior_mean,
        "cleanup_protection_presence_margin": args.cleanup_protection_presence_margin,
        "cleanup_protection_max_components": args.cleanup_protection_max_components,
        "volume_continuity_blend": float(args.volume_continuity_blend),
        "final_bias": args.final_bias,
        "final_min_gain": args.final_min_gain,
        "final_geometry_guard": args.final_geometry_guard,
        "source_fallback_margin": args.source_fallback_margin,
        "source_output_penalty": args.source_output_penalty,
        "anatomical_prior_enabled": anatomical_prior_enabled,
        "prior_guided": prior_guided,
        "apl_online_weight": float(args.apl_online_weight),
        "prior_source_blend": args.prior_source_blend,
        "prior_smooth_sigma": args.prior_smooth_sigma,
        "prior_generic_sigma": args.prior_generic_sigma,
        "prior_min": args.prior_min,
        "prior_presence_preserving": bool(args.prior_presence_preserving),
        "prior_presence_classes": prior_presence_classes,
        "prior_presence_min_pixels": args.prior_presence_min_pixels,
        "prior_presence_area_reference": args.prior_presence_area_reference,
        "prior_presence_confidence_low": args.prior_presence_confidence_low,
        "prior_presence_confidence_high": args.prior_presence_confidence_high,
        "prior_presence_floor": args.prior_presence_floor,
        "topology_prior_strength": args.topology_prior_strength,
        "topology_rv_strength": args.topology_rv_strength,
        "topology_ring_kernel": args.topology_ring_kernel,
        "soft_myo_ring_strength": args.soft_myo_ring_strength,
        "soft_rv_adjacency_strength": args.soft_rv_adjacency_strength,
        "soft_anatomy_ring_kernel": args.soft_anatomy_ring_kernel,
        "soft_anatomy_rv_kernel": args.soft_anatomy_rv_kernel,
        "soft_anatomy_gate_low": args.soft_anatomy_gate_low,
        "soft_anatomy_gate_high": args.soft_anatomy_gate_high,
        "soft_anatomy_max_delta": args.soft_anatomy_max_delta,
        "soft_anatomy_candidate_bank": bool(args.soft_anatomy_candidate_bank),
        "soft_anatomy_candidate_min_gain": args.soft_anatomy_candidate_min_gain,
        "rv_image_candidate": bool(args.rv_image_candidate),
        "rv_image_probability_cap": args.rv_image_probability_cap,
        "rv_image_adjacency_kernel": args.rv_image_adjacency_kernel,
        "rv_image_trigger_ratio": args.rv_image_trigger_ratio,
        "rv_image_max_area_ratio": args.rv_image_max_area_ratio,
        "rv_image_background_margin": args.rv_image_background_margin,
        "rv_image_min_lv_pixels": args.rv_image_min_lv_pixels,
        "rv_image_lv_confidence": args.rv_image_lv_confidence,
        "rv_image_myo_confidence": args.rv_image_myo_confidence,
        "rv_image_intensity_sigma_scale": args.rv_image_intensity_sigma_scale,
        "rv_atlas_candidate": bool(args.rv_atlas_candidate),
        "rv_atlas_path": "" if rv_atlas is None else str(rv_atlas["path"]),
        "rv_atlas_source_domain": None if rv_atlas is None else rv_atlas["metadata"].get("source_domain"),
        "rv_atlas_probability_cap": args.rv_atlas_probability_cap,
        "rv_atlas_support_threshold": args.rv_atlas_support_threshold,
        "rv_atlas_max_area_scale": args.rv_atlas_max_area_scale,
        "rv_atlas_background_margin": args.rv_atlas_background_margin,
        "rv_atlas_min_lv_pixels": args.rv_atlas_min_lv_pixels,
        "rv_atlas_lv_confidence": args.rv_atlas_lv_confidence,
        "rv_atlas_myo_confidence": args.rv_atlas_myo_confidence,
        "rv_atlas_competitor_margin": args.rv_atlas_competitor_margin,
        "rv_slice_candidate": bool(args.rv_slice_candidate),
        "rv_slice_probability_cap": args.rv_slice_probability_cap,
        "rv_slice_max_gap": args.rv_slice_max_gap,
        "rv_slice_require_two_sided": bool(args.rv_slice_require_two_sided),
        "rv_slice_seed_confidence": args.rv_slice_seed_confidence,
        "rv_slice_min_seed_pixels": args.rv_slice_min_seed_pixels,
        "rv_slice_min_lv_pixels": args.rv_slice_min_lv_pixels,
        "rv_slice_trigger_neighbor_ratio": args.rv_slice_trigger_neighbor_ratio,
        "rv_slice_max_area_scale": args.rv_slice_max_area_scale,
        "rv_slice_support_kernel": args.rv_slice_support_kernel,
        "rv_slice_direction_sigma_scale": args.rv_slice_direction_sigma_scale,
        "rv_slice_direction_consistency": args.rv_slice_direction_consistency,
        "rv_slice_background_margin": args.rv_slice_background_margin,
        "rv_slice_competitor_ratio": args.rv_slice_competitor_ratio,
        "rv_slice_target_rv_hint": args.rv_slice_target_rv_hint,
        "topology_score_weight": args.topology_score_weight,
        "topology_guard": args.topology_guard,
        "prior_entropy_lambda": args.prior_entropy_lambda,
        "prior_loc_lambda": args.prior_loc_lambda,
        "prior_bce_lambda": args.prior_bce_lambda,
        "prior_ts_lambda": args.prior_ts_lambda,
        "prior_source_consistency_lambda": args.prior_source_consistency_lambda,
        "prior_weight_gamma": args.prior_weight_gamma,
        "prior_weight_min": args.prior_weight_min,
        "prior_ema_alpha": args.prior_ema_alpha,
        "prior_refine_steps": args.prior_refine_steps,
        "prior_class_weights": prior_class_weights,
        "prior_consistency_class_weights": prior_consistency_class_weights,
        "boundary_prior_lambda": args.boundary_prior_lambda,
        "boundary_prior_classes": boundary_prior_classes,
        "boundary_prior_kernel": args.boundary_prior_kernel,
        "boundary_uncertainty_low": args.boundary_uncertainty_low,
        "boundary_uncertainty_high": args.boundary_uncertainty_high,
        "prior_refine_clone": not bool(args.inplace_prior_refine),
        "inplace_prior_refine": bool(args.inplace_prior_refine),
        "mre_lambda_ap": args.mre_lambda_ap,
        "mre_area_min": args.mre_area_min,
        "mre_area_max": args.mre_area_max,
        "mre_center_max": args.mre_center_max,
        "mre_use_entropy": bool(args.mre_use_entropy),
        "mre_use_prior_align": mre_use_prior_align,
        "mre_use_area": bool(args.mre_use_area),
        "mre_use_center": mre_use_center,
        "mre_use_prior_conf": mre_use_prior_conf,
        "mre_use_topology": mre_use_topology,
        "rv_guard": bool(args.rv_guard),
        "rv_guard_min_area": args.rv_guard_min_area,
        "rv_guard_min_ratio": args.rv_guard_min_ratio,
        "rv_guard_conf_drop": args.rv_guard_conf_drop,
        "final_takeover_v2": bool(args.final_takeover_v2),
        "final_takeover_score_gain": args.final_takeover_score_gain,
        "final_takeover_geometry_gain": args.final_takeover_geometry_gain,
        "final_takeover_align_gain": args.final_takeover_align_gain,
        "final_takeover_entropy_increase": args.final_takeover_entropy_increase,
        "source_rescue_v2": bool(args.source_rescue_v2),
        "source_rescue_margin": args.source_rescue_margin,
        "source_rescue_min_geometry": args.source_rescue_min_geometry,
        "source_rescue_rv_min_ratio": args.source_rescue_rv_min_ratio,
        "source_rescue_rv_max_ratio": args.source_rescue_rv_max_ratio,
        "source_rescue_align_margin": args.source_rescue_align_margin,
        "source_rescue_entropy_margin": args.source_rescue_entropy_margin,
        "samplewise_mre_output": bool(args.samplewise_mre_output),
        "guarded_mre_clean": bool(args.guarded_mre_clean),
        "guarded_mre_required_votes": args.guarded_mre_required_votes,
        "guarded_mre_entropy_margin": args.guarded_mre_entropy_margin,
        "guarded_mre_geometry_margin": args.guarded_mre_geometry_margin,
        "guarded_mre_align_margin": args.guarded_mre_align_margin,
        "guarded_mre_area_min_ratio": args.guarded_mre_area_min_ratio,
        "guarded_mre_rv_min_ratio": args.guarded_mre_rv_min_ratio,
        "guarded_mre_rv_conf_margin": args.guarded_mre_rv_conf_margin,
        "candidate_selector_v3": bool(args.candidate_selector_v3),
        "candidate_selector_v3_names": candidate_selector_v3_names,
        "candidate_selector_v3_name_bias": candidate_selector_v3_name_bias,
        "candidate_selector_v3_min_gain": args.candidate_selector_v3_min_gain,
        "candidate_selector_v3_geometry_weight": args.candidate_selector_v3_geometry_weight,
        "candidate_selector_v3_align_weight": args.candidate_selector_v3_align_weight,
        "candidate_selector_v3_center_weight": args.candidate_selector_v3_center_weight,
        "candidate_selector_v3_rv_weight": args.candidate_selector_v3_rv_weight,
        "candidate_selector_v3_clean_final_rv_min_ratio": args.candidate_selector_v3_clean_final_rv_min_ratio,
        "candidate_selector_v3_clean_final_area_min_ratio": args.candidate_selector_v3_clean_final_area_min_ratio,
        "candidate_selector_v3_clean_final_penalty": args.candidate_selector_v3_clean_final_penalty,
        "candidate_selector_v3_tta_geometry_guard": args.candidate_selector_v3_tta_geometry_guard,
        "candidate_selector_v3_tta_align_guard": args.candidate_selector_v3_tta_align_guard,
        "candidate_selector_v3_tta_penalty": args.candidate_selector_v3_tta_penalty,
        "candidate_selector_v3_allow_source": bool(args.candidate_selector_v3_allow_source),
        "classwise_candidate_select": bool(args.classwise_candidate_select),
        "classwise_candidate_names": classwise_candidate_names,
        "classwise_candidate_min_gain": args.classwise_candidate_min_gain,
        "classwise_candidate_name_bias": classwise_candidate_name_bias,
        "classwise_candidate_compose": args.classwise_candidate_compose,
        "classwise_candidate_logit_boost": args.classwise_candidate_logit_boost,
        "force_output_candidate": str(args.force_output_candidate or ""),
        "candidate_oracle_audit": bool(args.candidate_oracle_audit),
        "candidate_oracle_assd": bool(args.candidate_oracle_assd),
        "oracle_audit": bool(args.oracle_audit),
        "safe_branch_selector_conservative": bool(args.safe_branch_selector_conservative),
        "safe_branch_selector_adaptive": bool(args.safe_branch_selector_adaptive),
        "legacy_selector": bool(args.legacy_selector),
        "keep_prior_state": bool(args.keep_prior_state),
        "allow_source_output": not bool(args.no_source_output),
        "no_source_output": bool(args.no_source_output),
        "continuous_domains": bool(continuous_domains),
        "reset_each_domain": bool(args.reset_each_domain),
        "rollback_state": bool(rollback_state),
        "no_state_rollback": bool(args.no_state_rollback),
        "risk_aware_reversion": bool(args.risk_aware_reversion),
        "rpr_checkpoint_interval": int(args.rpr_checkpoint_interval),
        "rpr_checkpoint_mode": (
            "trusted_interval"
            if int(args.rpr_checkpoint_interval) > 1
            else "pre_update"
        ),
        "rar_applicable": bool(str(args.adapter).strip().lower() != "source"),
        "save_pred_masks_dir": "" if save_pred_masks_dir is None else str(save_pred_masks_dir),
        "save_pred_masks_cases": save_pred_masks_cases,
        "save_pred_masks_all": bool(args.save_pred_masks_all),
        "save_stage_masks": bool(args.save_stage_masks),
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    for domain, output_label in stream_items:
        if continuous_domains:
            a3_model, source_model = shared_a3_model, shared_source_model
        else:
            set_seed(args.seed)
            a3_model, source_model = build_models(
                device,
                args.adapter,
                tent_learning_rate=args.tent_lr,
                tent_train_scope=args.tent_train_scope,
                disable_adaptation_dropout=bool(args.disable_adaptation_dropout),
                safe_ema_candidate=bool(args.safe_ema_candidate),
                temporal_consistency=bool(args.temporal_consistency),
                safe_ema_alpha=args.safe_ema_alpha,
            )
        results.append(
            evaluate_domain(
                a3_model=a3_model,
                source_model=source_model,
                adapter_name=str(args.adapter),
                use_apssa=bool(use_apssa),
                domain=domain,
                output_label=output_label,
                save_dir=save_dir,
                batch_size=batch_size,
                num_workers=args.num_workers,
                max_cases=args.max_cases,
                calibration_corruption=str(args.source_calibration_corruption),
                surface_assd_protocol=str(args.surface_assd_protocol),
                entropy_margin=args.entropy_margin,
                drift_threshold=args.drift_threshold,
                geometry_margin=args.geometry_margin,
                source_fusion=args.source_fusion,
                source_fusion_classes=source_fusion_classes,
                adaptive_class_fusion=bool(args.adaptive_class_fusion),
                adaptive_fusion_strength=args.adaptive_fusion_strength,
                adaptive_fusion_max=args.adaptive_fusion_max,
                candidate_bank_v2=bool(args.candidate_bank_v2),
                candidate_bank_fusion_scales=candidate_bank_fusion_scales,
                candidate_bank_include_refined=not bool(args.candidate_bank_no_refined),
                candidate_bank_tta_flips=candidate_bank_tta_flips,
                candidate_bank_min_gain=args.candidate_bank_min_gain,
                candidate_bank_name_bias=candidate_bank_name_bias,
                rv_composed_candidate_bank=bool(args.rv_composed_candidate_bank),
                rv_composed_candidate_names=rv_composed_candidate_names,
                rv_consensus_candidate_bank=bool(args.rv_consensus_candidate_bank),
                rv_consensus_candidate_names=rv_consensus_candidate_names,
                rv_consensus_min_agreement=args.rv_consensus_min_agreement,
                rv_consensus_confidence=args.rv_consensus_confidence,
                rv_consensus_base_margin_floor=args.rv_consensus_base_margin_floor,
                rv_consensus_prior_threshold=args.rv_consensus_prior_threshold,
                rv_consensus_protect_confidence=args.rv_consensus_protect_confidence,
                rv_consensus_support_kernel=args.rv_consensus_support_kernel,
                rv_consensus_max_extra_ratio=args.rv_consensus_max_extra_ratio,
                rv_consensus_strength=args.rv_consensus_strength,
                mre_calibrator=mre_calibrator,
                rv_gain_selector=rv_gain_selector,
                candidate_cleanup_v2=bool(args.candidate_cleanup_v2),
                candidate_cleanup_sources=candidate_cleanup_sources,
                cleanup_min_area_abs=args.cleanup_min_area_abs,
                cleanup_min_area_ratio=args.cleanup_min_area_ratio,
                cleanup_logit_bias=args.cleanup_logit_bias,
                cleanup_keep_largest_classes=cleanup_keep_largest_classes,
                cleanup_fill_hole_classes=cleanup_fill_hole_classes,
                anatomy_protected_cleanup=bool(args.anatomy_protected_cleanup),
                cleanup_protected_sources=cleanup_protected_sources,
                cleanup_protected_classes=cleanup_protected_classes,
                cleanup_protection_min_area_abs=args.cleanup_protection_min_area_abs,
                cleanup_protection_candidate_confidence=args.cleanup_protection_candidate_confidence,
                cleanup_protection_reference_overlap=args.cleanup_protection_reference_overlap,
                cleanup_protection_consensus_overlap=args.cleanup_protection_consensus_overlap,
                cleanup_protection_prior_mean=args.cleanup_protection_prior_mean,
                cleanup_protection_presence_margin=args.cleanup_protection_presence_margin,
                cleanup_protection_max_components=args.cleanup_protection_max_components,
                volume_continuity_blend=args.volume_continuity_blend,
                final_bias=args.final_bias,
                final_min_gain=args.final_min_gain,
                final_geometry_guard=args.final_geometry_guard,
                source_fallback_margin=args.source_fallback_margin,
                source_output_penalty=args.source_output_penalty,
                rollback_state=rollback_state,
                mre_selection_enabled=not bool(args.disable_mre_selection),
                risk_aware_reversion=bool(args.risk_aware_reversion),
                preupdate_rpr=bool(args.preupdate_rpr),
                rpr_calibrator=rpr_calibrator,
                rpr_vote_threshold=args.rpr_vote_threshold,
                rpr_checkpoint_interval=args.rpr_checkpoint_interval,
                rpr_entropy_margin=args.rpr_entropy_margin,
                rpr_prior_align_margin=args.rpr_prior_align_margin,
                rpr_center_margin=args.rpr_center_margin,
                rpr_topology_margin=args.rpr_topology_margin,
                rpr_confidence_margin=args.rpr_confidence_margin,
                rpr_pre_adapt_drift_threshold=args.rpr_pre_adapt_drift_threshold,
                rpr_fg_min_ratio=args.rpr_fg_min_ratio,
                rpr_fg_max_ratio=args.rpr_fg_max_ratio,
                rpr_paired_probe=bool(args.rpr_paired_probe),
                rpr_probe_horizons=rpr_probe_horizons,
                rpr_probe_max_events=args.rpr_probe_max_events,
                anatomical_prior_enabled=anatomical_prior_enabled,
                prior_guided=prior_guided,
                apl_online_weight=args.apl_online_weight,
                prior_source_blend=args.prior_source_blend,
                prior_smooth_sigma=args.prior_smooth_sigma,
                prior_generic_sigma=args.prior_generic_sigma,
                prior_min=args.prior_min,
                prior_presence_preserving=bool(args.prior_presence_preserving),
                prior_presence_classes=prior_presence_classes,
                prior_presence_min_pixels=args.prior_presence_min_pixels,
                prior_presence_area_reference=args.prior_presence_area_reference,
                prior_presence_confidence_low=args.prior_presence_confidence_low,
                prior_presence_confidence_high=args.prior_presence_confidence_high,
                prior_presence_floor=args.prior_presence_floor,
                topology_prior_strength=args.topology_prior_strength,
                topology_rv_strength=args.topology_rv_strength,
                topology_ring_kernel=args.topology_ring_kernel,
                soft_myo_ring_strength=args.soft_myo_ring_strength,
                soft_rv_adjacency_strength=args.soft_rv_adjacency_strength,
                soft_anatomy_ring_kernel=args.soft_anatomy_ring_kernel,
                soft_anatomy_rv_kernel=args.soft_anatomy_rv_kernel,
                soft_anatomy_gate_low=args.soft_anatomy_gate_low,
                soft_anatomy_gate_high=args.soft_anatomy_gate_high,
                soft_anatomy_max_delta=args.soft_anatomy_max_delta,
                soft_anatomy_candidate_bank=bool(args.soft_anatomy_candidate_bank),
                soft_anatomy_candidate_min_gain=args.soft_anatomy_candidate_min_gain,
                rv_image_candidate=bool(args.rv_image_candidate),
                rv_image_probability_cap=args.rv_image_probability_cap,
                rv_image_adjacency_kernel=args.rv_image_adjacency_kernel,
                rv_image_trigger_ratio=args.rv_image_trigger_ratio,
                rv_image_max_area_ratio=args.rv_image_max_area_ratio,
                rv_image_background_margin=args.rv_image_background_margin,
                rv_image_min_lv_pixels=args.rv_image_min_lv_pixels,
                rv_image_lv_confidence=args.rv_image_lv_confidence,
                rv_image_myo_confidence=args.rv_image_myo_confidence,
                rv_image_intensity_sigma_scale=args.rv_image_intensity_sigma_scale,
                rv_atlas_candidate=bool(args.rv_atlas_candidate),
                rv_atlas=rv_atlas,
                rv_atlas_probability_cap=args.rv_atlas_probability_cap,
                rv_atlas_support_threshold=args.rv_atlas_support_threshold,
                rv_atlas_max_area_scale=args.rv_atlas_max_area_scale,
                rv_atlas_background_margin=args.rv_atlas_background_margin,
                rv_atlas_min_lv_pixels=args.rv_atlas_min_lv_pixels,
                rv_atlas_lv_confidence=args.rv_atlas_lv_confidence,
                rv_atlas_myo_confidence=args.rv_atlas_myo_confidence,
                rv_atlas_competitor_margin=args.rv_atlas_competitor_margin,
                rv_slice_candidate=bool(args.rv_slice_candidate),
                rv_slice_probability_cap=args.rv_slice_probability_cap,
                rv_slice_max_gap=args.rv_slice_max_gap,
                rv_slice_require_two_sided=bool(args.rv_slice_require_two_sided),
                rv_slice_seed_confidence=args.rv_slice_seed_confidence,
                rv_slice_min_seed_pixels=args.rv_slice_min_seed_pixels,
                rv_slice_min_lv_pixels=args.rv_slice_min_lv_pixels,
                rv_slice_trigger_neighbor_ratio=args.rv_slice_trigger_neighbor_ratio,
                rv_slice_max_area_scale=args.rv_slice_max_area_scale,
                rv_slice_support_kernel=args.rv_slice_support_kernel,
                rv_slice_direction_sigma_scale=args.rv_slice_direction_sigma_scale,
                rv_slice_direction_consistency=args.rv_slice_direction_consistency,
                rv_slice_background_margin=args.rv_slice_background_margin,
                rv_slice_competitor_ratio=args.rv_slice_competitor_ratio,
                rv_slice_target_rv_hint=args.rv_slice_target_rv_hint,
                safe_ema_candidate=bool(args.safe_ema_candidate),
                safe_ema_apl_refine=bool(args.safe_ema_apl_refine),
                safe_ema_candidate_min_gain=args.safe_ema_candidate_min_gain,
                temporal_consistency=bool(args.temporal_consistency),
                temporal_consistency_lambda=args.temporal_consistency_lambda,
                temporal_consistency_class_weights=temporal_consistency_class_weights,
                temporal_consistency_conf_threshold=args.temporal_consistency_conf_threshold,
                temporal_consistency_rv_conf_threshold=args.temporal_consistency_rv_conf_threshold,
                temporal_consistency_prior_threshold=args.temporal_consistency_prior_threshold,
                temporal_consistency_temperature=args.temporal_consistency_temperature,
                topology_score_weight=args.topology_score_weight,
                topology_guard=args.topology_guard,
                prior_entropy_lambda=args.prior_entropy_lambda,
                prior_loc_lambda=args.prior_loc_lambda,
                prior_bce_lambda=args.prior_bce_lambda,
                prior_ts_lambda=args.prior_ts_lambda,
                prior_source_consistency_lambda=args.prior_source_consistency_lambda,
                prior_weight_gamma=args.prior_weight_gamma,
                prior_weight_min=args.prior_weight_min,
                prior_ema_alpha=args.prior_ema_alpha,
                prior_refine_steps=args.prior_refine_steps,
                prior_refine_lr=float(cfg.OPTIM.LR) if args.prior_refine_lr is None else float(args.prior_refine_lr),
                prior_class_weights=prior_class_weights,
                prior_consistency_class_weights=prior_consistency_class_weights,
                boundary_prior_lambda=args.boundary_prior_lambda,
                boundary_prior_classes=boundary_prior_classes,
                boundary_prior_kernel=args.boundary_prior_kernel,
                boundary_uncertainty_low=args.boundary_uncertainty_low,
                boundary_uncertainty_high=args.boundary_uncertainty_high,
                mre_lambda_ap=args.mre_lambda_ap,
                mre_area_min=args.mre_area_min,
                mre_area_max=args.mre_area_max,
                mre_center_max=args.mre_center_max,
                mre_use_entropy=bool(args.mre_use_entropy),
                mre_use_prior_align=mre_use_prior_align,
                mre_use_area=bool(args.mre_use_area),
                mre_use_center=mre_use_center,
                mre_use_prior_conf=mre_use_prior_conf,
                mre_use_topology=mre_use_topology,
                rv_guard=bool(args.rv_guard),
                rv_guard_min_area=args.rv_guard_min_area,
                rv_guard_min_ratio=args.rv_guard_min_ratio,
                rv_guard_conf_drop=args.rv_guard_conf_drop,
                final_takeover_v2=bool(args.final_takeover_v2),
                final_takeover_score_gain=args.final_takeover_score_gain,
                final_takeover_geometry_gain=args.final_takeover_geometry_gain,
                final_takeover_align_gain=args.final_takeover_align_gain,
                final_takeover_entropy_increase=args.final_takeover_entropy_increase,
                source_rescue_v2=bool(args.source_rescue_v2),
                source_rescue_margin=args.source_rescue_margin,
                source_rescue_min_geometry=args.source_rescue_min_geometry,
                source_rescue_rv_min_ratio=args.source_rescue_rv_min_ratio,
                source_rescue_rv_max_ratio=args.source_rescue_rv_max_ratio,
                source_rescue_align_margin=args.source_rescue_align_margin,
                source_rescue_entropy_margin=args.source_rescue_entropy_margin,
                samplewise_mre_output=bool(args.samplewise_mre_output),
                guarded_mre_clean=bool(args.guarded_mre_clean),
                guarded_mre_required_votes=args.guarded_mre_required_votes,
                guarded_mre_entropy_margin=args.guarded_mre_entropy_margin,
                guarded_mre_geometry_margin=args.guarded_mre_geometry_margin,
                guarded_mre_align_margin=args.guarded_mre_align_margin,
                guarded_mre_area_min_ratio=args.guarded_mre_area_min_ratio,
                guarded_mre_rv_min_ratio=args.guarded_mre_rv_min_ratio,
                guarded_mre_rv_conf_margin=args.guarded_mre_rv_conf_margin,
                candidate_selector_v3=bool(args.candidate_selector_v3),
                candidate_selector_v3_names=candidate_selector_v3_names,
                candidate_selector_v3_name_bias=candidate_selector_v3_name_bias,
                candidate_selector_v3_min_gain=args.candidate_selector_v3_min_gain,
                candidate_selector_v3_geometry_weight=args.candidate_selector_v3_geometry_weight,
                candidate_selector_v3_align_weight=args.candidate_selector_v3_align_weight,
                candidate_selector_v3_center_weight=args.candidate_selector_v3_center_weight,
                candidate_selector_v3_rv_weight=args.candidate_selector_v3_rv_weight,
                candidate_selector_v3_clean_final_rv_min_ratio=args.candidate_selector_v3_clean_final_rv_min_ratio,
                candidate_selector_v3_clean_final_area_min_ratio=args.candidate_selector_v3_clean_final_area_min_ratio,
                candidate_selector_v3_clean_final_penalty=args.candidate_selector_v3_clean_final_penalty,
                candidate_selector_v3_tta_geometry_guard=args.candidate_selector_v3_tta_geometry_guard,
                candidate_selector_v3_tta_align_guard=args.candidate_selector_v3_tta_align_guard,
                candidate_selector_v3_tta_penalty=args.candidate_selector_v3_tta_penalty,
                candidate_selector_v3_allow_source=bool(args.candidate_selector_v3_allow_source),
                classwise_candidate_select=bool(args.classwise_candidate_select),
                classwise_candidate_names=classwise_candidate_names,
                classwise_candidate_min_gain=args.classwise_candidate_min_gain,
                classwise_candidate_name_bias=classwise_candidate_name_bias,
                classwise_candidate_compose=args.classwise_candidate_compose,
                classwise_candidate_logit_boost=args.classwise_candidate_logit_boost,
                force_output_candidate=str(args.force_output_candidate or ""),
                candidate_oracle_audit=bool(args.candidate_oracle_audit),
                candidate_oracle_assd=bool(args.candidate_oracle_assd),
                oracle_audit=bool(args.oracle_audit),
                safe_branch_selector_conservative=bool(args.safe_branch_selector_conservative),
                safe_branch_selector_adaptive=bool(args.safe_branch_selector_adaptive),
                use_legacy_selector=bool(args.legacy_selector),
                keep_prior_state=bool(args.keep_prior_state),
                allow_source_output=not bool(args.no_source_output),
                prior_refine_clone=not bool(args.inplace_prior_refine),
                save_pred_masks_dir=save_pred_masks_dir,
                save_pred_masks_cases=save_pred_masks_cases,
                save_pred_masks_all=bool(args.save_pred_masks_all),
                save_stage_masks=bool(args.save_stage_masks),
            )
        )
    write_summary(results, save_dir)
    print(f"[APSA-A3] wrote results to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
