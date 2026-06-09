"""Loss functions for cervical cancer classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_counts(labels, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[int(label)] += 1.0
    return counts


def effective_number_weights(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    counts = counts.float().clamp_min(1.0)
    beta = float(beta)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    weights = (1.0 - beta) / effective_num.clamp_min(1e-8)
    weights = weights / weights.sum() * len(counts)
    return weights


def _soft_targets(targets: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    hard = F.one_hot(targets.long(), num_classes=num_classes).float()
    if smoothing <= 0:
        return hard
    return hard * (1.0 - smoothing) + smoothing / num_classes


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("alpha", alpha.clone().float() if alpha is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        if targets.dtype in (torch.long, torch.int32, torch.int64):
            target_probs = _soft_targets(targets, num_classes, self.label_smoothing)
            hard_targets = targets.long()
        else:
            target_probs = targets.float()
            hard_targets = target_probs.argmax(dim=-1)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ce = -(target_probs * log_probs).sum(dim=-1)
        pt = (target_probs * probs).sum(dim=-1).clamp_min(1e-8)
        loss = (1.0 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if target_probs.dim() == 2:
                sample_alpha = (target_probs * alpha.unsqueeze(0)).sum(dim=-1)
            else:
                sample_alpha = alpha[hard_targets]
            loss = loss * sample_alpha

        return loss.mean()


class ClassBalancedFocalLoss(FocalLoss):
    def __init__(self, counts: torch.Tensor, beta: float = 0.9999, gamma: float = 2.0, label_smoothing: float = 0.0):
        alpha = effective_number_weights(counts, beta=beta)
        super().__init__(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
        self.counts = counts.float()
        self.beta = beta


def build_criterion(
    counts: torch.Tensor,
    loss_type: str = "class_balanced_focal",
    gamma: float = 2.0,
    beta: float = 0.9999,
    label_smoothing: float = 0.0,
) -> nn.Module:
    loss_type = (loss_type or "focal").lower()
    if loss_type in {"class_balanced_focal", "cb_focal", "balanced_focal"}:
        return ClassBalancedFocalLoss(counts=counts, beta=beta, gamma=gamma, label_smoothing=label_smoothing)
    if loss_type in {"focal", "focal_loss"}:
        alpha = effective_number_weights(counts, beta=beta)
        return FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
    if loss_type in {"ce", "cross_entropy", "crossentropy"}:
        return nn.CrossEntropyLoss(weight=effective_number_weights(counts, beta=beta))
    raise ValueError(f"Unsupported loss_type: {loss_type}")