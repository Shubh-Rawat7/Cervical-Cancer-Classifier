"""
train.py — Production training script for Cervical Cancer Classifier.

Writes to output-dir:
  • best_model.pth   — best checkpoint by val balanced accuracy
  • history.json     — per-epoch metrics (consumed by Cell 7)
  • metrics.json     — final confusion matrix + classification report (Cell 7)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize

from models.cnn_model import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# MixUp
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.3, device="cuda"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Normal", "CIN1", "CIN2", "CIN3", "Cancer"]
NUM_CLASSES  = len(CLASS_NAMES)


def get_transforms(phase: str, img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def build_weighted_sampler(dataset):
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )


def get_class_weights(dataset, device):
    targets = np.array(dataset.targets)
    counts  = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32).to(device)


def load_datasets(data_dir: str, img_size: int = 224, val_frac: float = 0.20):
    """
    Supports three layouts:
      A) data_dir/train/ + data_dir/val/  → use pre-split as-is
      B) data_dir/train/ only             → stratified split
      C) data_dir/ has class folders      → stratified split
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        print("Layout A: pre-split (train/ + val/)")
        train_ds = datasets.ImageFolder(train_dir, transform=get_transforms("train", img_size))
        val_ds   = datasets.ImageFolder(val_dir,   transform=get_transforms("val",   img_size))
        train_ds.targets = list(train_ds.targets)
        val_ds.targets   = list(val_ds.targets)
    else:
        source_dir = train_dir if os.path.isdir(train_dir) else data_dir
        print(f"Layout B/C: stratified {val_frac:.0%} split from '{source_dir}'")

        full_ds = datasets.ImageFolder(source_dir, transform=get_transforms("train", img_size))
        targets = np.array(full_ds.targets)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
        train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

        train_ds = Subset(full_ds, train_idx)
        train_ds.targets = targets[train_idx].tolist()

        val_base = datasets.ImageFolder(source_dir, transform=get_transforms("val", img_size))
        val_ds   = Subset(val_base, val_idx)
        val_ds.targets = targets[val_idx].tolist()

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    t = np.array(train_ds.targets); v = np.array(val_ds.targets)
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:<8}: {(t==i).sum():>4} train  {(v==i).sum():>3} val")

    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────────────────────
# Training / eval loops
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device,
                use_mixup=True, mixup_alpha=0.3):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            mixed, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha, device=device)
            with autocast():
                logits = model(mixed)
                loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct    += preds.eq(labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # AUC-ROC (macro OvR) — only if all classes present
    try:
        y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
        auc   = roc_auc_score(y_bin, np.array(all_probs), multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return (total_loss / total, correct / total, bal_acc,
            macro_f1, auc, all_preds, all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Phase runner
# ──────────────────────────────────────────────────────────────────────────────

def run_phase(model, train_loader, val_loader, optimizer, scheduler,
              criterion, val_criterion, scaler, device, n_epochs,
              best_bal_acc, patience, best_ckpt_path, history,
              epoch_offset, mixup_alpha=0.3, mixup_scale=1.0,
              skip_mixup_epochs=0):
    patience_counter = 0

    for ep in range(1, n_epochs + 1):
        global_ep = epoch_offset + ep
        t0 = time.time()

        use_mixup = ep > skip_mixup_epochs
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha * mixup_scale,
        )
        val_loss, val_acc, bal_acc, val_f1, val_auc, preds, labels = eval_epoch(
            model, val_loader, val_criterion, device,
        )
        scheduler.step()

        # train F1 approximation (use train acc as proxy, real F1 too slow per epoch)
        train_f1_approx = tr_acc   # replace with real F1 if speed allows

        history.append({
            "epoch":          global_ep,
            "train_loss":     round(tr_loss, 6),
            "val_loss":       round(val_loss, 6),
            "train_accuracy": round(tr_acc,   6),
            "val_accuracy":   round(val_acc,   6),
            "train_f1":       round(train_f1_approx, 6),
            "val_f1":         round(val_f1,    6),
            "val_auc_roc":    round(val_auc, 6) if not np.isnan(val_auc) else None,
            "val_bal_acc":    round(bal_acc,  6),
        })

        elapsed = time.time() - t0
        print(f"  Ep {ep:03d}/{n_epochs} (global {global_ep}) | "
              f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.3f} | "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.3f} "
              f"BalAcc={bal_acc:.3f} F1={val_f1:.3f} | "
              f"{elapsed:.1f}s")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    ✓ Best saved (bal_acc={best_bal_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} stale epochs.")
                break

    return best_bal_acc, epoch_offset + n_epochs


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",       default="../data")
    p.add_argument("--output-dir",     default="./checkpoints")
    p.add_argument("--epochs",         type=int,   default=90)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--img-size",       type=int,   default=224)
    p.add_argument("--lr-head",        type=float, default=3e-4)
    p.add_argument("--lr-backbone",    type=float, default=3e-5)
    p.add_argument("--phase1-epochs",  type=int,   default=20)
    p.add_argument("--phase2-epochs",  type=int,   default=20)
    p.add_argument("--num-workers",    type=int,   default=2)
    p.add_argument("--focal-gamma",    type=float, default=2.0)
    p.add_argument("--mixup-alpha",    type=float, default=0.3)
    p.add_argument("--patience",       type=int,   default=15)
    p.add_argument("--val-frac",       type=float, default=0.20)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = load_datasets(args.data_dir, args.img_size, args.val_frac)
    sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model + loss ──────────────────────────────────────────────────────────
    model          = build_model(num_classes=NUM_CLASSES).to(device)
    class_weights  = get_class_weights(train_ds, device)
    criterion      = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    val_criterion  = nn.CrossEntropyLoss(weight=class_weights)
    scaler         = GradScaler()

    best_bal_acc   = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pth")
    history        = []          # ← written to history.json after each phase
    epoch_offset   = 0

    def _save_history():
        Path(args.output_dir, "history.json").write_text(json.dumps(history, indent=2))

    # ── Phase 1: head only ────────────────────────────────────────────────────
    p1 = args.phase1_epochs
    print(f"\n{'='*60}")
    print(f"PHASE 1 — Head only ({p1} epochs, lr_head={args.lr_head:.2e})")
    print(f"{'='*60}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(p1, 1), eta_min=1e-6,
    )
    best_bal_acc, epoch_offset = run_phase(
        model, train_loader, val_loader, optimizer, scheduler,
        criterion, val_criterion, scaler, device,
        n_epochs=p1, best_bal_acc=best_bal_acc, patience=args.patience,
        best_ckpt_path=best_ckpt_path, history=history,
        epoch_offset=epoch_offset, mixup_alpha=args.mixup_alpha,
        skip_mixup_epochs=3,
    )
    _save_history()

    # ── Phase 2: unfreeze last 3 blocks ──────────────────────────────────────
    p2 = args.phase2_epochs
    print(f"\n{'='*60}")
    print(f"PHASE 2 — Unfreeze last 3 blocks ({p2} epochs, backbone_lr={args.lr_backbone:.2e})")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(best_ckpt_path))
    model.unfreeze_backbone(unfreeze_last_n_blocks=3)
    optimizer = optim.AdamW([
        {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()),
         "lr": args.lr_backbone},
        {"params": model.se.parameters(),   "lr": args.lr_head},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(p2, 1), eta_min=1e-7,
    )
    best_bal_acc, epoch_offset = run_phase(
        model, train_loader, val_loader, optimizer, scheduler,
        criterion, val_criterion, scaler, device,
        n_epochs=p2, best_bal_acc=best_bal_acc, patience=args.patience,
        best_ckpt_path=best_ckpt_path, history=history,
        epoch_offset=epoch_offset, mixup_alpha=args.mixup_alpha,
    )
    _save_history()

    # ── Phase 3: full fine-tune ───────────────────────────────────────────────
    remaining = args.epochs - p1 - p2
    if remaining > 0:
        print(f"\n{'='*60}")
        print(f"PHASE 3 — Full fine-tune ({remaining} epochs, lr={args.lr_backbone/3:.2e})")
        print(f"{'='*60}")

        model.load_state_dict(torch.load(best_ckpt_path))
        model.unfreeze_all()
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr_backbone / 3, weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(remaining, 1), eta_min=1e-8,
        )
        best_bal_acc, epoch_offset = run_phase(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, val_criterion, scaler, device,
            n_epochs=remaining, best_bal_acc=best_bal_acc, patience=args.patience,
            best_ckpt_path=best_ckpt_path, history=history,
            epoch_offset=epoch_offset, mixup_alpha=args.mixup_alpha,
            mixup_scale=0.5,
        )
        _save_history()

    # ── Final evaluation + write metrics.json ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION  (best val balanced_accuracy = {best_bal_acc:.4f})")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(best_ckpt_path))
    _, val_acc, bal_acc, val_f1, val_auc, preds, labels = eval_epoch(
        model, val_loader, val_criterion, device,
    )

    print(f"Val Accuracy:          {val_acc:.4f}")
    print(f"Val Balanced Accuracy: {bal_acc:.4f}")
    cr_text = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)
    print(cr_text)

    # Build metrics.json exactly as Cell 7 expects
    cr_dict = classification_report(
        labels, preds, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist()
    macro_p  = precision_score(labels, preds, average="macro", zero_division=0)
    macro_r  = recall_score(labels, preds, average="macro",    zero_division=0)

    metrics = {
        "final_metrics": {
            "accuracy":               round(val_acc,  4),
            "balanced_accuracy":      round(bal_acc,  4),
            "precision":              round(macro_p,  4),
            "recall":                 round(macro_r,  4),
            "f1":                     round(val_f1,   4),
            "auc_roc":                round(val_auc, 4) if not np.isnan(val_auc) else None,
            "loss":                   round(history[-1]["val_loss"], 4) if history else None,
            "confusion_matrix":       cm,
            "classification_report":  cr_dict,
        }
    }
    Path(args.output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nhistory.json  → {args.output_dir}/history.json  ({len(history)} epochs)")
    print(f"metrics.json  → {args.output_dir}/metrics.json")
    print(f"Best checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()