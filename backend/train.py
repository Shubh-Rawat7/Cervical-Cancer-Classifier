"""Herlev cervical cell Mamba training pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch.amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from config import CLASS_NAMES, DEFAULTS, DEFAULT_IMAGE_SIZE, CHECKPOINT_DIR, MODEL_PATH, NUM_CLASSES
from dataset import HerlevDataset, build_eval_transform, build_train_transform
from losses import build_criterion
from models.model import HerlevMambaClassifier
from preprocessing import ImageRecord, class_distribution, discover_samples, filter_quality_and_duplicates, split_samples, undersample_samples


@dataclass
class TrainState:
    epoch: int
    best_val_loss: float
    best_val_f1: float
    best_val_acc: float
    history: List[Dict[str, float]]


def seed_everything(seed: int = DEFAULTS.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Create a learning rate scheduler with linear warmup followed by cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def _build_loader(samples: Sequence[ImageRecord], batch_size: int, image_size: int, shuffle: bool, num_workers: int, transform=None) -> DataLoader:
    dataset = HerlevDataset(samples, transform=transform)
    workers = max(0, min(int(num_workers), torch.get_num_threads() or 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def _labels_from_samples(samples: Sequence[ImageRecord]) -> np.ndarray:
    return np.array([CLASS_NAMES.index(sample.label) for sample in samples], dtype=np.int64)


def _counts_tensor(samples: Sequence[ImageRecord], device: torch.device) -> torch.Tensor:
    labels = _labels_from_samples(samples)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    return torch.tensor(counts, dtype=torch.float32, device=device)


def _step_batch(model, batch, device):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return images, labels


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for step, batch in enumerate(loader, start=1):
        images, labels = _step_batch(model, batch, device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels) / max(1, accumulation_steps)

        scaler.scale(loss).backward()

        if step % accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * images.size(0) * max(1, accumulation_steps)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    accuracy = accuracy_score(all_targets, all_preds) if all_targets else 0.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) if all_targets else 0.0
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": float(accuracy),
        "f1": float(macro_f1),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device, use_amp: bool = True) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    probabilities: List[np.ndarray] = []
    predictions: List[int] = []
    targets: List[int] = []

    for batch in loader:
        images, labels = _step_batch(model, batch, device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

        total_loss += float(loss.item()) * images.size(0)
        probabilities.append(probs.detach().cpu().numpy())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    y_true = np.array(targets, dtype=np.int64)
    y_pred = np.array(predictions, dtype=np.int64)
    y_prob = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, NUM_CLASSES), dtype=np.float32)

    metrics = {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
    }
    try:
        y_true_one_hot = np.eye(NUM_CLASSES)[y_true]
        metrics["auc_roc"] = float(roc_auc_score(y_true_one_hot, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc_roc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist() if len(y_true) else []
    metrics["classification_report"] = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, output_dict=True) if len(y_true) else {}
    return metrics


def build_model(args) -> HerlevMambaClassifier:
    return HerlevMambaClassifier(
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        image_size=args.image_size,
        embed_dim=args.embed_dim,
        mamba_layers=args.mamba_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        activation=args.activation,
        pretrained=not args.no_pretrained,
    )


def save_checkpoint(path: Path, model: torch.nn.Module, args, metrics: Dict[str, float], class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "backbone": args.backbone,
                "num_classes": NUM_CLASSES,
                "image_size": args.image_size,
                "embed_dim": args.embed_dim,
                "mamba_layers": args.mamba_layers,
                "attn_heads": args.attn_heads,
                "dropout": args.dropout,
                "activation": args.activation,
                "class_names": list(class_names),
            },
            "class_names": list(class_names),
            "metrics": metrics,
        },
        path,
    )


def fit_single_split(args) -> Dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    class_names, samples = discover_samples(args.data_dir, class_names=CLASS_NAMES)
    samples, duplicates, quality_issues = filter_quality_and_duplicates(samples)
    train_samples, val_samples, _ = split_samples(samples, val_size=args.val_split, seed=args.seed)
    train_samples = undersample_samples(train_samples, strategy=args.undersample, seed=args.seed)

    train_loader = _build_loader(train_samples, args.batch_size, args.image_size, shuffle=True, num_workers=args.workers, transform=build_train_transform(args.image_size))
    val_loader = _build_loader(val_samples, args.batch_size, args.image_size, shuffle=False, num_workers=args.workers, transform=build_eval_transform(args.image_size))

    model = build_model(args).to(device)
    criterion = build_criterion(
        _counts_tensor(train_samples, device),
        loss_type=args.loss_type,
        gamma=args.gamma,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _get_warmup_scheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    state = TrainState(epoch=0, best_val_loss=float("inf"), best_val_f1=0.0, best_val_acc=0.0, history=[])
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, accumulation_steps=args.accumulation_steps, grad_clip=args.grad_clip, use_amp=args.amp and device.type == "cuda")
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=args.amp and device.type == "cuda")
        scheduler.step()

        epoch_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items() if k != "classification_report" and k != "confusion_matrix"})
        epoch_metrics["epoch"] = epoch
        state.history.append(epoch_metrics)
        state.epoch = epoch

        improved = val_metrics["loss"] < state.best_val_loss
        if improved:
            state.best_val_loss = float(val_metrics["loss"])
            state.best_val_f1 = float(val_metrics["f1"])
            state.best_val_acc = float(val_metrics["accuracy"])
            save_checkpoint(args.output_dir / "best_model.pt", model, args, val_metrics, class_names)
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(args.output_dir / "last_model.pt", model, args, val_metrics, class_names)
        if patience_counter >= args.patience:
            break

    best_model_path = args.output_dir / "best_model.pt"
    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"], strict=False)
    final_metrics = evaluate(model, val_loader, criterion, device, use_amp=args.amp and device.type == "cuda")

    report = {
        "mode": "single_split",
        "device": str(device),
        "class_names": list(class_names),
        "train_distribution": class_distribution(train_samples, class_names=class_names),
        "val_distribution": class_distribution(val_samples, class_names=class_names),
        "quality_issues": len(quality_issues),
        "duplicate_groups": len(duplicates),
        "history": state.history,
        "best_val_loss": state.best_val_loss,
        "best_val_f1": state.best_val_f1,
        "best_val_acc": state.best_val_acc,
        "final_metrics": final_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "history.json").write_text(json.dumps(state.history, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def fit_kfold(args) -> Dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    class_names, samples = discover_samples(args.data_dir, class_names=CLASS_NAMES)
    samples, duplicates, quality_issues = filter_quality_and_duplicates(samples)
    labels = _labels_from_samples(samples)
    kfold = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    fold_reports = []
    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels), start=1):
        fold_train = [samples[i] for i in train_idx]
        fold_val = [samples[i] for i in val_idx]
        fold_train = undersample_samples(fold_train, strategy=args.undersample, seed=args.seed + fold_index)
        train_loader = _build_loader(fold_train, args.batch_size, args.image_size, shuffle=True, num_workers=args.workers, transform=build_train_transform(args.image_size))
        val_loader = _build_loader(fold_val, args.batch_size, args.image_size, shuffle=False, num_workers=args.workers, transform=build_eval_transform(args.image_size))

        model = build_model(args).to(device)
        criterion = build_criterion(
            _counts_tensor(fold_train, device),
            loss_type=args.loss_type,
            gamma=args.gamma,
            beta=args.beta,
            label_smoothing=args.label_smoothing,
        )
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = _get_warmup_scheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)
        scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

        best_val_f1 = 0.0
        best_val_loss = float("inf")
        patience_counter = 0
        best_path = args.output_dir / f"fold_{fold_index}_best.pt"

        for _epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, accumulation_steps=args.accumulation_steps, grad_clip=args.grad_clip, use_amp=args.amp and device.type == "cuda")
            val_metrics = evaluate(model, val_loader, criterion, device, use_amp=args.amp and device.type == "cuda")
            scheduler.step()
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                best_val_f1 = float(val_metrics["f1"])
                save_checkpoint(best_path, model, args, val_metrics, class_names)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                break

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        final_metrics = evaluate(model, val_loader, criterion, device, use_amp=args.amp and device.type == "cuda")
        final_metrics["best_val_f1"] = best_val_f1
        final_metrics["best_val_loss"] = best_val_loss
        final_metrics["fold"] = fold_index
        fold_reports.append(final_metrics)

    summary = {
        "mode": "kfold",
        "device": str(device),
        "class_names": list(class_names),
        "folds": fold_reports,
        "mean_accuracy": float(np.nanmean([fold["accuracy"] for fold in fold_reports])) if fold_reports else 0.0,
        "mean_f1": float(np.nanmean([fold["f1"] for fold in fold_reports])) if fold_reports else 0.0,
        "mean_auc_roc": float(np.nanmean([fold.get("auc_roc", float("nan")) for fold in fold_reports])) if fold_reports else 0.0,
        "quality_issues": len(quality_issues),
        "duplicate_groups": len(duplicates),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "kfold_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_activation_ablation(args) -> Dict[str, object]:
    results = []
    for activation in ("silu", "gelu", "mish"):
        ablation_args = argparse.Namespace(**vars(args))
        ablation_args.activation = activation
        ablation_dir = args.output_dir / activation
        ablation_args.output_dir = ablation_dir
        report = fit_single_split(ablation_args)
        results.append({"activation": activation, **report["final_metrics"]})
    summary = {"mode": "activation_ablation", "results": results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Herlev cervical cell Mamba training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--image-size", type=int, default=DEFAULTS.image_size)
    parser.add_argument("--val-split", type=float, default=DEFAULTS.val_split)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--use-kfold", action="store_true")
    parser.add_argument("--activation-ablation", action="store_true")
    parser.add_argument("--undersample", choices=["random", "nearmiss"], default="random")
    parser.add_argument("--backbone", type=str, default=DEFAULTS.backbone)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--mamba-layers", type=int, default=2)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", choices=["silu", "gelu", "mish"], default=DEFAULTS.activation)
    parser.add_argument("--loss-type", type=str, default=DEFAULTS.loss_type)
    parser.add_argument("--gamma", type=float, default=DEFAULTS.gamma)
    parser.add_argument("--beta", type=float, default=DEFAULTS.beta)
    parser.add_argument("--label-smoothing", type=float, default=DEFAULTS.label_smoothing)
    parser.add_argument("--lr", type=float, default=DEFAULTS.lr)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULTS.warmup_epochs)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=DEFAULTS.grad_clip)
    parser.add_argument("--accumulation-steps", type=int, default=DEFAULTS.accumulation_steps)
    parser.add_argument("--workers", type=int, default=DEFAULTS.num_workers)
    parser.add_argument("--patience", type=int, default=DEFAULTS.patience)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--amp", action="store_true", default=DEFAULTS.use_amp)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, object]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.activation_ablation:
        return run_activation_ablation(args)
    if args.use_kfold:
        return fit_kfold(args)
    return fit_single_split(args)


if __name__ == "__main__":
    main()
