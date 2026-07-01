"""Core training engine for the cervical cancer classifier."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, NUM_CLASSES
from dataset import build_dataloaders
from evaluation.reporting import generate_evaluation_artifacts
from explainability.gradcam import save_gradcam_visualization
from explainability.shap_tools import save_handcrafted_shap_summary
from losses import build_criterion
from models.hybrid_model import build_model


@dataclass(slots=True)
class TrainArtifacts:
    """Convenience container for the outputs of a training run."""

    history: list[dict[str, Any]]
    metrics: dict[str, Any]
    best_path: Path
    last_path: Path
    swa_path: Path | None


def set_seed(seed: int = 42) -> None:
    """Set deterministic RNG seeds where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train the hybrid cervical cancer classifier")
    parser.add_argument("--data-dir", default="../Herlev Dataset")
    parser.add_argument("--output-dir", default="./Checkpoints")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", default="mambavision_small")
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-backbone", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-epochs", type=int, default=24)
    parser.add_argument("--stage2-epochs", type=int, default=26)
    parser.add_argument("--stage3-epochs", type=int, default=40)
    parser.add_argument("--cb-beta", type=float, default=0.9999)
    parser.add_argument("--cb-gamma", type=float, default=2.0)
    parser.add_argument("--scheduler-t0", type=int, default=10)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--mixup-alpha", type=float, default=0.30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")
    parser.add_argument("--swa-epochs", type=int, default=20)
    parser.add_argument("--swa-lr", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _batch_to_device(batch, device: torch.device):
    """Normalize batch structures from the dataset loader."""
    if isinstance(batch, dict):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        path = batch.get("path")
        return image, label, path
    if isinstance(batch, (list, tuple)):
        image = batch[0].to(device)
        label = batch[1].to(device)
        path = batch[2] if len(batch) > 2 else None
        return image, label, path
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _soft_mix_targets(y_a: torch.Tensor, y_b: torch.Tensor, lam: float, num_classes: int) -> torch.Tensor:
    """Create soft targets for mixup-compatible focal loss."""
    one_hot_a = torch.nn.functional.one_hot(y_a.long(), num_classes=num_classes).float()
    one_hot_b = torch.nn.functional.one_hot(y_b.long(), num_classes=num_classes).float()
    return lam * one_hot_a + (1.0 - lam) * one_hot_b


def _mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return images, labels
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    mixed_targets = _soft_mix_targets(labels, labels[index], lam, num_classes=NUM_CLASSES)
    return mixed_images, mixed_targets


def _collect_param_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list[dict[str, Any]]:
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone") or name.startswith("backbone_projector"):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return groups


def save_checkpoint(model: nn.Module, path: Path, config: dict[str, Any]) -> None:
    """Persist a checkpoint with metadata for later reconstruction."""
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "class_names": CLASS_NAMES,
    }
    torch.save(payload, path)


def load_checkpoint_state_dict(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    """Load a model state dict from a checkpoint file."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "ema_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {path}")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    gradient_clip: float,
    use_amp: bool,
    mixup_alpha: float,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images, labels, _ = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if mixup_alpha > 0:
            images, soft_targets = _mixup_batch(images, labels, mixup_alpha)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, soft_targets)
        else:
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_loss += float(loss.item()) * images.size(0)
        total_samples += int(images.size(0))

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Evaluate a model and return full validation metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    labels: list[int] = []
    preds: list[int] = []
    probs: list[list[float]] = []
    paths: list[str] = []

    for batch in loader:
        images, targets, batch_paths = _batch_to_device(batch, device)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        probability = torch.softmax(logits, dim=1)
        prediction = probability.argmax(dim=1)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((prediction == targets).sum().item())
        total_samples += int(images.size(0))

        labels.extend(targets.detach().cpu().tolist())
        preds.extend(prediction.detach().cpu().tolist())
        probs.extend(probability.detach().cpu().tolist())
        if batch_paths is not None:
            paths.extend([str(path) for path in batch_paths])

    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    probs_arr = np.asarray(probs)
    loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    balanced_acc = balanced_accuracy_score(labels_arr, preds_arr)
    macro_precision = precision_score(labels_arr, preds_arr, average="macro", zero_division=0)
    macro_recall = recall_score(labels_arr, preds_arr, average="macro", zero_division=0)
    macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    try:
        y_bin = label_binarize(labels_arr, classes=list(range(NUM_CLASSES)))
        roc_auc = roc_auc_score(y_bin, probs_arr, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = None

    report = classification_report(labels_arr, preds_arr, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    confusion = confusion_matrix(labels_arr, preds_arr, labels=list(range(NUM_CLASSES)))

    return {
        "loss": loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "roc_auc": roc_auc,
        "labels": labels,
        "preds": preds,
        "probs": probs_arr,
        "paths": paths,
        "confusion_matrix": confusion,
        "classification_report": report,
        "per_class_metrics": {name: report.get(name, {}) for name in CLASS_NAMES},
    }


def _stage_name(index: int) -> str:
    return {1: "stage1", 2: "stage2", 3: "stage3"}.get(index, f"stage{index}")


def _build_optimizers(
    model: nn.Module,
    stage: int,
    args: argparse.Namespace,
) -> tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Build stage-specific optimizer and scheduler objects."""
    if stage == 1:
        params = _collect_param_groups(model, backbone_lr=0.0, head_lr=args.lr_head)
    elif stage == 2:
        params = _collect_param_groups(model, backbone_lr=args.lr_backbone, head_lr=args.lr_head)
    else:
        params = _collect_param_groups(model, backbone_lr=args.lr_backbone, head_lr=args.lr_head)
        if not params:
            params = [{"params": model.parameters(), "lr": args.lr_backbone}]

    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(int(args.scheduler_t0), 1),
        T_mult=1,
        eta_min=args.eta_min,
    )
    return optimizer, scheduler


def _run_stage(
    model: nn.Module,
    stage_index: int,
    epochs: int,
    train_loader,
    val_loader,
    criterion: nn.Module,
    val_criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    global_epoch: int,
    best_bal_acc: float,
    best_path: Path,
    scaler: GradScaler,
    swa_model: AveragedModel | None,
    swa_start_epoch: int,
) -> tuple[int, float, bool]:
    """Run a training stage and return updated epoch, best metric, and stop flag."""
    optimizer, scheduler = _build_optimizers(model, stage_index, args)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr) if swa_model is not None else None
    stale_epochs = 0
    stage_name = _stage_name(stage_index)
    print(f"\n{'=' * 72}")
    print(f"{stage_name.upper()} — {epochs} epochs")
    print(f"{'=' * 72}")

    if stage_index == 1:
        if hasattr(model, "freeze_classifier_only"):
            model.freeze_classifier_only()
        else:
            model.freeze_backbone()
    elif stage_index == 2:
        if hasattr(model, "unfreeze_last_blocks"):
            model.unfreeze_last_blocks(2)
    else:
        if hasattr(model, "unfreeze_all"):
            model.unfreeze_all()

    for local_epoch in range(1, epochs + 1):
        global_epoch += 1
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            gradient_clip=args.gradient_clip,
            use_amp=args.use_amp,
            mixup_alpha=args.mixup_alpha if global_epoch > 1 else 0.0,
        )
        val_metrics = evaluate(model, val_loader, val_criterion, device, use_amp=args.use_amp)

        if global_epoch >= swa_start_epoch and swa_model is not None and swa_scheduler is not None:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(global_epoch - 1 + local_epoch / max(1, len(train_loader)))

        history.append(
            {
                "epoch": global_epoch,
                "stage": stage_name,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_metrics["loss"], 6),
                "train_accuracy": round(train_acc, 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_bal_acc": round(val_metrics["balanced_accuracy"], 6),
                "val_precision": round(val_metrics["precision"], 6),
                "val_recall": round(val_metrics["recall"], 6),
                "val_f1": round(val_metrics["f1"], 6),
                "val_auc_roc": None if val_metrics["roc_auc"] is None else round(float(val_metrics["roc_auc"]), 6),
            }
        )

        print(
            f"  Ep {global_epoch:03d} | TrLoss={train_loss:.4f} TrAcc={train_acc:.3f} | "
            f"ValLoss={val_metrics['loss']:.4f} ValAcc={val_metrics['accuracy']:.3f} "
            f"BalAcc={val_metrics['balanced_accuracy']:.3f} F1={val_metrics['f1']:.3f}"
        )

        if val_metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = float(val_metrics["balanced_accuracy"])
            stale_epochs = 0
            save_checkpoint(
                model,
                best_path,
                {
                    "backbone": getattr(model, "backbone_name", args.backbone),
                    "num_classes": NUM_CLASSES,
                    "image_size": args.img_size,
                    "dropout": args.dropout,
                    "stage": stage_name,
                    "epoch": global_epoch,
                    "cb_beta": args.cb_beta,
                    "cb_gamma": args.cb_gamma,
                    "stage1_epochs": args.stage1_epochs,
                    "stage2_epochs": args.stage2_epochs,
                    "stage3_epochs": args.stage3_epochs,
                    "class_names": CLASS_NAMES,
                },
            )
            print(f"    ✓ Best model saved at bal_acc={best_bal_acc:.4f}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"    Early stopping triggered after {args.patience} stale epochs.")
                return global_epoch, best_bal_acc, True

    return global_epoch, best_bal_acc, False


def _save_history(history: Sequence[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(list(history), indent=2))


def _explain_sample(model: nn.Module, sample_path: str, output_dir: Path, device: torch.device) -> None:
    """Generate a Grad-CAM overlay and a handcrafted SHAP summary for one sample."""
    try:
        from dataset import build_eval_transform
        from PIL import Image

        transform = build_eval_transform(getattr(model, "image_size", DEFAULT_IMAGE_SIZE))
        gradcam_path = output_dir / "gradcam" / f"{Path(sample_path).stem}_gradcam.png"
        save_gradcam_visualization(model, sample_path, transform, gradcam_path, device)

        image = np.asarray(Image.open(sample_path).convert("RGB"))
        shap_path = output_dir / "shap" / f"{Path(sample_path).stem}_handcrafted_shap.png"
        save_handcrafted_shap_summary(model, image, shap_path)
    except Exception as exc:
        print(f"Explainability export skipped: {exc}")


def main() -> TrainArtifacts:
    """Entry point used by the Kaggle notebook and CLI wrapper."""
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, class_names, train_counts, train_samples, val_samples = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.0 if args.val_split <= 0 else args.val_split,
        under_sample=False,
    )

    model = build_model(num_classes=NUM_CLASSES, backbone_name=args.backbone, dropout=args.dropout, pretrained=True).to(device)
    counts_tensor = torch.tensor([train_counts[name] for name in class_names], dtype=torch.float32, device=device)
    criterion = build_criterion(counts_tensor, loss_type="cb_focal", gamma=args.cb_gamma, beta=args.cb_beta)
    val_criterion = nn.CrossEntropyLoss(weight=(counts_tensor.sum() / counts_tensor.clamp_min(1.0)).to(device))
    scaler = GradScaler(enabled=args.use_amp)

    history: list[dict[str, Any]] = []
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    swa_path = output_dir / "swa_model.pt"
    best_bal_acc = 0.0
    epoch = 0

    swa_start_epoch = max(1, args.epochs - args.swa_epochs + 1)
    swa_model = AveragedModel(model) if args.swa_epochs > 0 else None
    stage_plan = [
        (1, min(args.stage1_epochs, args.epochs)),
        (2, min(args.stage2_epochs, max(args.epochs - args.stage1_epochs, 0))),
    ]
    remaining = max(args.epochs - sum(epochs for _, epochs in stage_plan), 0)
    stage_plan.append((3, remaining if remaining > 0 else max(args.stage3_epochs, 0)))

    stop_training = False
    for stage_index, stage_epochs in stage_plan:
        if stage_epochs <= 0:
            continue
        epoch, best_bal_acc, stop_training = _run_stage(
            model=model,
            stage_index=stage_index,
            epochs=stage_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            val_criterion=val_criterion,
            device=device,
            args=args,
            history=history,
            global_epoch=epoch,
            best_bal_acc=best_bal_acc,
            best_path=best_path,
            scaler=scaler,
            swa_model=swa_model,
            swa_start_epoch=swa_start_epoch,
        )
        save_checkpoint(
            model,
            last_path,
            {
                "backbone": getattr(model, "backbone_name", args.backbone),
                "num_classes": NUM_CLASSES,
                "image_size": args.img_size,
                "dropout": args.dropout,
                "stage": _stage_name(stage_index),
                "epoch": epoch,
                "cb_beta": args.cb_beta,
                "cb_gamma": args.cb_gamma,
                "stage1_epochs": args.stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "stage3_epochs": args.stage3_epochs,
                "class_names": CLASS_NAMES,
            },
        )
        _save_history(history, output_dir)
        if stop_training:
            break

    if swa_model is not None and swa_start_epoch <= epoch:
        update_bn(train_loader, swa_model, device=device)
        save_checkpoint(
            swa_model,
            swa_path,
            {
                "backbone": getattr(model, "backbone_name", args.backbone),
                "num_classes": NUM_CLASSES,
                "image_size": args.img_size,
                "dropout": args.dropout,
                "stage": "swa",
                "epoch": epoch,
                "cb_beta": args.cb_beta,
                "cb_gamma": args.cb_gamma,
                "class_names": CLASS_NAMES,
            },
        )

    best_state = load_checkpoint_state_dict(best_path)
    model.load_state_dict(best_state, strict=False)
    val_metrics = evaluate(model, val_loader, val_criterion, device, use_amp=args.use_amp)
    report_metrics = generate_evaluation_artifacts(
        history=history,
        y_true=val_metrics["labels"],
        y_pred=val_metrics["preds"],
        y_prob=val_metrics["probs"],
        class_names=class_names,
        output_dir=output_dir,
        sample_paths=val_metrics["paths"],
    )

    metrics_payload = {
        "final_metrics": {
            "accuracy": round(float(report_metrics["accuracy"]), 4),
            "balanced_accuracy": round(float(report_metrics["balanced_accuracy"]), 4),
            "precision": round(float(report_metrics["precision"]), 4),
            "recall": round(float(report_metrics["recall"]), 4),
            "f1": round(float(report_metrics["f1"]), 4),
            "auc_roc": None if report_metrics["roc_auc"] is None else round(float(report_metrics["roc_auc"]), 4),
            "loss": round(float(val_metrics["loss"]), 4),
            "confusion_matrix": report_metrics["confusion_matrix"],
            "classification_report": report_metrics["classification_report"],
            "per_class_metrics": report_metrics["per_class_metrics"],
        },
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    if val_metrics["paths"]:
        _explain_sample(model, val_metrics["paths"][0], output_dir, device)

    print(f"history.json  -> {output_dir / 'history.json'}")
    print(f"metrics.json  -> {output_dir / 'metrics.json'}")
    print(f"best_model.pt -> {best_path}")
    if swa_path.exists():
        print(f"swa_model.pt  -> {swa_path}")

    return TrainArtifacts(
        history=history,
        metrics=metrics_payload,
        best_path=best_path,
        last_path=last_path,
        swa_path=swa_path if swa_path.exists() else None,
    )


if __name__ == "__main__":
    main()