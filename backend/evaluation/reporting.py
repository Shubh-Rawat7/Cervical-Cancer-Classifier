"""Evaluation and reporting utilities for training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _safe_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    return np.asarray(values)


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()


def plot_learning_curves(history: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    """Save training/validation accuracy and loss curves."""
    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss") for row in history]
    val_loss = [row.get("val_loss") for row in history]
    train_acc = [row.get("train_accuracy") for row in history]
    val_acc = [row.get("val_accuracy") for row in history]
    val_bal_acc = [row.get("val_bal_acc") for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Accuracy")
    axes[1].plot(epochs, val_acc, label="Val Accuracy")
    axes[1].plot(epochs, val_bal_acc, label="Val Balanced Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    path = output_dir / "learning_curves.png"
    _save_figure(path)
    return path


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], output_dir: Path) -> Path:
    """Save a confusion matrix heatmap."""
    fig = plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    path = output_dir / "confusion_matrix.png"
    _save_figure(path)
    return path


def plot_roc_curves(y_true: Sequence[int], y_prob: np.ndarray, class_names: Sequence[str], output_dir: Path) -> Path | None:
    """Save one-vs-rest ROC curves."""
    y_true = _safe_array(y_true)
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    if y_bin.shape[1] != len(class_names):
        return None

    fig = plt.figure(figsize=(8, 7))
    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, class_index], y_prob[:, class_index])
        auc = roc_auc_score(y_bin[:, class_index], y_prob[:, class_index])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    path = output_dir / "roc_curves.png"
    _save_figure(path)
    return path


def plot_precision_recall_curves(y_true: Sequence[int], y_prob: np.ndarray, class_names: Sequence[str], output_dir: Path) -> Path | None:
    """Save one-vs-rest precision-recall curves."""
    y_true = _safe_array(y_true)
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    if y_bin.shape[1] != len(class_names):
        return None

    fig = plt.figure(figsize=(8, 7))
    for class_index, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, class_index], y_prob[:, class_index])
        plt.plot(recall, precision, label=class_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    path = output_dir / "precision_recall_curves.png"
    _save_figure(path)
    return path


def plot_misclassified_samples(
    sample_paths: Sequence[str],
    labels: Sequence[int],
    preds: Sequence[int],
    class_names: Sequence[str],
    output_dir: Path,
    max_samples: int = 16,
) -> Path | None:
    """Save a small grid of misclassified validation samples when paths are available."""
    misclassified = [
        (path, label, pred)
        for path, label, pred in zip(sample_paths, labels, preds)
        if int(label) != int(pred)
    ]
    if not misclassified:
        return None

    selected = misclassified[:max_samples]
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for axis, (path, label, pred) in zip(axes.flat, selected):
        image = plt.imread(path)
        axis.imshow(image)
        axis.set_title(f"T: {class_names[int(label)]}\nP: {class_names[int(pred)]}", fontsize=9)
        axis.axis("off")

    for axis in axes.flat[len(selected):]:
        axis.axis("off")

    path = output_dir / "misclassified_samples.png"
    _save_figure(path)
    return path


def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, target_names=list(class_names), output_dict=True, zero_division=0)
    report["accuracy"] = accuracy_score(y_true, y_pred)
    report["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    report["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return report


def generate_evaluation_artifacts(
    history: Sequence[dict[str, Any]],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    sample_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Save metrics, plots, and a classification report for a training run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y_true_arr = _safe_array(y_true)
    y_pred_arr = _safe_array(y_pred)
    y_prob_arr = _safe_array(y_prob)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))
    report = classification_report_dict(y_true_arr, y_pred_arr, class_names)

    metrics = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "balanced_accuracy": report.get("macro avg", {}).get("recall", 0.0),
        "precision": precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "roc_auc": None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class_metrics": {name: report.get(name, {}) for name in class_names},
    }

    try:
        y_bin = label_binarize(y_true_arr, classes=list(range(len(class_names))))
        metrics["roc_auc"] = roc_auc_score(y_bin, y_prob_arr, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc"] = None

    plot_learning_curves(history, output_path)
    plot_confusion_matrix(cm, class_names, output_path)
    plot_roc_curves(y_true_arr, y_prob_arr, class_names, output_path)
    plot_precision_recall_curves(y_true_arr, y_prob_arr, class_names, output_path)
    if sample_paths is not None:
        plot_misclassified_samples(sample_paths, y_true_arr, y_pred_arr, class_names, output_path)

    return metrics