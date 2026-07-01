"""Evaluate the best checkpoint on the test set and save confusion matrix and metrics."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from models.hybrid_model import HerlevHybridClassifier
from preprocessing import discover_class_directories, list_image_files, ImageRecord, filter_quality_and_duplicates
from dataset import HerlevDataset, build_eval_transform
from torch.utils.data import DataLoader
from training.engine import evaluate
from losses import build_criterion
from config import CLASS_NAMES, NUM_CLASSES, DEFAULT_IMAGE_SIZE


def collect_test_samples(root: Path, class_names=CLASS_NAMES):
    test_samples = []
    if (root / 'test').exists():
        for class_name in class_names:
            class_dir = root / 'test' / class_name
            if not class_dir.exists():
                continue
            for p in list_image_files(class_dir):
                test_samples.append(ImageRecord(path=p, label=class_name))
        return test_samples
    # fallback: if no explicit test dir, return empty
    return []


def plot_and_save_cm(cm, labels, out_path: Path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()


def main():
    print('Running evaluate_test.py from', Path(__file__).resolve())
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('Checkpoints')/ 'best_model.pt')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--min-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nEvaluating checkpoint:', args.checkpoint)
    # Load checkpoint and try to reconstruct model from saved metadata.
    ck = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(ck, dict):
        state_dict = ck.get('state_dict', ck)
        metadata = ck.get('config', {}) or {}
    else:
        state_dict = ck
        metadata = {}

    # strip common prefixes from state dict keys
    def _strip_prefix(sd):
        stripped = {}
        for k, v in sd.items():
            new_k = k
            for prefix in ('module.', 'ema_model.', 'model.'):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            stripped[new_k] = v
        return stripped

    state_dict = _strip_prefix(state_dict)

    # Infer model construction parameters from metadata or state_dict shapes
    backbone = metadata.get('backbone', 'vim_base_patch16_224')
    image_size = int(metadata.get('image_size', args.image_size))
    mamba_layers = int(metadata.get('mamba_layers', 2))
    attn_heads = int(metadata.get('attn_heads', 4))
    dropout = float(metadata.get('dropout', 0.2))
    if 'embed_dim' in metadata:
        embed_dim = int(metadata.get('embed_dim'))
    else:
        # head.1.weight is expected shape (hidden_dim, embed_dim)
        if 'head.1.weight' in state_dict:
            embed_dim = int(state_dict['head.1.weight'].shape[1])
        else:
            embed_dim = 256

    if 'num_classes' in metadata:
        num_classes = int(metadata.get('num_classes'))
    else:
        if 'head.5.weight' in state_dict:
            num_classes = int(state_dict['head.5.weight'].shape[0])
        else:
            num_classes = NUM_CLASSES

    model = HerlevHybridClassifier(
        backbone=backbone,
        num_classes=num_classes,
        image_size=image_size,
        embed_dim=embed_dim,
        mamba_layers=mamba_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        pretrained=False,
    )

    # If checkpoint head uses a different hidden dim / output shape, rebuild head to match
    try:
        if 'head.1.weight' in state_dict:
            head_hidden = int(state_dict['head.1.weight'].shape[0])
        else:
            head_hidden = max(embed_dim // 2, 128)
        if 'head.5.weight' in state_dict:
            head_out = int(state_dict['head.5.weight'].shape[0])
        else:
            head_out = num_classes

        # Only replace head if shapes differ
        cur_state = model.head[1].weight.shape if hasattr(model.head[1], 'weight') else None
        if cur_state is None or cur_state[0] != head_hidden or head_out != num_classes:
            new_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, head_out),
            )
            model.head = new_head
            # update metadata local var
            num_classes = head_out
    except Exception:
        pass

    # Load parameters (allowing missing keys).
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        print('Warning: partial load failed:', exc)
        # Attempt to load selectively
        model_state = model.state_dict()
        compatible = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(compatible)
        model.load_state_dict(model_state)

    model.to(device)
    model.eval()

    # collect test samples
    class_dirs = discover_class_directories(Path(args.data_dir), class_names=CLASS_NAMES)
    test_samples = collect_test_samples(Path(args.data_dir), class_names=class_dirs)
    if not test_samples:
        print('No explicit test split found under data dir.')
        return

    # filter quality with same min_size
    test_samples, duplicates, issues = filter_quality_and_duplicates(test_samples, min_size=args.min_size)

    dataset = HerlevDataset(test_samples, transform=build_eval_transform(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # build dummy criterion using counts from test set (for loss only)
    counts = torch.tensor(np.bincount([CLASS_NAMES.index(s.label) for s in test_samples], minlength=NUM_CLASSES), dtype=torch.float32, device=device)
    criterion = build_criterion(counts=counts, loss_type='class_balanced_focal')

    metrics = evaluate(model, loader, criterion, device, use_amp=False)

    out_dir = Path('backend') / 'Checkpoints'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))

    cm = np.array(metrics.get('confusion_matrix', []))
    if cm.size:
        plot_and_save_cm(cm, CLASS_NAMES, out_dir / 'test_confusion_matrix.png')
        print('Saved confusion matrix to', out_dir / 'test_confusion_matrix.png')
    print('Metrics saved to', out_dir / 'test_metrics.json')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
