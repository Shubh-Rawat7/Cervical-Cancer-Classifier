"""Analyze Herlev dataset structure and class distribution."""

from pathlib import Path
from collections import defaultdict
import argparse


def analyze_dataset(root_dir: str):
    """Analyze dataset structure and report statistics."""
    root = Path(root_dir)
    
    print(f"\n{'='*70}")
    print(f"Analyzing dataset: {root_dir}")
    print(f"{'='*70}\n")
    
    for split in ['train', 'val', 'test']:
        split_dir = root / split
        if not split_dir.exists():
            print(f"{split:10} : MISSING")
            continue
        
        class_counts = defaultdict(int)
        total = 0
        
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = sum(1 for _ in class_dir.glob('*') if _.is_file())
                class_counts[class_dir.name] = count
                total += count
        
        print(f"{split.upper()} split:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {class_name:15} : {count:4d} ({pct:5.1f}%)")
        print(f"  {'TOTAL':15} : {total:4d}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Herlev dataset')
    parser.add_argument('--data-dir', type=str, required=True, help='Root dataset directory')
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir)
