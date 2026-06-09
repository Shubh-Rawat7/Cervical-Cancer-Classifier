"""Dataset analysis and visualization for the Herlev cervical cell dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing import (
    class_distribution,
    compute_image_statistics,
    discover_samples,
    plot_class_distribution,
    plot_sample_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Herlev dataset before training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()

    class_names, samples = discover_samples(args.data_dir)
    distribution = class_distribution(samples, class_names=class_names)
    statistics = compute_image_statistics(samples, image_size=args.image_size)
    sample_subset = []
    for class_name in class_names:
        sample_subset.extend([sample for sample in samples if sample.label == class_name][: args.samples_per_class])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(distribution, args.output_dir / "class_distribution.png")
    plot_sample_grid(sample_subset, args.output_dir / "sample_grid.png", image_size=args.image_size)

    report = {
        "class_names": class_names,
        "distribution": distribution,
        "statistics": statistics,
        "sample_count": len(samples),
    }
    (args.output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
