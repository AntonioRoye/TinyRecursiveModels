import os
import json
from typing import List

from dataset.build_arc_dataset import DataProcessConfig, convert_dataset


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_augmented(prefix: str, subsets: List[str], test_set_name: str, out_dir: str, num_aug: int, seed: int = 42):
    ensure_dir(out_dir)
    cfg = DataProcessConfig(
        input_file_prefix=prefix,
        output_dir=out_dir,
        subsets=subsets,
        test_set_name=test_set_name,
        seed=seed,
        num_aug=num_aug,
    )
    convert_dataset(cfg)


def main():
    # Assumes the ARC-AGI combined jsons are already present under TinyRecursiveModels/kaggle/combined
    base = os.path.join("TinyRecursiveModels", "kaggle", "combined")

    prefix = os.path.join(base, "arc-agi")
    subsets = [
        "training",
        "training2",
        "concept",
        "evaluation",
        "evaluation2",
    ]

    # Build multiple augmentation levels for optional accuracy-vs-aug curve
    levels = [0, 1, 4, 16, 64, 256, 1000]
    outputs = {}
    for k in levels:
        out_dir = os.path.join("data", f"arc-aug-{k}")
        build_augmented(prefix=prefix, subsets=subsets, test_set_name="evaluation", out_dir=out_dir, num_aug=k)
        outputs[str(k)] = out_dir

    # Write a small manifest with dataset locations
    manifest = {
        "paper_mode": outputs.get("1000"),
        "single_aug": outputs.get("0"),
        "all_levels": outputs,
    }
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "arc_exp1_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Prepared datasets:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


