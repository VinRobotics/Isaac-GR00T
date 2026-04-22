"""
Test script for LeRobotMixtureDatasetV2.

Usage:
    python scripts/test_mixture_dataset_v2.py \
        --dataset_path /path/to/old_data1 /path/to/old_data2 \
        --new_dataset_path /path/to/new_data \
        --data_config fourier_gr1_arms_only \
        --num_batches 5 \
        --batch_size 4
"""

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import tyro
from torch.utils.data import DataLoader

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotMixtureDatasetV2, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config


@dataclass
class Args:
    dataset_path: List[str]
    """Old / replay dataset path(s)."""

    new_dataset_path: List[str]
    """New task dataset path(s)."""

    data_config: str = "fourier_gr1_arms_only"
    """Data config name (same as gr00t_finetune.py)."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag."""

    batch_size: int = 4
    """Batch size (should be even so each batch is 50/50)."""

    num_batches: int = 5
    """Number of batches to inspect."""

    video_backend: str = "torchcodec"
    """Video backend."""

    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(arr, decimals: int = 4) -> str:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 0:
        return f"{float(a):.{decimals}f}"
    if a.size <= 6:
        vals = ", ".join(f"{v:.{decimals}f}" for v in a.flat)
        return f"[{vals}]"
    return (
        f"[{a.flat[0]:.{decimals}f}, {a.flat[1]:.{decimals}f}, ... "
        f"{a.flat[-1]:.{decimals}f}]  (shape {list(a.shape)})"
    )


def print_stats(label: str, metadata_dict: dict, indent: int = 2):
    pad = " " * indent
    print(f"\n{'='*60}")
    print(f"  STATISTICS: {label}")
    print(f"{'='*60}")
    for tag, metadata in metadata_dict.items():
        print(f"{pad}tag = {tag}")
        for modality in ("state", "action"):
            mod_stats = getattr(metadata.statistics, modality)
            print(f"{pad}  [{modality}]")
            for key, stat in mod_stats.items():
                print(f"{pad}    {key}:")
                print(f"{pad}      mean = {_fmt(stat.mean)}")
                print(f"{pad}      std  = {_fmt(stat.std)}")
                print(f"{pad}      min  = {_fmt(stat.min)}")
                print(f"{pad}      max  = {_fmt(stat.max)}")
                print(f"{pad}      q01  = {_fmt(stat.q01)}")
                print(f"{pad}      q99  = {_fmt(stat.q99)}")


def load_as_mixture(
    dataset_path: List[str],
    balance_dataset_weights: bool,
    balance_trajectory_weights: bool,
    modality_configs,
    transforms,
    embodiment_tag,
    video_backend: str,
) -> LeRobotMixtureDataset:
    single_datasets = []
    for p in dataset_path:
        assert os.path.exists(p), f"Dataset path does not exist: {p}"
        single_datasets.append(
            LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=video_backend,
            )
        )
    return LeRobotMixtureDataset(
        data_mixture=[(d, 1.0) for d in single_datasets],
        mode="train",
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=42,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )


def collate_fn(batch):
    """Collate a list of sample dicts, keeping lists for string fields."""
    collated = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], np.ndarray):
            collated[key] = np.stack(values)
        elif isinstance(values[0], list) and isinstance(values[0][0], str):
            collated[key] = values  # list of lists of strings
        else:
            collated[key] = values
    return collated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    embodiment_tag = EmbodimentTag(args.embodiment_tag)
    data_config_cls = load_data_config(args.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    kwargs = dict(
        balance_dataset_weights=args.balance_dataset_weights,
        balance_trajectory_weights=args.balance_trajectory_weights,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=args.video_backend,
    )

    print("\n>>> Loading OLD dataset mixture …")
    old_mixture = load_as_mixture(args.dataset_path, **kwargs)
    print(f"    len(old_mixture) = {len(old_mixture)}")

    print("\n>>> Loading NEW dataset mixture …")
    new_mixture = load_as_mixture(args.new_dataset_path, **kwargs)
    print(f"    len(new_mixture) = {len(new_mixture)}")

    # ----- per-mixture stats -----
    print_stats("OLD dataset", old_mixture.merged_metadata)
    print_stats("NEW dataset", new_mixture.merged_metadata)

    # ----- build V2 and show merged stats -----
    print("\n>>> Building LeRobotMixtureDatasetV2 …")
    v2 = LeRobotMixtureDatasetV2(
        old_data_mixture=old_mixture,
        new_data_mixture=new_mixture,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    print(f"    len(v2) = {len(v2)}  (= 2 × {len(v2)//2})")
    print(v2)

    print_stats("MERGED (old + new, 50/50)", v2.merged_metadata)

    # ----- batch inspection -----
    print(f"\n{'='*60}")
    print(f"  BATCH INSPECTION  (batch_size={args.batch_size}, num_batches={args.num_batches})")
    print(f"{'='*60}")

    loader = DataLoader(
        v2,
        batch_size=args.batch_size,
        shuffle=False,        # keep even=old / odd=new interleaving visible
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Find language keys in the modality configs
    language_keys = modality_configs.get("language")
    lang_keys = language_keys.modality_keys if language_keys is not None else []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        print(f"\n--- Batch {batch_idx} ---")

        # Which half came from old vs new (even global index = old, odd = new)
        start_idx = batch_idx * args.batch_size
        sources = ["OLD" if (start_idx + i) % 2 == 0 else "NEW" for i in range(args.batch_size)]

        if lang_keys:
            for lang_key in lang_keys:
                if lang_key in batch:
                    print(f"  [{lang_key}]")
                    for i, (src, instructions) in enumerate(zip(sources, batch[lang_key])):
                        # instructions is a list of strings (one per delta-index step)
                        text = instructions[0] if isinstance(instructions, list) else instructions
                        print(f"    sample[{i}] ({src}): {text!r}")
        else:
            print("  (no language modality configured)")

        # Print a brief shape summary for numeric keys on the first batch
        if batch_idx == 0:
            print("  Key shapes:")
            for key, val in batch.items():
                if isinstance(val, np.ndarray):
                    print(f"    {key}: {list(val.shape)}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
