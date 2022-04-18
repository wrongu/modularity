#!/usr/bin/env python
import torch
import argparse
import warnings
from pathlib import Path
from typing import Union
from eval import evaluate


def bestify(weights_dir: Union[str, Path],
            field: str = "val_loss",
            mode: str = "min",
            overwrite: bool = False,
            data_dir: Union[str, Path] = Path("data")):
    weights_dir = Path(weights_dir)
    data_dir = Path(data_dir)
    best_file = weights_dir / "best.ckpt"

    if mode not in ("max", "min"):
        raise ValueError(f"Argument 'mode' must be 'max' or 'min' but is {mode}")

    best_ckpt, best_val = None, float("-inf") if mode == "max" else float("+inf")
    for ckpt in weights_dir.glob("*.ckpt"):
        if ckpt.is_symlink():
            continue
        evaluate(ckpt, data_dir, metrics=[field])
        data = torch.load(ckpt, map_location="cpu")
        val = data[field]
        if (mode == "max" and val > best_val) or (mode == "min" and val < best_val):
            best_ckpt, best_val = ckpt.resolve(), val

    if best_ckpt is None:
        warnings.warn(f"No checkpoints to check in {weights_dir}")
        return

    # Only raise error if 'overwrite' is False and the 'best.ckpt' referee would change.
    if best_file .exists():
        previous_best = best_file .resolve()
        if previous_best != best_ckpt and not overwrite:
            raise FileExistsError()
        elif previous_best == best_ckpt:
            return

    # If we reach here, it's time to create a new symlink called "best.ckpt" that points to the best_ckpt file
    best_file .symlink_to(target=best_ckpt, target_is_directory=False)


def main(args):
    assert args.directory.is_dir(), f"{args.directory} is not a directory"
    bestify(args.directory, args.field, args.mode, args.overwrite)
    if args.recurse:
        for sub in args.directory.iterdir():
            if sub.is_dir():
                args.directory = sub
                main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", default=Path("."), type=Path)
    parser.add_argument("--field", default="val_loss", type=str)
    parser.add_argument("--mode", default="min", type=str, choices=["min", "max"])
    parser.add_argument("--recurse", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    main(args)