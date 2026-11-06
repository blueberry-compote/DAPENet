#!/usr/bin/env python3
import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export all mask values from .npy to plain text (one value per line)."
    )
    parser.add_argument("npy_path", help="Path to input .npy mask file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output txt path. Default: <npy_path_without_ext>_values.txt",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Float precision for output values.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.npy_path):
        raise FileNotFoundError(f"Input file not found: {args.npy_path}")

    arr = np.load(args.npy_path)
    values = arr.reshape(-1)

    if args.out is None:
        base, _ = os.path.splitext(args.npy_path)
        out_path = base + "_values.txt"
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fmt = f"%.{args.precision}f"

    with open(out_path, "w", encoding="utf-8") as f:
        for v in values:
            f.write((fmt % float(v)) + "\n")

    print(f"input_shape: {arr.shape}")
    print(f"total_values: {values.size}")
    print(f"saved_to: {out_path}")


if __name__ == "__main__":
    main()
