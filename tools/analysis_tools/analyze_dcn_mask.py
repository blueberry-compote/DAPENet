#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class RunningMaskStats:
    hist_bins: int
    sample_cap: int
    count: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    value_min: float = float("inf")
    value_max: float = float("-inf")
    hist: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=np.int64))
    samples: List[np.ndarray] = field(default_factory=list)
    sample_count: int = 0

    def __post_init__(self):
        self.hist = np.zeros(self.hist_bins, dtype=np.int64)

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        arr = arr.astype(np.float64, copy=False)
        self.count += int(arr.size)
        self.value_sum += float(arr.sum())
        self.value_sq_sum += float((arr * arr).sum())
        cur_min = float(arr.min())
        cur_max = float(arr.max())
        self.value_min = min(self.value_min, cur_min)
        self.value_max = max(self.value_max, cur_max)
        self.hist += np.histogram(arr, bins=self.hist_bins, range=(0.0, 1.0))[0]

        # Reservoir sampling for quantiles.
        flat = arr.ravel()
        for v in flat:
            if self.sample_count < self.sample_cap:
                self.samples.append(np.array([v], dtype=np.float64))
            else:
                j = np.random.randint(0, self.sample_count + 1)
                if j < self.sample_cap:
                    self.samples[j] = np.array([v], dtype=np.float64)
            self.sample_count += 1

    def finalize(self) -> Dict:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "quantiles": {},
                "histogram": {"bins": self.hist_bins, "range": [0.0, 1.0], "counts": self.hist.tolist()},
            }

        mean = self.value_sum / self.count
        var = max(self.value_sq_sum / self.count - mean * mean, 0.0)
        std = float(np.sqrt(var))
        sample_arr = np.concatenate(self.samples) if self.samples else np.array([], dtype=np.float64)
        quantiles = {}
        if sample_arr.size > 0:
            for q in [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]:
                quantiles[f"p{int(q * 100):02d}"] = float(np.quantile(sample_arr, q))

        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": std,
            "min": float(self.value_min),
            "max": float(self.value_max),
            "quantiles": quantiles,
            "histogram": {"bins": self.hist_bins, "range": [0.0, 1.0], "counts": self.hist.tolist()},
            "quantile_sampled_points": int(sample_arr.size),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze raw DCN mask (m) statistics from a trained checkpoint.")
    parser.add_argument("config", help="Path to config file.")
    parser.add_argument("checkpoint", help="Path to model checkpoint.")
    parser.add_argument("--img-dir", required=True, help="Directory containing images for analysis.")
    parser.add_argument("--device", default="cuda:0", help='Device, e.g. "cuda:0" or "cpu".')
    parser.add_argument("--max-images", type=int, default=200, help="Max number of images to process.")
    parser.add_argument("--hist-bins", type=int, default=20, help="Histogram bins in [0, 1].")
    parser.add_argument("--quantile-sample-cap", type=int, default=200000, help="Reservoir cap for quantile estimation.")
    parser.add_argument("--out", default="work_dirs/dcn_mask_stats.json", help="Output JSON path.")
    parser.add_argument(
        "--dump-raw",
        action="store_true",
        help="Dump raw mask values (all sample points) to .npy files.",
    )
    parser.add_argument(
        "--raw-out-dir",
        default="work_dirs/dcn_mask_raw",
        help="Directory for raw mask .npy files when --dump-raw is enabled.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only dump raw mask values and skip statistics JSON.",
    )
    return parser.parse_args()


def collect_images(img_dir: str) -> List[str]:
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    items = []
    for name in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, name)
        if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
            items.append(path)
    return items


def main():
    args = parse_args()
    np.random.seed(0)

    image_paths = collect_images(args.img_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {args.img_dir}")
    image_paths = image_paths[: args.max_images]

    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    align_modules = getattr(model.neck, "align_modules", None)
    if align_modules is None:
        raise AttributeError("model.neck has no attribute 'align_modules'. Check neck type.")

    if args.raw_only and not args.dump_raw:
        raise ValueError("--raw-only requires --dump-raw.")

    stats = None
    if not args.raw_only:
        stats = {
            idx: RunningMaskStats(hist_bins=args.hist_bins, sample_cap=args.quantile_sample_cap)
            for idx in range(len(align_modules))
        }

    handles = []
    current_image_info = {"index": -1, "name": "unknown"}

    if args.dump_raw:
        os.makedirs(args.raw_out_dir, exist_ok=True)
        for i in range(len(align_modules)):
            os.makedirs(os.path.join(args.raw_out_dir, f"align_{i}"), exist_ok=True)

    def _hook_factory(idx):
        def _hook(module, module_inputs, module_output):
            # module_inputs = (p_i, p_ip1_up)
            p_i, p_ip1_up = module_inputs
            with torch.no_grad():
                out = module.offset_and_mask(torch.cat([p_i, p_ip1_up], dim=1))
                mask = torch.sigmoid(out[:, module.offset_dim:, :, :])
                mask_np = mask.detach().float().cpu().numpy()
                if stats is not None:
                    stats[idx].update(mask_np)
                if args.dump_raw:
                    img_idx = current_image_info["index"]
                    img_name = current_image_info["name"]
                    save_name = f"{img_idx:06d}_{img_name}.npy"
                    save_path = os.path.join(args.raw_out_dir, f"align_{idx}", save_name)
                    np.save(save_path, mask_np)

        return _hook

    for i, mod in enumerate(align_modules):
        handles.append(mod.register_forward_hook(_hook_factory(i)))

    try:
        for i, img_path in enumerate(image_paths, start=1):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            current_image_info["index"] = i
            current_image_info["name"] = img_name
            inference_detector(model, img_path)
            if i % 20 == 0 or i == len(image_paths):
                print(f"[{i}/{len(image_paths)}] processed")
    finally:
        for h in handles:
            h.remove()

    if stats is not None:
        result = {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "img_dir": args.img_dir,
            "processed_images": len(image_paths),
            "levels": {f"align_{k}": v.finalize() for k, v in stats.items()},
        }

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Saved mask stats to: {args.out}")
        for level_name, level_stats in result["levels"].items():
            if level_stats["count"] == 0:
                print(f"{level_name}: no data")
                continue
            print(
                f"{level_name}: mean={level_stats['mean']:.4f}, std={level_stats['std']:.4f}, "
                f"min={level_stats['min']:.4f}, max={level_stats['max']:.4f}"
            )

    if args.dump_raw:
        print(f"Saved raw mask values to: {args.raw_out_dir}")


if __name__ == "__main__":
    main()
