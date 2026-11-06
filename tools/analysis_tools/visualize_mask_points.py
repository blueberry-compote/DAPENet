#!/usr/bin/env python3
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize sampling-point locations and mask values from a mask .npy file."
    )
    parser.add_argument("image", help="Path to original image.")
    parser.add_argument("mask_npy", help="Path to mask .npy file. Expected shape [N, K, H, W] or [K, H, W].")
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch index if mask is 4D. Default: 0.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=200,
        help="Number of highest-value points to draw per channel.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/mask_point_vis",
        help="Output directory for visualizations.",
    )
    parser.add_argument(
        "--mark-low-thr",
        type=float,
        default=None,
        help="Mark all points whose mask value is below this threshold (e.g. 0.5).",
    )
    return parser.parse_args()


def ensure_mask_3d(mask: np.ndarray, batch_index: int) -> np.ndarray:
    if mask.ndim == 4:
        if batch_index < 0 or batch_index >= mask.shape[0]:
            raise ValueError(f"batch_index={batch_index} out of range for shape {mask.shape}")
        return mask[batch_index]
    if mask.ndim == 3:
        return mask
    raise ValueError(f"Unsupported mask shape: {mask.shape}, expected [N,K,H,W] or [K,H,W].")


def to_overlay(base_bgr: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    m = mask_2d.astype(np.float32)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m_uint8 = (m * 255).astype(np.uint8)
    heat = cv2.applyColorMap(m_uint8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(base_bgr, 1 - alpha, heat, alpha, 0)


def draw_topk_points(base_bgr: np.ndarray, mask_2d: np.ndarray, topk: int) -> np.ndarray:
    vis = base_bgr.copy()
    h_img, w_img = vis.shape[:2]
    h_m, w_m = mask_2d.shape
    flat = mask_2d.reshape(-1)
    k = min(topk, flat.size)
    if k <= 0:
        return vis

    idx = np.argpartition(flat, -k)[-k:]
    ys, xs = np.unravel_index(idx, (h_m, w_m))
    vals = flat[idx]
    order = np.argsort(-vals)
    ys, xs, vals = ys[order], xs[order], vals[order]

    for i, (y, x, v) in enumerate(zip(ys, xs, vals)):
        px = int((x + 0.5) * w_img / w_m)
        py = int((y + 0.5) * h_img / h_m)
        color = (0, 255, 255) if i < min(20, k) else (0, 200, 0)
        radius = 3 if i < min(50, k) else 2
        cv2.circle(vis, (px, py), radius, color, -1)
        if i < min(20, k):
            cv2.putText(
                vis,
                f"{v:.3f}",
                (px + 3, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return vis


def draw_low_value_points(base_bgr: np.ndarray, mask_2d: np.ndarray, thr: float) -> np.ndarray:
    vis = base_bgr.copy()
    h_img, w_img = vis.shape[:2]
    h_m, w_m = mask_2d.shape

    ys, xs = np.where(mask_2d < thr)
    for y, x in zip(ys, xs):
        px = int((x + 0.5) * w_img / w_m)
        py = int((y + 0.5) * h_img / h_m)
        # 低值点用红色标注
        cv2.circle(vis, (px, py), 1, (0, 0, 255), -1)

    total = mask_2d.size
    cnt = len(xs)
    ratio = cnt / max(total, 1)
    cv2.putText(
        vis,
        f"mask < {thr:.3f}: {cnt}/{total} ({ratio:.2%})",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def save_histogram(mask_2d: np.ndarray, out_path: str):
    plt.figure(figsize=(6, 4))
    plt.hist(mask_2d.reshape(-1), bins=40, range=(0.0, 1.0))
    plt.title("Mask Value Histogram")
    plt.xlabel("mask value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    mask_raw = np.load(args.mask_npy)
    mask = ensure_mask_3d(mask_raw, args.batch_index).astype(np.float32)  # [K,H,W]

    base_name = os.path.splitext(os.path.basename(args.mask_npy))[0]
    k_channels = mask.shape[0]

    # 1) 每个采样通道：热力图 + topk 点位标注图
    for k in range(k_channels):
        mk = mask[k]
        overlay = to_overlay(img, mk)
        points = draw_topk_points(img, mk, args.topk)
        panels = [overlay, points]
        if args.mark_low_thr is not None:
            panels.append(draw_low_value_points(img, mk, args.mark_low_thr))
        merged = np.hstack(panels)
        cv2.putText(
            merged,
            f"channel={k}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(os.path.join(args.out_dir, f"{base_name}_ch{k}_vis.png"), merged)
        save_histogram(mk, os.path.join(args.out_dir, f"{base_name}_ch{k}_hist.png"))

    # 2) 通道均值图（整体mask响应）
    mean_mask = mask.mean(axis=0)
    mean_overlay = to_overlay(img, mean_mask)
    mean_points = draw_topk_points(img, mean_mask, args.topk)
    mean_panels = [mean_overlay, mean_points]
    if args.mark_low_thr is not None:
        mean_panels.append(draw_low_value_points(img, mean_mask, args.mark_low_thr))
    mean_merged = np.hstack(mean_panels)
    cv2.putText(
        mean_merged,
        "channel_mean",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(os.path.join(args.out_dir, f"{base_name}_channel_mean_vis.png"), mean_merged)
    save_histogram(mean_mask, os.path.join(args.out_dir, f"{base_name}_channel_mean_hist.png"))

    print(f"saved visualizations to: {args.out_dir}")
    print(f"mask shape used: {mask.shape} (K,H,W)")


if __name__ == "__main__":
    main()
