#!/usr/bin/env python3
"""可视化不同 hfp_init_alpha 对同一张图的高通效果（使用 HFP_Learnable 的软掩码滤波）。

用法示例：
  python tools/visualize_hfp_init_alpha.py --image demo/demo.jpg
  python tools/visualize_hfp_init_alpha.py --image path/to/img.jpg --alphas 0.15,0.2,0.25,0.3,0.4 --out hfp_init_alpha_vis.png

与 visualize_hfp_highpass_alpha.py 的区别：本脚本使用 fft_highpass_filter_adaptive（软掩码），
与 HFP_FPN_Learnable 中的滤波一致，便于理解 init_alpha 的强弱效果。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np


def _add_repo_root_to_path() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _read_image_rgb(path: str) -> np.ndarray:
    """读取 RGB 图像为 float32, range [0,1], shape [H,W,3]."""
    try:
        from PIL import Image  # type: ignore
        img = Image.open(path).convert('RGB')
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr
    except Exception:
        import cv2  # type: ignore
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f'Cannot read image: {path}')
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        return rgb


def _to_display_rgb(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """把任意 float 图像映射到 [0,1] 以显示（整图 min-max）。"""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def _parse_alphas(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if not parts:
        raise ValueError('alphas is empty')
    return [float(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize hfp_init_alpha effect (soft-mask highpass)')
    parser.add_argument('--image', required=True, help='输入 RGB 图片路径')
    parser.add_argument(
        '--alphas',
        default='0.15,0.2,0.25,0.3,0.4',
        help='alpha 列表，逗号分隔（对应弱高通→强高通），范围 [0,1]',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=2.0,
        help='软掩码 temperature，与 config 中 hfp_highpass_temperature 一致',
    )
    parser.add_argument('--out', default='hfp_init_alpha_vis.png', help='输出图片路径')
    parser.add_argument('--device', default='cpu', help="'cpu' 或 'cuda:0'")
    parser.add_argument('--show', action='store_true', help='弹窗显示并保存')
    args = parser.parse_args()

    _add_repo_root_to_path()

    import torch
    from mmdet.models.necks.hfp_fpn_learnable import fft_highpass_filter_adaptive

    img = _read_image_rgb(args.image)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    x = x.to(device=args.device)

    alphas = _parse_alphas(args.alphas)

    outs_rgb = []
    outs_heat = []
    for a in alphas:
        alpha_t = torch.tensor([a], device=x.device, dtype=x.dtype)
        y = fft_highpass_filter_adaptive(
            x, alpha_t, temperature=args.temperature
        )
        y_np = y[0].permute(1, 2, 0).detach().float().cpu().numpy()
        y_abs = np.abs(y_np)
        outs_rgb.append(_to_display_rgb(y_abs))
        outs_heat.append(_to_display_rgb(y_abs.mean(axis=2)))

    import matplotlib.pyplot as plt

    n = len(alphas)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.2 * (n + 1), 6.0))

    axes[0, 0].imshow(_to_display_rgb(img))
    axes[0, 0].set_title('orig')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(
        np.zeros((img.shape[0], img.shape[1]), dtype=np.float32), cmap='gray'
    )
    axes[1, 0].set_title('heat (empty)')
    axes[1, 0].axis('off')

    for i, a in enumerate(alphas, start=1):
        axes[0, i].imshow(outs_rgb[i - 1])
        axes[0, i].set_title(f'highpass | alpha={a:g}')
        axes[0, i].axis('off')
        axes[1, i].imshow(outs_heat[i - 1], cmap='magma')
        axes[1, i].set_title(f'heat | alpha={a:g}')
        axes[1, i].axis('off')

    fig.suptitle('hfp_init_alpha 效果对比（软掩码，弱→强高通）', fontsize=12)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f'Saved: {args.out}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
