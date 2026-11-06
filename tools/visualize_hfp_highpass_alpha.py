#!/usr/bin/env python3
"""可视化 `hfp_fpn.py` 中不同 alpha 的 FFT 高通滤波效果（对 RGB 图像）。

用法示例：
  python tools/visualize_hfp_highpass_alpha.py \
    --image demo/demo.jpg \
    --alphas 0.0,0.1,0.2,0.3,0.5 \
    --out out_highpass.png

说明：
  - 这里直接把 RGB 图像当作 [1,3,H,W] 的张量送入 `fft_highpass_filter`。
  - 高通输出会包含负值，默认用 abs() 后做 min-max 归一化用于显示。
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
    # 优先 PIL（更轻），如果没有则回退到 cv2
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
    """把任意 float 图像张量映射到 [0,1] 以显示（按整张图 min-max）。"""
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def _parse_alphas(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if not parts:
        raise ValueError('alphas is empty')
    alphas: List[float] = []
    for p in parts:
        alphas.append(float(p))
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='输入 RGB 图片路径')
    parser.add_argument(
        '--alphas',
        default='0.0,0.1,0.2,0.3,0.5',
        help='alpha 列表，逗号分隔（范围 [0,1]）',
    )
    parser.add_argument('--out', default='hfp_highpass_alpha.png', help='输出图片路径')
    parser.add_argument(
        '--device',
        default='cpu',
        help="运行设备：'cpu' 或 'cuda:0' 等",
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='如果可用，弹窗显示（也会保存到 --out）',
    )
    args = parser.parse_args()

    _add_repo_root_to_path()

    import torch

    from mmdet.models.necks.hfp_fpn import fft_highpass_filter

    img = _read_image_rgb(args.image)  # [H,W,3] in [0,1]
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    x = x.to(device=args.device)

    alphas = _parse_alphas(args.alphas)

    # 计算各 alpha 的高通结果
    outs_rgb = []
    outs_heat = []
    for a in alphas:
        y = fft_highpass_filter(x, alpha=float(a))  # [1,3,H,W]
        y_np = y[0].permute(1, 2, 0).detach().float().cpu().numpy()  # [H,W,3]

        # 用 abs() 更直观展示“高频强度”
        y_abs = np.abs(y_np)
        y_vis = _to_display_rgb(y_abs)

        # 灰度热力图：通道均值幅值
        heat = y_abs.mean(axis=2)
        heat_vis = _to_display_rgb(heat)

        outs_rgb.append(y_vis)
        outs_heat.append(heat_vis)

    # 绘图：第一行原图 + 每个 alpha 的 RGB；第二行是对应热力图
    import matplotlib.pyplot as plt

    n = len(alphas)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.2 * (n + 1), 6.0))

    axes[0, 0].imshow(_to_display_rgb(img))
    axes[0, 0].set_title('orig')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(np.zeros((img.shape[0], img.shape[1]), dtype=np.float32), cmap='gray')
    axes[1, 0].set_title('heat (empty)')
    axes[1, 0].axis('off')

    for i, a in enumerate(alphas, start=1):
        axes[0, i].imshow(outs_rgb[i - 1])
        axes[0, i].set_title(f'highpass | alpha={a:g}')
        axes[0, i].axis('off')

        axes[1, i].imshow(outs_heat[i - 1], cmap='magma')
        axes[1, i].set_title(f'heat | alpha={a:g}')
        axes[1, i].axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f'Saved: {args.out}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()

