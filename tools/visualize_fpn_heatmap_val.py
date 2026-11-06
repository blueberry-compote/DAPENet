import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.image import tensor2imgs

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import (
    build_dp,
    compat_cfg,
    setup_multi_processes,
    update_data_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FPN heatmaps (feature response) on val/test"
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/fpn_heatmap_vis",
        help="output directory for heatmaps",
    )
    parser.add_argument(
        "--level-idx",
        type=int,
        default=0,
        help="FPN output index to visualize (0->P3, 1->P4, ...)",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="overlay alpha")
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="maximum images to process (default: 50)",
    )
    parser.add_argument(
        "--target-stems",
        default=None,
        type=str,
        help=(
            "comma-separated image stems to visualize only, e.g. "
            "`0000001_05999_d_0000011`"
        ),
    )
    parser.add_argument(
        "--workers-per-gpu", type=int, default=2, help="dataloader workers"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="device string, e.g. cuda:0 or cpu",
    )

    # Optional composition with existing det visualization directory
    parser.add_argument(
        "--det-dir",
        default=None,
        help="optional: directory that contains detection images saved by tools/test.py --show-dir",
    )
    parser.add_argument(
        "--compose-dir",
        default=None,
        help="optional: output directory for side-by-side (heatmap + det)",
    )
    return parser.parse_args()


#def compute_heatmap_rgb(fmap_chw, cmap="jet"):
    """
    fmap_chw: torch Tensor or np array with shape [C,H,W]
    return:
      heat_rgb_u8: uint8 RGB image [H,W,3]
    """
    if torch.is_tensor(fmap_chw):
        fmap = fmap_chw.detach().cpu().float().numpy()
    else:
        fmap = fmap_chw.astype(np.float32)

    heat = fmap.mean(axis=0)  # [H,W]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    colormap = plt.get_cmap(cmap)
    heat_rgba = colormap(heat)  # [H,W,4] in [0,1]
    heat_rgb_u8 = (heat_rgba[..., :3] * 255.0).astype(np.uint8)
    return heat_rgb_u8


#def compute_heatmap_rgb(fmap_chw, cmap="viridis"):

    """
    生成和示例图一致的专业热力图：深蓝色背景，高响应亮蓝/紫色。
    - cmap: viridis (论文最常用, 蓝-黄渐变) 或 jet_r (红蓝反转)
    """
    if torch.is_tensor(fmap_chw):
        fmap = fmap_chw.detach().cpu().float().numpy()
    else:
        fmap = fmap_chw.astype(np.float32)

    # 1. 计算通道均值
    heat = fmap.mean(axis=0)  # [H, W]
    
    # 2. 关键改进：只保留前 20% 的高响应区域（类似示例图）
    # 如果你想更极致，可以改成 percentile=10，背景就会更黑
    threshold = np.percentile(heat, 80) 
    heat = np.maximum(heat - threshold, 0)  # 低于阈值的置为0
    
    # 3. 归一化（只在非零区域归一化，防止背景被拉伸成亮色）
    heat_max = heat.max()
    if heat_max > 1e-8:
        heat = heat / heat_max
    
    # 4.  Colormap 选择：
    # - viridis: 蓝-黄，专业耐看，和示例图最像
    # - jet_r: 红-蓝 反转，颜色对比强烈
    colormap = plt.get_cmap(cmap)
    
    # 生成 RGBA (A通道为alpha)
    heat_rgba = colormap(heat)  # [H, W, 4]
    
    # 5. 关键步骤：设置透明度，让背景接近黑色/透明
    # A通道即透明度，这里我们让响应低的点透明
    heat_rgba[..., 3] = (heat > 0.1).astype(np.float32)  # 低于0.1的完全透明
    
    # 转换为 uint8
    heat_rgb_u8 = (heat_rgba[..., :3] * 255.0).astype(np.uint8)
    
    return heat_rgb_u8    


def compute_heatmap_rgb(fmap_chw, cmap="jet"):
    """
    生成专业的热力图：深色背景，突出高响应区域。
    """
    if torch.is_tensor(fmap_chw):
        fmap = fmap_chw.detach().cpu().float().numpy()
    else:
        fmap = fmap_chw.astype(np.float32)

    # 1. 聚合通道信息 (取均值或最大值)
    # 对于小目标检测，取 max 有时比 mean 更能捕获微弱信号
    heat = np.mean(fmap, axis=0) 

    # 2. 【关键】背景过滤 - 抑制底噪
    # 别人的图之所以干净，是因为他们只显示了 Top 10% 或 20% 的响应
    # 计算一个鲁棒的阈值：均值 + 标准差，或者使用分位数
    threshold = np.percentile(heat, 40) # 过滤掉 70% 的低响应区域
    heat = heat - threshold
    heat = np.maximum(heat, 0) # 负值全部归零

    # 3. 归一化到 [0, 1]
    h_max = heat.max()
    if h_max > 1e-8:
        heat = heat / h_max

    # 4. 映射颜色
    # 如果想要深蓝底，用 'jet'
    # 如果想要纯黑底，用 'inferno' 或 'magma'
    colormap = plt.get_cmap(cmap)
    heat_rgba = colormap(heat) # 返回 [H, W, 4] 浮点型数据
    
    # 5. 转换为 uint8 RGB
    heat_rgb_u8 = (heat_rgba[..., :3] * 255.0).astype(np.uint8)
    
    return heat_rgb_u8



def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compose_dir:
        compose_dir = Path(args.compose_dir)
        compose_dir.mkdir(parents=True, exist_ok=True)
    else:
        compose_dir = None

    target_stems = None
    if args.target_stems:
        target_stems = {s.strip() for s in args.target_stems.split(",") if s.strip()}

    cfg = Config.fromfile(args.config)
    update_data_root(cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    # Align with tools/test.py behavior
    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    else:
        if cfg.model.get("backbone", None) is not None and "init_cfg" in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None

    if cfg.model.get("neck") is not None and isinstance(cfg.model.neck, list):
        # Keep simple: most configs here use a single neck dict
        pass

    cfg.model.train_cfg = None

    # Parse device id for DataParallel (only when CUDA is available)
    if (
        isinstance(args.device, str)
        and args.device.startswith("cuda:")
        and torch.cuda.is_available()
    ):
        gpu_id = int(args.device.split(":")[1])
        device_ids = [gpu_id]
    else:
        device_ids = None

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")

    # Ensure test_mode=True so the dataset returns test pipeline outputs
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    # Match tools/test.py: wrap in DP so DataContainer is correctly scattered/unwrapped
    if device_ids:
        # build_dp expects device type string like "cuda"/"cpu"
        model = build_dp(model, device="cuda", device_ids=device_ids)
        # forward hook should be registered on the underlying module
        model_to_hook = model.module
    else:
        model_to_hook = model
        model = model.to(device)

    model.eval()

    if not hasattr(model_to_hook, "neck") or model_to_hook.neck is None:
        raise RuntimeError("Model has no `neck` module to visualize.")

    captured = {}

    def neck_hook(module, inputs, outputs):
        # FPN-like neck returns a tuple of feature maps
        captured["neck_outs"] = outputs

    hook_handle = model_to_hook.neck.register_forward_hook(neck_hook)

    max_images = args.max_images
    saved = 0

    for i, data in enumerate(data_loader):
        if saved >= max_images:
            break

        captured.clear()

        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)

        if "neck_outs" not in captured:
            continue

        neck_outs = captured["neck_outs"]
        if not isinstance(neck_outs, (tuple, list)):
            raise RuntimeError(f"Unexpected neck outputs type: {type(neck_outs)}")

        if args.level_idx < 0 or args.level_idx >= len(neck_outs):
            raise ValueError(
                f"--level-idx={args.level_idx} out of range, neck outputs length={len(neck_outs)}"
            )

        # Convert input tensor to uint8 RGB-like image (same idea as tools/test.py visualization)
        if len(data["img"]) == 1 and torch.is_tensor(data["img"][0]):
            img_tensor = data["img"][0]
        else:
            # DataContainer path (matches tools/test.py)
            img_tensor = data["img"][0].data[0]

        img_metas = data["img_metas"][0].data[0]
        img_meta = img_metas[0]

        imgs = tensor2imgs(img_tensor, **img_meta["img_norm_cfg"])
        img_show = imgs[0]
        h, w, _ = img_meta["img_shape"]
        img_show = img_show[:h, :w, :]

        ori_filename = img_meta["ori_filename"]
        stem = Path(ori_filename).stem
        if target_stems is not None:
            if stem not in target_stems:
                continue
            target_stems.remove(stem)

        ori_h, ori_w = img_meta["ori_shape"][:-1]
        # Resize to original resolution for overlay
        img_show_pil = Image.fromarray(img_show)
        img_show_u8 = np.array(
            img_show_pil.resize((ori_w, ori_h), resample=Image.BILINEAR)
        )

        # Feature heatmap: take the first sample in batch dimension
        fmap = neck_outs[args.level_idx][0]  # [C,Hf,Wf]
        heat_rgb_u8 = compute_heatmap_rgb(fmap)
        heat_pil = Image.fromarray(heat_rgb_u8)
        heat_resized = heat_pil.resize((ori_w, ori_h), resample=Image.BILINEAR)

        heat_path = out_dir / f"{stem}_L{args.level_idx}_heat.png"
        overlay_path = out_dir / f"{stem}_L{args.level_idx}_overlay.png"

        heat_resized.save(heat_path)

        base_pil = Image.fromarray(img_show_u8)
        overlay_pil = Image.blend(base_pil, heat_resized.convert("RGB"), args.alpha)
        overlay_pil.save(overlay_path)

        # Optional compose with pre-rendered detection visualization
        if compose_dir is not None and args.det_dir is not None:
            det_path = Path(args.det_dir) / ori_filename
            if det_path.exists():
                det_img = Image.open(det_path).convert("RGB")
                heat_overlay = Image.open(overlay_path).convert("RGB").resize(det_img.size, resample=Image.BILINEAR)
                combined = Image.new("RGB", (heat_overlay.width + det_img.width, heat_overlay.height))
                combined.paste(heat_overlay, (0, 0))
                combined.paste(det_img, (heat_overlay.width, 0))
                combined_out = compose_dir / f"{stem}_L{args.level_idx}_heat_det.png"
                combined.save(combined_out)

        saved += 1
        if (saved % 10) == 0:
            print(f"[{saved}/{max_images}] saved to: {out_dir}")
        if target_stems is not None and len(target_stems) == 0:
            break

    hook_handle.remove()
    print(f"Done. Saved {saved} images. Output: {out_dir}")


if __name__ == "__main__":
    main()

