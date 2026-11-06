import os
import cv2
import torch
import numpy as np
from mmdet.apis import init_detector
import matplotlib.pyplot as plt

# --------------------------
# 配置（改成你的）
# --------------------------
CONFIG = "./configs/UAV/dynamic_retinanet_res18_visdrone_guided.py"
CKPT   = "./work_dirs/dynamic_retinanet_res18_visdrone_guided/latest.pth"
IMG    = "demo/test.jpg"
OUTDIR = "work_dirs/fpn_feat_vis"
DEVICE = "cuda:0"


os.makedirs(OUTDIR, exist_ok=True)


def to_uint8_heatmap(arr):
    """生成彩色热力图(BGR格式)"""
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    heat = cv2.applyColorMap(arr, cv2.COLORMAP_JET)  # BGR
    return heat


def overlay_heat_on_img(img, heat, alpha=0.5):
    """heat(BGR) overlay 到 img(BGR)"""
    heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
    out = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    return out


def extract_fpn_feats(model, img_path):
    """返回 P3~P7 的 feature maps"""
    ori = cv2.imread(img_path)

    # MMDetection transform to tensor
    data = model.data_preprocessor({'img': [ori]})
    img_tensor = data['inputs']

    # Backbone + GuidedFPN
    with torch.no_grad():
        feats = model.extract_feat(img_tensor)

    return ori, feats


def main():
    model = init_detector(CONFIG, CKPT, device=DEVICE)

    ori, feats = extract_fpn_feats(model, IMG)

    fpn_names = ["P3", "P4", "P5", "P6", "P7"]

    for i, feat in enumerate(feats):
        fmap = feat[0].detach().cpu().numpy()  # C,H,W
        fmap_avg = fmap.mean(0)  # H,W

        heat = to_uint8_heatmap(fmap_avg)
        overlay = overlay_heat_on_img(ori, heat)

        # 保存 2 张图：heatmap + overlay
        cv2.imwrite(os.path.join(OUTDIR, f"{fpn_names[i]}_heat.png"), heat)
        cv2.imwrite(os.path.join(OUTDIR, f"{fpn_names[i]}_overlay.png"), overlay)

        print(f"[Saved] {fpn_names[i]}")

    print("All feature maps saved in:", OUTDIR)


if __name__ == "__main__":
    main()
