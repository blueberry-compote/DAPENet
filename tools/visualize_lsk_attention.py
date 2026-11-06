import cv2
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector

def save_heatmap(tensor, save_path):
    t = tensor.squeeze().cpu().numpy()
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    t = (t * 255).astype(np.uint8)
    heat = cv2.applyColorMap(t, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heat)

def main():
    config = "./configs/UAV/dynamic_retinanet_res18_visdrone_guided.py"
    checkpoint = "./work_dirs/dynamic_retinanet_res18_visdrone_guided/epoch_15.pth"
    img_path = "test.jpg"

    model = init_detector(config, checkpoint, device="cuda:0")

    # 运行一次推理
    result = inference_detector(model, img_path)

    # 遍历所有 LSK 模块
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, GuidedFPN):
            if hasattr(module, "_last_att_map"):
                save_heatmap(module._last_att_map[0], f"lsk_att_{idx}.png")
            if hasattr(module, "_last_guided"):
                save_heatmap(module._last_guided[0].mean(0), f"lsk_guided_{idx}.png")
            idx += 1

if __name__ == "__main__":
    main()
