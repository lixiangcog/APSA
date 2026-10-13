import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import nibabel as nib
import argparse
import json
import timm
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ==========================================
# 1. 模型定义 & 核心逻辑
# ==========================================

class BiomedCLIPWrapper(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        cfg = json.load(open(os.path.join(model_path, "open_clip_config.json"), "r"))

        # 初始化核心模型
        self.model = _BiomedCLIPCore(cfg["model_cfg"]["vision_cfg"]["timm_model_name"],
                                     os.path.join(model_path, "biomedbert"),
                                     cfg["model_cfg"]["embed_dim"])

        sd = torch.load(os.path.join(model_path, "open_clip_pytorch_model.bin"), map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]

        # 过滤并加载权重
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items() if "model." not in k}, strict=False)
        self.model.to(device).eval()

        # 🔥 修复点：强制把 mean/std 放到指定 device 上
        mean = torch.tensor(cfg.get("preprocess_cfg", {})["mean"]).view(1, 3, 1, 1).to(device)
        std  = torch.tensor(cfg.get("preprocess_cfg", {})["std"]).view(1, 3, 1, 1).to(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "biomedbert"))

    def encode_image(self, img):
        # 确保输入图像也在同一设备
        return self.model.encode_image((img.to(self.device) - self.mean) / self.std)

    def encode_text(self, text):
        tok = self.tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(self.device)
        return self.model.encode_text(tok["input_ids"], tok["attention_mask"])

class _BiomedCLIPCore(torch.nn.Module):
    def __init__(self, timm_name, bert_path, embed_dim):
        super().__init__()
        self.visual = torch.nn.Module()
        self.text = torch.nn.Module()
        self.visual.trunk = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool="")
        self.visual.proj = torch.nn.Linear(self.visual.trunk.num_features, embed_dim, bias=False)
        self.text.trunk = AutoModel.from_pretrained(bert_path)
        self.text.proj = torch.nn.Sequential(torch.nn.Linear(768, 640), torch.nn.GELU(), torch.nn.Linear(640, embed_dim))

    def encode_image(self, x):
        feat = self.visual.trunk.forward_features(x)
        return F.normalize(self.visual.proj(feat[:, 0, :] if feat.dim() == 3 else feat), dim=-1)

    def encode_text(self, ids, mask):
        return F.normalize(self.text.proj(self.text.trunk(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]), dim=-1)

# --- 辅助函数 ---
def normalize01(x):
    min_v, max_v = x.min(), x.max()
    if max_v - min_v < 1e-8: return np.zeros_like(x)
    return (x - min_v) / (max_v - min_v)

def get_saliency(model, img_tensor, prompt, device):
    H, W = img_tensor.shape[-2:]
    text_emb = model.encode_text([prompt])
    text_emb = text_emb.mean(dim=0, keepdim=True)
    text_emb = F.normalize(text_emb, dim=-1)

    # 确保 img_tensor 上传到 device
    img_in = F.interpolate(img_tensor.to(device).unsqueeze(0).repeat(1,3,1,1), size=(224,224), mode='bilinear')

    #此时 model.mean 已经在 device 上了，不会报错
    img_in = (img_in - model.mean) / model.std

    feat = model.model.visual.trunk.forward_features(img_in)
    tokens = feat[:, 1:, :]
    patch_emb = F.normalize(model.model.visual.proj(tokens), dim=-1)

    sim = (patch_emb * text_emb.unsqueeze(1)).sum(dim=-1).squeeze()
    sim = sim.detach().cpu().numpy().reshape(14, 14)
    sim = cv2.resize(normalize01(sim), (W, H), interpolation=cv2.INTER_CUBIC)
    return normalize01(sim)

# --- V8 逻辑 (抗噪 + 凝聚度检查) ---
def compute_v8_prior(img, score_map):
    H, W = img.shape
    # 1. 膀胱检测 (带高斯模糊降噪)
    img_blur = cv2.GaussianBlur(img, (15, 15), 0)

    # 2. 依然只看上半部分 + 中间区域
    x_start = int(W * 0.20)
    x_end = int(W * 0.80)
    roi = img_blur[:int(H * 0.6), x_start:x_end]

    # 3. 阈值提取
    thr = np.percentile(roi, 99.5)
    bw = (roi >= thr).astype(np.uint8)

    # 4. 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)

    use_default = True
    cx, cy = 0.5, 0.6 # 默认解剖学位置

    if num > 1:
        max_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_area = stats[max_idx, cv2.CC_STAT_AREA]

        # 如果最大的块不是散点 (大于 0.5% 面积)
        if largest_area > (H * W * 0.005):
            # 还原坐标
            M = cv2.moments((labels == max_idx).astype(np.uint8))
            if M["m00"] > 0:
                bcx = (M["m10"]/M["m00"] + x_start) / W
                bcy = (M["m01"]/M["m00"]) / H

                # Sanity Check: 膀胱必须在上面
                if bcy < 0.6:
                    cx = (bcx * 0.5 + 0.5 * 0.5)
                    cy = min(0.92, bcy + 0.12)
                    use_default = False

    # 生成 Prior Gaussian
    sx, sy = W * 0.18, H * 0.10
    yg, xg = np.mgrid[:H, :W].astype(np.float32)
    prior = np.exp(-((xg - cx*W)**2)/(2*sx**2) - ((yg - cy*H)**2)/(2*sy**2))
    return normalize01(prior) ** 2

# ==========================================
# 2. Main Logic
# ==========================================
def main(args):
    print(f"🔥 Loading model to {args.device}...")
    model = BiomedCLIPWrapper(args.model_path, args.device)

    domains = ["BIDMC", "HK", "UCL", "I2CVB", "RUNMC", "BMC"]
    prompt = "Prostate gland."

    for domain in domains:
        print(f"🚀 Processing {domain}...")
        save_root = os.path.join(args.save_path, domain)
        os.makedirs(save_root, exist_ok=True)

        files = sorted(glob.glob(os.path.join(args.data_path, domain, "*.nii.gz")))
        if not files: files = sorted(glob.glob(os.path.join(args.data_path, domain, "image", "*.nii.gz")))

        for f in tqdm(files):
            case_name = os.path.basename(f).replace(".nii.gz", "")

            # Load NIfTI
            nii = nib.load(f)
            vol = nii.get_fdata()

            # Simple Norm
            vol = (vol - np.percentile(vol, 1)) / (np.percentile(vol, 99) - np.percentile(vol, 1) + 1e-8)
            vol = np.clip(vol, 0, 1)

            # Handle Dimensions (H, W, D) vs (W, H, D)
            # Assuming standard orientation where last dim is slices
            if vol.ndim == 3:
                num_slices = vol.shape[2]

                # Take middle slice
                mid_idx = num_slices // 2
                img_mid = np.rot90(vol[:, :, mid_idx]) # Rotate to match CLIP view

                # CLAHE
                img_u8 = (img_mid * 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_mid = clahe.apply(img_u8).astype(np.float32) / 255.0

                # CLIP Saliency
                raw_saliency = get_saliency(model, torch.from_numpy(img_mid), prompt, args.device)

                # V8 Prior Logic
                prior = compute_v8_prior(img_mid, raw_saliency)

                # Final Soft Map
                soft_map_mid = normalize01(raw_saliency * prior)

                # Restore orientation
                soft_map_mid_restored = np.rot90(soft_map_mid, k=-1)

                # Save
                save_dict = {
                    "soft_map": soft_map_mid_restored.astype(np.float16),
                    "prior_map": np.rot90(prior, k=-1).astype(np.float16),
                    "slice_idx": mid_idx
                }

                np.save(os.path.join(save_root, f"{case_name}_clip.npy"), save_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_path", required=True, help="Local BioMedCLIP snapshot directory.")
    parser.add_argument("--data_path", required=True, help="Root of the prostate MRI volumes.")
    parser.add_argument("--save_path", required=True, help="Output root for fixed prior maps.")
    args = parser.parse_args()
    main(args)
