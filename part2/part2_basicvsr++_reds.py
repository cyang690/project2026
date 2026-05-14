#!/usr/bin/env python3
# part2_reds.py
# BasicVSR++ inference on REDS dataset (full validation set)

import os
import cv2
import torch
import numpy as np
import lpips
import subprocess
from glob import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from mmcv import Config
from mmedit.apis import init_model

# ========== Configuration ==========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

PROJ_ROOT = "/data/cyang690/vsr_project_2026"
CKPT_DIR = os.path.join(PROJ_ROOT, "checkpoints")
DATASET_DIR = os.path.join(PROJ_ROOT, "datasets")
BASICVSR_ROOT = os.path.join(PROJ_ROOT, "BasicVSR_PlusPlus")
OUT_ROOT = os.path.join(PROJ_ROOT, "results")
os.makedirs(OUT_ROOT, exist_ok=True)

CKPT_REDS = os.path.join(CKPT_DIR, "basicvsr_plusplus_reds4.pth")
CONFIG_REDS = os.path.join(BASICVSR_ROOT, "configs/basicvsr_plusplus_reds4.py")

REDS_LR_ROOT = os.path.join(DATASET_DIR, "reds/val_sharp_bicubic/val/val_sharp_bicubic/X4")
REDS_GT_ROOT = os.path.join(DATASET_DIR, "reds/val_sharp/val/val_sharp")

OUT_REDS = os.path.join(OUT_ROOT, "reds_basicvsrpp")
os.makedirs(OUT_REDS, exist_ok=True)

# ========== Helper functions ==========
def load_frames(frame_paths):
    frames = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Cannot read {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.stack(frames, axis=0)

def preprocess_frames(frames, device):
    frames = frames.astype(np.float32) / 255.0
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    return tensor

def extract_output_tensor(model_output):
    if isinstance(model_output, dict):
        if 'output' in model_output:
            return model_output['output']
        else:
            return list(model_output.values())[0]
    else:
        return model_output

def sliding_window_inference(model, lq_paths, window_size=5):
    model.eval()
    num_frames = len(lq_paths)
    pad = window_size // 2
    padded_paths = [lq_paths[0]] * pad + lq_paths + [lq_paths[-1]] * pad
    outputs = []
    for i in range(num_frames):
        window_paths = padded_paths[i:i+window_size]
        window_frames = load_frames(window_paths)
        input_tensor = preprocess_frames(window_frames, device)
        with torch.no_grad():
            raw_output = model(lq=input_tensor, test_mode=True)
            output_tensor = extract_output_tensor(raw_output)
        middle = output_tensor[0, pad].cpu().permute(1,2,0).numpy() * 255
        outputs.append(middle.clip(0,255).astype(np.uint8))
    return outputs

def save_frames(frames, out_dir, basename_pattern="{idx:08d}.png"):
    os.makedirs(out_dir, exist_ok=True)
    for idx, img in enumerate(frames):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        path = os.path.join(out_dir, basename_pattern.format(idx=idx))
        cv2.imwrite(path, img_bgr)

def images_to_video(img_dir, video_path, fps=25):
    img_list = sorted(glob(os.path.join(img_dir, "*.png")))
    if not img_list:
        return
    cmd = f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{img_dir}/*.png' -c:v libx264 -pix_fmt yuv420p {video_path}"
    subprocess.run(cmd, shell=True, check=True)

# ========== Main: REDS ==========
print("\n" + "="*50)
print("BasicVSR++ on REDS dataset")
print("="*50)

model = init_model(CONFIG_REDS, CKPT_REDS, device=device)
seqs = sorted([d for d in os.listdir(REDS_LR_ROOT) if os.path.isdir(os.path.join(REDS_LR_ROOT, d))])
print(f"Found {len(seqs)} sequences: {seqs}")

all_psnr, all_ssim, all_lpips = [], [], []
loss_fn = lpips.LPIPS(net='alex').to(device)

for seq in tqdm(seqs, desc="REDS inference"):
    lr_seq_dir = os.path.join(REDS_LR_ROOT, seq)
    lq_paths = sorted(glob(os.path.join(lr_seq_dir, "*.png")))
    if not lq_paths:
        continue
    sr_frames = sliding_window_inference(model, lq_paths, window_size=5)
    out_seq_dir = os.path.join(OUT_REDS, seq)
    save_frames(sr_frames, out_seq_dir, "{idx:08d}.png")
    images_to_video(out_seq_dir, os.path.join(OUT_REDS, f"{seq}.mp4"))

    # Compute metrics against GT
    gt_seq_dir = os.path.join(REDS_GT_ROOT, seq)
    gt_paths = sorted(glob(os.path.join(gt_seq_dir, "*.png")))
    if len(gt_paths) == len(sr_frames):
        for sr_frame, gt_path in zip(sr_frames, gt_paths):
            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            if sr_frame.shape != gt.shape:
                gt = cv2.resize(gt, (sr_frame.shape[1], sr_frame.shape[0]))
            all_psnr.append(psnr(gt, sr_frame, data_range=255))
            all_ssim.append(ssim(gt, sr_frame, multichannel=True, data_range=255, channel_axis=2))
            sr_t = torch.tensor(sr_frame/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
            gt_t = torch.tensor(gt/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
            all_lpips.append(loss_fn(sr_t, gt_t).item())

if all_psnr:
    print(f"\nREDS Overall Results:")
    print(f"  PSNR: {np.mean(all_psnr):.2f} dB")
    print(f"  SSIM: {np.mean(all_ssim):.4f}")
    print(f"  LPIPS: {np.mean(all_lpips):.4f}")
else:
    print("No metrics computed. Check GT alignment.")