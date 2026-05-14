#!/usr/bin/env python3
# part2_vimeo.py
# BasicVSR++ inference on Vimeo-90K (first 10 groups only)

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

VIMEO_ROOT = os.path.join(DATASET_DIR, "vimeo90k", "vimeo90k")
VIMEO_SEQ_ROOT = os.path.join(VIMEO_ROOT, "sequences")
VIMEO_TGT_ROOT = os.path.join(VIMEO_ROOT, "target")

CKPT_VIMEO = os.path.join(CKPT_DIR, "basicvsr_plusplus_vimeo90k_bi.pth")
CONFIG_VIMEO = os.path.join(BASICVSR_ROOT, "configs/basicvsr_plusplus_vimeo90k_bi.py")

OUT_BASE = os.path.join(OUT_ROOT, "vimeo_basicvsrpp")
OUT_IMAGES = os.path.join(OUT_BASE, "images")
OUT_VIDEOS = os.path.join(OUT_BASE, "videos")
os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_VIDEOS, exist_ok=True)

# ========== Helper functions (same as REDS) ==========
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

# ========== Main: Vimeo-90K (first 10 groups) ==========
print("\n" + "="*50)
print("BasicVSR++ on Vimeo-90K (first 10 groups)")
print("Output: images under 'images/', videos under 'videos/'")
print("="*50)

if not os.path.exists(VIMEO_SEQ_ROOT):
    print(f"Error: {VIMEO_SEQ_ROOT} not found")
    exit(1)

all_groups = sorted([d for d in os.listdir(VIMEO_SEQ_ROOT) if os.path.isdir(os.path.join(VIMEO_SEQ_ROOT, d))])
group_dirs = all_groups[:10]
print(f"Total groups: {len(all_groups)}. Processing first {len(group_dirs)} groups: {group_dirs}")

model = init_model(CONFIG_VIMEO, CKPT_VIMEO, device=device)
loss_fn = lpips.LPIPS(net='alex').to(device)

all_psnr, all_ssim, all_lpips = [], [], []

for group in tqdm(group_dirs, desc="Processing groups"):
    group_seq_dir = os.path.join(VIMEO_SEQ_ROOT, group)
    clip_dirs = [d for d in os.listdir(group_seq_dir) if os.path.isdir(os.path.join(group_seq_dir, d))]
    for clip in clip_dirs:
        clip_lq_dir = os.path.join(group_seq_dir, clip)
        lq_paths = sorted(glob(os.path.join(clip_lq_dir, "*.png")))
        if len(lq_paths) != 7:
            print(f"Warning: {group}/{clip} has {len(lq_paths)} images, expected 7. Skipping.")
            continue

        sr_frames = sliding_window_inference(model, lq_paths, window_size=5)
        img_out_dir = os.path.join(OUT_IMAGES, f"{group}_{clip}")
        save_frames(sr_frames, img_out_dir, "{idx:08d}.png")
        video_out_path = os.path.join(OUT_VIDEOS, f"{group}_{clip}.mp4")
        images_to_video(img_out_dir, video_out_path)

        # Find corresponding GT (im4.png) under target folder
        group_tgt_dir = os.path.join(VIMEO_TGT_ROOT, group)
        if not os.path.exists(group_tgt_dir):
            print(f"Warning: Target group {group} not found, skipping GT for {clip}")
            continue
        if clip in os.listdir(group_tgt_dir):
            gt_path = os.path.join(group_tgt_dir, clip, "im4.png")
        else:
            print(f"Warning: GT clip {clip} not found under {group_tgt_dir}, skipping")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: GT file missing for {group}/{clip}")
            continue

        gt = cv2.imread(gt_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        sr_mid = sr_frames[3]  # middle frame of 7
        if sr_mid.shape != gt.shape:
            gt = cv2.resize(gt, (sr_mid.shape[1], sr_mid.shape[0]))

        all_psnr.append(psnr(gt, sr_mid, data_range=255))
        all_ssim.append(ssim(gt, sr_mid, multichannel=True, data_range=255, channel_axis=2))
        sr_t = torch.tensor(sr_mid/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
        gt_t = torch.tensor(gt/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
        all_lpips.append(loss_fn(sr_t, gt_t).item())

if all_psnr:
    print(f"\nVimeo-90K Overall Results (over {len(all_psnr)} clips, first 10 groups):")
    print(f"  PSNR: {np.mean(all_psnr):.2f} dB")
    print(f"  SSIM: {np.mean(all_ssim):.4f}")
    print(f"  LPIPS: {np.mean(all_lpips):.4f}")
else:
    print("No metrics computed. Check your Vimeo-90K structure.")

print(f"\nAll results saved under: {OUT_BASE}")
print("  - Images:  ", OUT_IMAGES)
print("  - Videos:  ", OUT_VIDEOS)