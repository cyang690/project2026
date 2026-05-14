#!/usr/bin/env python3
# part2_wild.py
# Real-ESRGAN video enhancement for wild_video.mp4

import os
import cv2
import torch
import subprocess
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ========== Configuration ==========
PROJ_ROOT = "/data/cyang690/vsr_project_2026"
WILD_VIDEO = os.path.join(PROJ_ROOT, "datasets/wild_video/wild_video.mp4")
# Output video name includes method name: RealESRGAN_x4
OUT_VIDEO = os.path.join(PROJ_ROOT, "results/wild_video_RealESRGAN_x4.mp4")
TEMP_FRAMES_DIR = os.path.join(PROJ_ROOT, "results/temp_frames_input")
OUT_FRAMES_DIR = os.path.join(PROJ_ROOT, "results/temp_frames_output")
MODEL_PATH = os.path.join(PROJ_ROOT, "Real-ESRGAN/weights/realesrgan-x4plus.pth")

os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
os.makedirs(OUT_FRAMES_DIR, exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading Real-ESRGAN model weights...")
    subprocess.run([
        "wget", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "-O", MODEL_PATH
    ], check=True)

# ========== 1. Extract frames from video ==========
print("1. Extracting frames from video...")
cmd_extract = f"ffmpeg -y -i {WILD_VIDEO} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 {TEMP_FRAMES_DIR}/frame_%08d.png"
subprocess.run(cmd_extract, shell=True, check=True)
frame_paths = sorted([os.path.join(TEMP_FRAMES_DIR, f) for f in os.listdir(TEMP_FRAMES_DIR) if f.endswith('.png')])
print(f"   Extracted {len(frame_paths)} frames")

# ========== 2. Load Real-ESRGAN model ==========
print("2. Loading Real-ESRGAN model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False if device == torch.device('cpu') else True,
    device=device
)
print(f"   Model loaded on device: {device}")

# ========== 3. Enhance each frame ==========
print("3. Enhancing frames (may take a while)...")
for i, img_path in enumerate(tqdm(frame_paths, desc="Processing")):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    output, _ = upsampler.enhance(img, outscale=4)
    out_path = os.path.join(OUT_FRAMES_DIR, f"sr_{i:08d}.png")
    cv2.imwrite(out_path, output)

# ========== 4. Merge frames into video ==========
print("4. Merging enhanced frames into video...")
cmd_merge = f"ffmpeg -y -framerate 25 -i {OUT_FRAMES_DIR}/sr_%08d.png -c:v libx264 -pix_fmt yuv420p -crf 18 {OUT_VIDEO}"
subprocess.run(cmd_merge, shell=True, check=True)

print(f"\n✅ Done! Enhanced video saved to: {OUT_VIDEO}")