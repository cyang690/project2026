# ===================== SRCNN Model Definition =====================
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import lpips

class SRCNN(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# ===================== Interpolation-based Upsampling =====================
def interpolation_upscale(img, scale=4, mode='bicubic'):
    h, w = img.shape[:2]
    new_h, new_w = h * scale, w * scale
    if mode == 'bicubic':
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif mode == 'lanczos':
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        raise ValueError("mode must be 'bicubic' or 'lanczos'")

# ===================== Sliding Window Temporal Averaging =====================
def temporal_averaging_sliding(frames, window_size=3, use_unsharp=True):
    half = window_size // 2
    padded = [frames[0]] * half + frames + [frames[-1]] * half
    result = []
    for i in range(len(frames)):
        window = padded[i : i + window_size]
        weights = np.linspace(0.5, 1.0, window_size)[::-1]
        weights /= weights.sum()
        avg = np.zeros_like(window[0], dtype=np.float32)
        for w, f in zip(weights, window):
            avg += w * f.astype(np.float32)
        avg = avg.astype(np.uint8)
        if use_unsharp:
            avg = unsharp_masking(avg)
        result.append(avg)
    return result

# ===================== SRCNN Inference =====================
def infer_srcnn(img, model):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        lr_tensor = torch.from_numpy(preprocess(img)).unsqueeze(0).float().to(device)
        sr_tensor = model(lr_tensor).squeeze(0).cpu().numpy()
    return postprocess(sr_tensor)

# ===================== Temporal LPIPS Calculation =====================
def compute_tlpips(frames_list, lpips_model, device):
    if len(frames_list) < 2:
        return 0.0
    total = 0.0
    with torch.no_grad():
        for i in range(len(frames_list) - 1):
            t1 = lpips.im2tensor(frames_list[i]).to(device)
            t2 = lpips.im2tensor(frames_list[i+1]).to(device)
            total += lpips_model(t1, t2).item()
    return total / (len(frames_list) - 1)

# ===================== REDS Dataset Processing =====================
def process_reds(model, method, scale, output_dir, window_size=3):
    print(f"\nProcessing REDS dataset with method: {method}")
    seq = "000"
    lr_dir = f"../datasets/reds/val_sharp_bicubic/val/val_sharp_bicubic/X4/{seq}"
    hr_dir = f"../datasets/reds/val_sharp/val/val_sharp/{seq}"
    if not os.path.exists(lr_dir):
        print(f"REDS directory not found: {lr_dir}")
        return
    lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
    if not lr_files:
        print("No image files found in REDS dataset")
        return
    lr_frames = [cv2.imread(os.path.join(lr_dir, f)) for f in lr_files]
    hr_frames = [cv2.imread(os.path.join(hr_dir, f)) for f in lr_files] if os.path.exists(hr_dir) else None

    # Super-resolution reconstruction
    if method == 'bicubic':
        sr_frames = [interpolation_upscale(f, scale, 'bicubic') for f in lr_frames]
    elif method == 'lanczos':
        sr_frames = [interpolation_upscale(f, scale, 'lanczos') for f in lr_frames]
    elif method == 'srcnn':
        sr_frames = [infer_srcnn(f, model) for f in lr_frames]
    else:
        raise ValueError("Unknown super-resolution method")

    sr_avg = temporal_averaging_sliding(sr_frames, window_size, use_unsharp=(method != 'srcnn'))
    h, w = sr_avg[0].shape[:2]
    out_path = os.path.join(output_dir, f"reds_{method}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 30, (w, h))
    for f in sr_avg:
        out.write(f)
    out.release()
    print(f"Video saved to: {out_path}")

    if hr_frames:
        mid = len(hr_frames) // 2
        psnr_val, ssim_val, lpips_val = calculate_metrics(hr_frames[mid], sr_avg[mid])
        tlpips_val = compute_tlpips(sr_avg, lpips_model, device)
        print(f"  PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | tLPIPS: {tlpips_val:.4f}")

# ===================== Vimeo-90K Dataset Processing =====================
def process_vimeo(model, method, scale, output_dir, window_size=3):
    print(f"\nProcessing Vimeo-90K dataset with method: {method}")
    lr_root = "../datasets/vimeo90k/vimeo_super_resolution_test/low_resolution"
    hr_root = "../datasets/vimeo90k/vimeo_super_resolution_test/target"
    test_list = "../datasets/vimeo90k/vimeo_super_resolution_test/sep_testlist.txt"
    if not os.path.exists(lr_root):
        print(f"Vimeo directory not found: {lr_root}")
        return
    with open(test_list, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    seq = sequences[0]
    lr_dir = os.path.join(lr_root, seq)
    hr_dir = os.path.join(hr_root, seq)
    if not os.path.exists(lr_dir):
        print(f"Sequence directory not found: {lr_dir}")
        return
    lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
    lr_frames = [cv2.imread(os.path.join(lr_dir, f)) for f in lr_files]

    # Super-resolution reconstruction
    if method == 'bicubic':
        sr_frames = [interpolation_upscale(f, scale, 'bicubic') for f in lr_frames]
    elif method == 'lanczos':
        sr_frames = [interpolation_upscale(f, scale, 'lanczos') for f in lr_frames]
    elif method == 'srcnn':
        sr_frames = [infer_srcnn(f, model) for f in lr_frames]
    else:
        raise ValueError("Unknown super-resolution method")

    sr_avg = temporal_averaging_sliding(sr_frames, window_size, use_unsharp=(method != 'srcnn'))
    h, w = sr_avg[0].shape[:2]
    out_path = os.path.join(output_dir, f"vimeo_{method}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 30, (w, h))
    for f in sr_avg:
        out.write(f)
    out.release()
    print(f"Video saved to: {out_path}")

    # Evaluation: Only middle frame has ground truth in Vimeo test set
    if os.path.exists(hr_dir):
        hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        if len(hr_files) == 1:
            hr_mid = cv2.imread(os.path.join(hr_dir, hr_files[0]))
            if len(sr_avg) >= 4:
                sr_mid = sr_avg[3]
            else:
                sr_mid = sr_avg[-1]
            psnr_val, ssim_val, lpips_val = calculate_metrics(hr_mid, sr_mid)
            tlpips_val = compute_tlpips(sr_avg, lpips_model, device)
            print(f"  PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | tLPIPS: {tlpips_val:.4f}")
        elif len(hr_files) == len(lr_files):
            hr_frames = [cv2.imread(os.path.join(hr_dir, f)) for f in hr_files]
            mid = len(hr_frames) // 2
            psnr_val, ssim_val, lpips_val = calculate_metrics(hr_frames[mid], sr_avg[mid])
            tlpips_val = compute_tlpips(sr_avg, lpips_model, device)
            print(f"  PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | tLPIPS: {tlpips_val:.4f}")
        else:
            print("  Mismatched GT frames, skipping quantitative evaluation")
    else:
        print("  No Ground Truth available, skipping quantitative evaluation")

# ===================== Real-world Wild Video Processing =====================
def process_wild_video(model, method, scale, output_dir, window_size=3):
    print(f"\nProcessing real-world wild video with method: {method}")
    input_path = "../datasets/wild_video/wild_video.mp4"
    if not os.path.exists(input_path):
        print(f"Input video not found: {input_path}")
        return
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    lr_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lr_frames.append(frame)
    cap.release()
    print(f"Loaded {len(lr_frames)} frames")
    if not lr_frames:
        print("Empty video, skipping processing")
        return

    if method == 'bicubic':
        sr_frames = [interpolation_upscale(f, scale, 'bicubic') for f in lr_frames]
    elif method == 'lanczos':
        sr_frames = [interpolation_upscale(f, scale, 'lanczos') for f in lr_frames]
    elif method == 'srcnn':
        sr_frames = [infer_srcnn(f, model) for f in lr_frames]
    else:
        raise ValueError("Unknown super-resolution method")

    sr_avg = temporal_averaging_sliding(sr_frames, window_size, use_unsharp=(method != 'srcnn'))
    h, w = sr_avg[0].shape[:2]
    out_path = os.path.join(output_dir, f"wild_{method}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in sr_avg:
        out.write(f)
    out.release()
    print(f"Video saved to: {out_path}")

# ===================== Main Function =====================
if __name__ == "__main__":
    SCALE = 4
    WINDOW_SIZE = 3
    SAVE_DIR = "../results/part1_videos"
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN(SCALE).to(device)

    # Model weight path
    weight_path = "../checkpoints/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth"
    if not os.path.exists(weight_path):
        weight_path = "../checkpoints/srcnn_x4_div2k.pth"
    if not os.path.exists(weight_path):
        print("Error: SRCNN pre-trained weights not found, please check the checkpoints directory")
        exit(1)
    print(f"Loading pre-trained weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("generator.", "").replace("params.", "conv").replace("module.", "")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    print("SRCNN weights loaded successfully\n")

    methods = ['bicubic', 'lanczos', 'srcnn']
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    for method in methods:
        cur_model = model if method == 'srcnn' else None
        print("=" * 50)
    
        print(f"Evaluating method: {method.upper()}")
        process_reds(cur_model, method, SCALE, SAVE_DIR, WINDOW_SIZE)
        process_vimeo(cur_model, method, SCALE, SAVE_DIR, WINDOW_SIZE)
        process_wild_video(cur_model, method, SCALE, SAVE_DIR, WINDOW_SIZE)
        print("=" * 50)

    print(f"\n🎉 Part 1 evaluation completed! All output videos are saved in: {SAVE_DIR}")