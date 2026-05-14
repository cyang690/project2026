import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import warnings
warnings.filterwarnings("ignore")  # 彻底屏蔽torchvision警告

# 统一设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化LPIPS（无警告、无device错误）
lpips_model = lpips.LPIPS(net='vgg', verbose=False).to(device)
lpips_model.eval()

# 图像预处理：HWC(BGR)→CHW(RGB)，归一化到[0,1]
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC→CHW
    return img

# 图像后处理：CHW(RGB)→HWC(BGR)，反归一化到[0,255]，彻底修复格式错误
def postprocess(img):
    img = np.transpose(img, (1, 2, 0))  # CHW→HWC
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)  # 强制截断，避免溢出
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# 计算指标：彻底修复尺寸不匹配、类型错误
def calculate_metrics(hr_img, sr_img):
    # 强制对齐尺寸（防止意外尺寸偏差）
    if hr_img.shape != sr_img.shape:
        sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # 强制转uint8
    hr_img = hr_img.astype(np.uint8)
    sr_img = sr_img.astype(np.uint8)
    
    psnr_val = psnr(hr_img, sr_img)
    ssim_val = ssim(hr_img, sr_img, channel_axis=2)
    
    # LPIPS计算：禁用梯度，统一设备
    with torch.no_grad():
        hr_tensor = lpips.im2tensor(hr_img).to(device)
        sr_tensor = lpips.im2tensor(sr_img).to(device)
        lpips_val = lpips_model(hr_tensor, sr_tensor).item()
    
    return psnr_val, ssim_val, lpips_val

# 反锐化掩膜：增强边缘，无副作用
def unsharp_masking(img, sigma=1.0, amount=1.5, threshold=0):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    if threshold > 0:
        mask = cv2.absdiff(img, blur) > threshold
        sharp = np.where(mask, sharp, img)
    return sharp