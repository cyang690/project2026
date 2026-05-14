# Video Super-Resolution: From Classical Baselines to SOTA Alignment and Generative Priors

## 1. Project Overview

This project addresses the video super-resolution (VSR) task: reconstructing high-resolution video sequences from low-resolution inputs while preserving spatial fidelity and temporal coherence. The experimental workflow comprises three phases:

1. **Phase I**: Establishing baselines using traditional interpolation methods and early deep learning approaches
2. **Phase II**: Reproducing state-of-the-art modern VSR algorithms
3. **Phase III**: Analyzing limitations of existing methods and proposing enhancements for temporal consistency


## 2. Directory Structure

```text
vsr_project_2026/
  part1/
    sr_baseline.py
    utils.py
  part2/
    part2_basicvsr++_reds.py
    part2_basicvsr++_vimeo90k.py
    part2_realesrgan_wild.py
  part3/
    temporal_hybrid_vsr.py
  tests/
    test_part3_temporal_hybrid.py
    run_part3_checks.py
  checkpoints/
  BasicVSR_PlusPlus/
  Real-ESRGAN/
  results/
```

**Module Responsibilities:**

- **part1/sr_baseline.py**: Implements Bicubic, Lanczos, SRCNN, and sliding-window temporal averaging
- **part1/utils.py**: Provides image preprocessing, post-processing, and metric calculation utilities
- **part2/part2_basicvsr++_reds.py**: Executes BasicVSR++ on REDS validation set
- **part2/part2_basicvsr++_vimeo90k.py**: Executes BasicVSR++ on Vimeo-90K video clips
- **part2/part2_realesrgan_wild.py**: Applies Real-ESRGAN to real-world wild videos
- **part3/temporal_hybrid_vsr.py**: Enhances temporal consistency via uncertainty estimation and motion compensation

## 3. Environment Setup

**Python Environment:** Pre-configured server environment accessible via `python`

**Core Dependencies:**
- Python 3.8
- PyTorch
- OpenCV
- NumPy
- scikit-image
- LPIPS
- MMCV / MMEditing
- BasicSR / Real-ESRGAN
- ffmpeg


> **Note:** Use the pre-configured project environment rather than system-default Python for optimal compatibility.

## 4. Datasets and Checkpoints

### 4.1 Dataset Structure

**Default Location:** `./datasets`


```text
datasets/
  reds/
    val_sharp_bicubic/val/val_sharp_bicubic/X4/
    val_sharp/val/val_sharp/
  vimeo90k/
    vimeo90k/sequences/
    vimeo90k/target/
    vimeo_super_resolution_test/
  wild_video/
    wild_video.mp4
```

### 4.2 Checkpoint Locations

```text
checkpoints/
  srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth
  basicvsr_plusplus_reds4.pth
  basicvsr_plusplus_vimeo90k_bi.pth
  basicvsr_reds4_20120409-0e599677.pth
  basicvsr_vimeo90k_bi_20210409-d2d8f760.pth
  spynet_20210409-c6c1bd09.pth
```

**Real-ESRGAN Weights:** `Real-ESRGAN/weights/realesrgan-x4plus.pth`

> **Note:** If Real-ESRGAN weights are missing, `part2/part2_realesrgan_wild.py` will auto-download them. For offline execution, place weights manually.

## 5. Phase I: Baseline Methods

### 5.1 Methodology

Three baseline categories are implemented:
- **Interpolation-based**: Bicubic, Lanczos
- **Deep learning-based**: SRCNN
- **Temporal fusion**: Sliding-window temporal averaging with unsharp masking (non-SRCNN branches)

This phase establishes performance lower bounds for traditional spatial upsampling and simple temporal fusion strategies.

### 5.2 Implementation

**Code Location:**

```text
part1/sr_baseline.py
part1/utils.py
```

**Key Configurations** (defined in `part1/sr_baseline.py`):


```python
SCALE = 4
WINDOW_SIZE = 3
SAVE_DIR = "../results/part1_videos"
```

To modify dataset paths, update the following functions:
- `process_reds()`
- `process_vimeo()`
- `process_wild_video()`

### 5.3 Execution

```bash
cd ./part1
python sr_baseline.py
```

**Output Directory:** `results/part1_videos/`

```text
results/part1_videos/
  reds_bicubic.avi
  reds_lanczos.avi
  reds_srcnn.avi
  vimeo_bicubic.avi
  vimeo_lanczos.avi
  vimeo_srcnn.avi
  wild_bicubic.avi
  wild_lanczos.avi
  wild_srcnn.avi
```

## 6. Phase II: Modern VSR Methods

### 6.1 Overview

Two technical routes are employed:
1. **BasicVSR++**: Deployed on REDS and Vimeo-90K datasets for explicit temporal propagation and feature alignment
2. **Real-ESRGAN**: Applied to real-world wild videos for perceptual enhancement exploration

---

### 6.2 BasicVSR++ on REDS

**Script:** `part2/part2_basicvsr++_reds.py`

**Configurations:**

```python
PROJ_ROOT = "./"
CKPT_REDS = os.path.join(CKPT_DIR, "basicvsr_plusplus_reds4.pth")
CONFIG_REDS = os.path.join(BASICVSR_ROOT, "configs/basicvsr_plusplus_reds4.py")
REDS_LR_ROOT = os.path.join(DATASET_DIR, "reds/val_sharp_bicubic/val/val_sharp_bicubic/X4")
REDS_GT_ROOT = os.path.join(DATASET_DIR, "reds/val_sharp/val/val_sharp")
OUT_REDS = os.path.join(OUT_ROOT, "reds_basicvsrpp")
```

**Execution:**

```bash
cd ./
python part2/part2_basicvsr++_reds.py
```

**Output:** `results/reds_basicvsrpp/`

```text
results/reds_basicvsrpp/
  000/
  000.mp4
  ...
```

---

### 6.3 BasicVSR++ on Vimeo-90K

**Script:** `part2/part2_basicvsr++_vimeo90k.py`

**Configurations:**


```python
PROJ_ROOT = "./"
VIMEO_ROOT = os.path.join(DATASET_DIR, "vimeo90k", "vimeo90k")
VIMEO_SEQ_ROOT = os.path.join(VIMEO_ROOT, "sequences")
VIMEO_TGT_ROOT = os.path.join(VIMEO_ROOT, "target")
CKPT_VIMEO = os.path.join(CKPT_DIR, "basicvsr_plusplus_vimeo90k_bi.pth")
CONFIG_VIMEO = os.path.join(BASICVSR_ROOT, "configs/basicvsr_plusplus_vimeo90k_bi.py")
OUT_BASE = os.path.join(OUT_ROOT, "vimeo_basicvsrpp")
```

**Execution:**


```bash
cd ./
python part2/part2_basicvsr++_vimeo90k.py
```
**Output:** `results/vimeo_basicvsrpp/`

```text
results/vimeo_basicvsrpp/
  images/
  videos/
```
---

### 6.4 Real-ESRGAN on Wild Videos

**Script:** `part2/part2_realesrgan_wild.py`

**Configurations:**


```python
PROJ_ROOT = "./"
WILD_VIDEO = os.path.join(PROJ_ROOT, "datasets/wild_video/wild_video.mp4")
OUT_VIDEO = os.path.join(PROJ_ROOT, "results/wild_video_RealESRGAN_x4.mp4")
TEMP_FRAMES_DIR = os.path.join(PROJ_ROOT, "results/temp_frames_input")
OUT_FRAMES_DIR = os.path.join(PROJ_ROOT, "results/temp_frames_output")
MODEL_PATH = os.path.join(PROJ_ROOT, "Real-ESRGAN/weights/realesrgan-x4plus.pth")
```
**Execution:**


```bash
cd ./
python part2/part2_realesrgan_wild.py
```
**Output Files:**

```text
results/wild_video_RealESRGAN_x4.mp4
results/temp_frames_input/
results/temp_frames_output/
```

## 7. Phase III: Adaptive Hybrid Enhancement with Temporal Consistency

### 7.1 Motivation

Analysis of Phase I and II outputs reveals inherent limitations:
- **Phase I methods**: Lack fine-grained details; temporal averaging causes motion blur
- **BasicVSR++**: Achieves stable temporal coherence but produces conservative textures
- **Real-ESRGAN**: Generates strong single-frame textures but suffers from severe frame flickering due to insufficient inter-frame constraints

### 7.2 Core Implementation

**Main Script:** `part3/temporal_hybrid_vsr.py`

**Operating Modes:**
- `--mode analyze`: Analyze Phase I/II outputs and generate defect statistical reports
- `--mode run`: Execute the proposed hybrid enhancement algorithm

**Command-Line Arguments:**


```text
--project-root          Specify project root directory, default: /data/cyang690/vsr_project_2026
--out-dir               Custom output directory, default: results/part3_temporal_hybrid
--mode                  analyze, run, all
--dataset               reds, vimeo, wild, all
--seq                   Sequence ID for REDS or clip name for Vimeo
--max-frames            Maximum frames to process in single run
--fps                   Frame rate of output videos
--temporal-strength     Fusion weight of motion compensation temporal constraints
--max-detail-alpha      Maximum weight of local detail enhancement
```
**Hyperparameters** (defined in `TemporalHybridConfig`):

```python
sharpen_amount
clahe_clip
motion_sigma
edge_sigma
detail_delta_sigma
min_texture_gate
```
### 7.3 Execution Examples

**Defect Analysis (Phase I & II):**

```bash
cd ./
python part3/temporal_hybrid_vsr.py --mode analyze --max-frames 24
```
**REDS Dataset (Sequence 000):**

```bash
cd ./
python part3/temporal_hybrid_vsr.py --mode run --dataset reds --seq 000 --max-frames 24 --temporal-strength 0.72 --max-detail-alpha 0.32
```
**Vimeo-90K Dataset:**

```bash
cd ./
python part3/temporal_hybrid_vsr.py --mode run --dataset vimeo --max-frames 7 --temporal-strength 0.70 --max-detail-alpha 0.28
```
**Wild Videos:**

```bash
cd ./
python part3/temporal_hybrid_vsr.py --mode run --dataset wild --max-frames 16 --temporal-strength 0.72 --max-detail-alpha 0.22
```

### 7.4 Output Structure


```text
results/part3_temporal_hybrid/
  part12_limitations.csv
  part12_limitations.md
  reds/reds_000/
    frame_stats.csv
    metrics.json
    frames/
    reds_000_part3_temporal_hybrid.mp4
  vimeo/vimeo_00001_0266/
    frame_stats.csv
    metrics.json
    frames/
    vimeo_00001_0266_part3_temporal_hybrid.mp4
  wild/wild_realesrgan/
    frame_stats.csv
    metrics.json
    frames/
    wild_realesrgan_part3_temporal_hybrid.mp4
```

## 8. Configuration Guidelines

### 8.1 Project Root Directory

Fixed root paths are hard-coded in Phase II scripts:

```python
PROJ_ROOT = "/data/cyang690/vsr_project_2026"
```
**Files requiring updates after path migration:**

```text
part2/part2_basicvsr++_reds.py
part2/part2_basicvsr++_vimeo90k.py
part2/part2_realesrgan_wild.py
```
**Note:** `part3/temporal_hybrid_vsr.py` supports runtime path overriding:

```bash
python part3/temporal_hybrid_vsr.py --project-root /new/project/path --mode analyze
```

### 8.2 BasicVSR++ Model Configurations

Official configuration files:


```text
BasicVSR_PlusPlus/configs/basicvsr_plusplus_reds4.py
BasicVSR_PlusPlus/configs/basicvsr_plusplus_vimeo90k_bi.py
```
Modifiable parameters include network architecture, inference window size, data pipeline, and pretrained weight paths. When replacing checkpoint files only, update `CKPT_REDS` or `CKPT_VIMEO` variables in corresponding scripts.

### 8.3 Output Directories

**Phase I:** Controlled by `SAVE_DIR`


```python
SAVE_DIR = "../results/part1_videos"
```
**Phase II:** Defined by global variables at script tops

```python
OUT_REDS
OUT_BASE
OUT_IMAGES
OUT_VIDEOS
OUT_VIDEO
TEMP_FRAMES_DIR
OUT_FRAMES_DIR
```

**Phase III:** Customize via `--out-dir` argument

```bash
python part3/temporal_hybrid_vsr.py --mode run --dataset wild --out-dir results/part3_custom
```

## 9. Experimental Results and Evaluation Metrics

All experimental results are stored under `results/`:


```text
results/
  part1_videos/
  reds_basicvsrpp/
  vimeo_basicvsrpp/
  wild_video_RealESRGAN_x4.mp4
  temp_frames_input/
  temp_frames_output/
  part3_temporal_hybrid/
```

**Result Characteristics:**
- **Phase I**: Visual comparison among interpolation methods and SRCNN with temporal fusion
- **Phase II**: Demonstrates temporal propagation capability (BasicVSR++) and perceptual enhancement (Real-ESRGAN)
- **Phase III**: Contains defect analysis reports, frame-wise statistics, quantitative metrics, enhanced videos, and intermediate frames

### 9.1 Phase I Quantitative Results

**REDS Dataset (Sequence 000)**

| Method                     | PSNR↑ | SSIM↑  | LPIPS↓ | tLPIPS↓ |
|----------------------------|-------|--------|--------|---------|
| Bicubic + Temporal Avg.    | 19.63 | 0.5040 | 0.5247 | 0.3253  |
| Lanczos + Temporal Avg.    | 19.64 | 0.5055 | 0.5252 | 0.3246  |
| SRCNN + Temporal Avg.      | 19.66 | 0.5119 | 0.5131 | 0.3089  |

**Vimeo-90K Dataset**

| Method   | PSNR↑ | SSIM↑  | LPIPS↓ | tLPIPS↓ |
|----------|-------|--------|--------|---------|
| Bicubic  | 28.83 | 0.8793 | 0.2915 | 0.0520  |
| Lanczos  | 29.03 | 0.8822 | 0.2987 | 0.0524  |
| SRCNN    | 29.91 | 0.9009 | 0.2804 | 0.0390  |

### 9.2 Phase II Quantitative Results

| Dataset   | PSNR↑ | SSIM↑  | LPIPS↓ |
|-----------|-------|--------|--------|
| REDS      | 30.64 | 0.8746 | 0.1603 |
| Vimeo-90K | 35.77 | 0.9450 | 0.0649 |


### 9.3 Phase III Quantitative Results

**Temporal Consistency Improvements**

| Dataset | Base warp error | Part 3 warp error | Base edge flicker | Part 3 edge flicker |
|---|---:|---:|---:|---:|
| Vimeo `00001_0266` | 2.187 | 1.394 | 0.0213 | 0.0170 |
| Wild video | 1.766 | 1.596 | 0.0807 | 0.0751 |
| REDS `000` | 4.761 | 4.874 | 0.2562 | 0.2563 |

For the 24-frame REDS `000` sample, the proposed Part 3 method raises Laplacian sharpness from  `265.04` to `267.63`, with ground-truth referenced metrics reaching PSNR=30.09 dB and SSIM=0.8508.

## 10. Unit Tests and Validation

Independent test scripts verify Phase III core logic:

**Quick Validation:**

```bash
cd ./
python tests/run_part3_checks.py
```
**Pytest-based Unit Tests:**


```bash
python -m pytest tests/test_part3_temporal_hybrid.py -q
```

**Test Coverage:**
- ✓ Detail injection suppression in high-motion regions
- ✓ Temporal fusion with motion compensation reduces frame flickering
- ✓ Downsampling before optical flow computation for high-resolution videos

## 11. Recommended Execution Order

Execute experiments sequentially:


```bash
cd vsr_project_2026
cd vsr_project_2026

# Phase I: Baseline Methods
cd part1
python sr_baseline.py
cd ..

# Phase II: Modern VSR Methods
python part2/part2_basicvsr++_reds.py
python part2/part2_basicvsr++_vimeo90k.py
python part2/part2_realesrgan_wild.py

# Phase III: Temporal Enhancement
python part3/temporal_hybrid_vsr.py --mode analyze --max-frames 24
python part3/temporal_hybrid_vsr.py --mode run --dataset vimeo --max-frames 7 --temporal-strength 0.70 --max-detail-alpha 0.28
python part3/temporal_hybrid_vsr.py --mode run --dataset wild --max-frames 16 --temporal-strength 0.72 --max-detail-alpha 0.22
python part3/temporal_hybrid_vsr.py --mode run --dataset reds --seq 000 --max-frames 24 --temporal-strength 0.72 --max-detail-alpha 0.32

```
> **Storage Note:** Full-scale datasets and complete intermediate frame files occupy substantial storage space and are not distributed with source codes. Organize datasets strictly following Section 4 specifications for full experiment reproduction.
