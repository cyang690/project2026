#!/usr/bin/env python3
"""
Part 3: uncertainty-aware temporal hybrid refinement.

The script is deliberately tied to the Part 1/Part 2 outputs in this project:
it first measures where the earlier methods lose detail or flicker, then refines
the Part 2 frames with an adaptive detail branch and motion-compensated temporal
smoothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass
class TemporalHybridConfig:
    max_detail_alpha: float = 0.58
    temporal_strength: float = 0.62
    sharpen_amount: float = 0.72
    clahe_clip: float = 1.45
    motion_sigma: float = 4.0
    edge_sigma: float = 22.0
    detail_delta_sigma: float = 28.0
    min_texture_gate: float = 0.12
    max_frames: int = 80
    fps: int = 25
    flow_max_side: int = 480


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame, 0, 255).astype(np.uint8)


def _normalize_map(value: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> np.ndarray:
    value = value.astype(np.float32)
    lo, hi = np.percentile(value, [p_low, p_high])
    if hi - lo < 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _flow_cur_to_prev(cur_gray: np.ndarray, prev_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        cur_gray.astype(np.uint8),
        prev_gray.astype(np.uint8),
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def _resize_gray_for_flow(cur_gray: np.ndarray, prev_gray: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray, float]:
    if max_side <= 0:
        return cur_gray, prev_gray, 1.0
    h, w = cur_gray.shape
    side = max(h, w)
    if side <= max_side:
        return cur_gray, prev_gray, 1.0
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    cur_small = cv2.resize(cur_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    prev_small = cv2.resize(prev_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cur_small, prev_small, scale


def _remap_with_flow(prev_frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def warp_previous_to_current(
    prev_frame: np.ndarray,
    cur_base: np.ndarray,
    prev_base: np.ndarray,
    flow_max_side: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    cur_gray = _gray(cur_base)
    prev_gray = _gray(prev_base)
    cur_flow_gray, prev_flow_gray, scale = _resize_gray_for_flow(cur_gray, prev_gray, flow_max_side)
    flow = _flow_cur_to_prev(cur_flow_gray, prev_flow_gray)

    if scale != 1.0:
        h, w = cur_gray.shape
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] /= scale
        flow[..., 1] /= scale

    warped = _remap_with_flow(prev_frame, flow)
    return warped, flow


def build_detail_candidate(base_frame: np.ndarray, config: TemporalHybridConfig) -> np.ndarray:
    lab = cv2.cvtColor(base_frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    contrast = cv2.cvtColor(cv2.merge([l_enhanced, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    base_f = base_frame.astype(np.float32)
    blur = cv2.GaussianBlur(base_f, (0, 0), sigmaX=1.05)
    high = base_f - blur
    sharp = base_f + config.sharpen_amount * high
    detail = 0.68 * sharp + 0.32 * contrast.astype(np.float32)
    return _to_uint8(detail)


def build_adaptive_detail_mask(
    base_frame: np.ndarray,
    detail_frame: np.ndarray,
    prev_base_frame: Optional[np.ndarray],
    config: TemporalHybridConfig,
) -> np.ndarray:
    base_gray = _gray(base_frame)
    texture = np.abs(cv2.Laplacian(base_gray, cv2.CV_32F, ksize=3))
    texture_gate = _normalize_map(texture)
    texture_gate = np.clip((texture_gate - config.min_texture_gate) / (1.0 - config.min_texture_gate), 0.0, 1.0)

    detail_delta = np.mean(np.abs(detail_frame.astype(np.float32) - base_frame.astype(np.float32)), axis=2)
    detail_safety = np.exp(-detail_delta / max(config.detail_delta_sigma, 1e-6)).astype(np.float32)

    motion_conf = np.ones_like(base_gray, dtype=np.float32)
    edge_conf = np.ones_like(base_gray, dtype=np.float32)
    if prev_base_frame is not None:
        warped_prev, flow = warp_previous_to_current(
            prev_base_frame,
            base_frame,
            prev_base_frame,
            config.flow_max_side,
        )
        flow_mag = np.linalg.norm(flow, axis=2)
        motion_conf = np.exp(-flow_mag / max(config.motion_sigma, 1e-6)).astype(np.float32)

        cur_edge = cv2.Sobel(base_gray, cv2.CV_32F, 1, 1, ksize=3)
        prev_edge = cv2.Sobel(_gray(warped_prev), cv2.CV_32F, 1, 1, ksize=3)
        edge_residual = np.abs(cur_edge - prev_edge)
        edge_conf = np.exp(-edge_residual / max(config.edge_sigma, 1e-6)).astype(np.float32)

    mask = config.max_detail_alpha * texture_gate * detail_safety * motion_conf * edge_conf
    return np.clip(mask, 0.0, config.max_detail_alpha).astype(np.float32)


def adaptive_detail_refine(
    base_frame: np.ndarray,
    prev_base_frame: Optional[np.ndarray],
    config: TemporalHybridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detail = build_detail_candidate(base_frame, config)
    mask = build_adaptive_detail_mask(base_frame, detail, prev_base_frame, config)
    mask3 = mask[..., None]
    refined = base_frame.astype(np.float32) * (1.0 - mask3) + detail.astype(np.float32) * mask3
    return _to_uint8(refined), mask, detail


def motion_compensated_smooth(
    cur_frame: np.ndarray,
    prev_out: Optional[np.ndarray],
    cur_base: np.ndarray,
    prev_base: Optional[np.ndarray],
    config: TemporalHybridConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if prev_out is None or prev_base is None:
        return cur_frame.copy(), np.ones(cur_frame.shape[:2], dtype=np.float32)

    warped_prev_out, flow = warp_previous_to_current(prev_out, cur_base, prev_base, config.flow_max_side)
    warped_prev_base = _remap_with_flow(prev_base, flow)
    flow_mag = np.linalg.norm(flow, axis=2)

    cur_edge = cv2.Sobel(_gray(cur_base), cv2.CV_32F, 1, 1, ksize=3)
    prev_edge = cv2.Sobel(_gray(warped_prev_base), cv2.CV_32F, 1, 1, ksize=3)
    edge_residual = np.abs(cur_edge - prev_edge)

    motion_conf = np.exp(-flow_mag / max(config.motion_sigma, 1e-6))
    edge_conf = np.exp(-edge_residual / max(config.edge_sigma, 1e-6))
    stability = np.clip(motion_conf * edge_conf, 0.0, 1.0).astype(np.float32)

    blend = (config.temporal_strength * stability)[..., None]
    out = cur_frame.astype(np.float32) * (1.0 - blend) + warped_prev_out.astype(np.float32) * blend
    return _to_uint8(out), stability


def refine_sequence(base_frames: list[np.ndarray], config: TemporalHybridConfig) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    outputs: list[np.ndarray] = []
    stats: list[dict[str, float]] = []
    prev_base: Optional[np.ndarray] = None
    prev_out: Optional[np.ndarray] = None

    for idx, base in enumerate(base_frames):
        detail_refined, detail_mask, _ = adaptive_detail_refine(base, prev_base, config)
        out, stability = motion_compensated_smooth(detail_refined, prev_out, base, prev_base, config)
        outputs.append(out)
        stats.append(
            {
                "frame": float(idx),
                "detail_alpha_mean": float(detail_mask.mean()),
                "detail_alpha_p95": float(np.percentile(detail_mask, 95)),
                "stability_mean": float(stability.mean()),
                "sharpness_laplacian": laplacian_variance(out),
            }
        )
        prev_base = base
        prev_out = out
    return outputs, stats


def laplacian_variance(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(frame), cv2.CV_32F).var())


def read_frames_from_dir(frame_dir: Path, max_frames: Optional[int] = None) -> list[np.ndarray]:
    paths = sorted([p for p in frame_dir.glob("*.png")])
    if max_frames:
        paths = paths[:max_frames]
    frames = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def read_frames_from_video(video_path: Path, max_frames: Optional[int] = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def save_frames(frames: Iterable[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:08d}.png"), frame)


def save_video(frames: list[np.ndarray], video_path: Path, fps: int) -> None:
    if not frames:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def temporal_warp_error(frames: list[np.ndarray]) -> tuple[float, list[tuple[int, float]]]:
    if len(frames) < 2:
        return 0.0, []
    errors: list[tuple[int, float]] = []
    for idx in range(1, len(frames)):
        warped, _ = warp_previous_to_current(frames[idx - 1], frames[idx], frames[idx - 1])
        err = float(np.mean(np.abs(frames[idx].astype(np.float32) - warped.astype(np.float32))))
        errors.append((idx, err))
    return float(np.mean([e for _, e in errors])), sorted(errors, key=lambda x: x[1], reverse=True)[:5]


def edge_flicker(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    vals = []
    prev_edge = cv2.Canny(frames[0], 80, 160).astype(np.float32)
    for frame in frames[1:]:
        cur_edge = cv2.Canny(frame, 80, 160).astype(np.float32)
        vals.append(float(np.mean(np.abs(cur_edge - prev_edge)) / 255.0))
        prev_edge = cur_edge
    return float(np.mean(vals))


def frame_diff(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    vals = [
        float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
        for i in range(1, len(frames))
    ]
    return float(np.mean(vals))


def _resize_for_diagnostics(frames: list[np.ndarray], max_side: int) -> list[np.ndarray]:
    if max_side <= 0:
        return frames
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        side = max(h, w)
        if side <= max_side:
            resized.append(frame)
            continue
        scale = max_side / float(side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized.append(cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA))
    return resized


def evaluate_against_gt(sr_frames: list[np.ndarray], gt_frames: list[np.ndarray]) -> dict[str, float]:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except Exception:
        return {}
    psnr_vals = []
    ssim_vals = []
    for sr, gt in zip(sr_frames, gt_frames):
        if sr.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_CUBIC)
        psnr_vals.append(float(peak_signal_noise_ratio(gt, sr, data_range=255)))
        ssim_vals.append(float(structural_similarity(gt, sr, channel_axis=2, data_range=255)))
    if not psnr_vals:
        return {}
    return {"psnr": float(np.mean(psnr_vals)), "ssim": float(np.mean(ssim_vals))}


def sequence_diagnostics(name: str, frames: list[np.ndarray], diagnostic_max_side: int = 360) -> dict[str, object]:
    temporal_frames = _resize_for_diagnostics(frames, diagnostic_max_side)
    warp_err, worst_pairs = temporal_warp_error(temporal_frames)
    return {
        "name": name,
        "num_frames": len(frames),
        "sharpness_laplacian": float(np.mean([laplacian_variance(f) for f in frames])) if frames else 0.0,
        "raw_frame_diff": frame_diff(temporal_frames),
        "warp_error": warp_err,
        "edge_flicker": edge_flicker(temporal_frames),
        "worst_pairs": "; ".join([f"{idx}:{val:.2f}" for idx, val in worst_pairs]),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_limitations_report(rows: list[dict[str, object]], path: Path) -> None:
    part1 = [r for r in rows if str(r["name"]).startswith("part1")]
    part2 = [r for r in rows if str(r["name"]).startswith("part2")]
    basicvsrpp = [r for r in part2 if "basicvsrpp" in str(r["name"]).lower()]
    realesrgan = [r for r in part2 if "realesrgan" in str(r["name"]).lower()]

    def _best(rows_in: list[dict[str, object]], key: str, reverse: bool = False) -> Optional[dict[str, object]]:
        if not rows_in:
            return None
        return sorted(rows_in, key=lambda r: float(r[key]), reverse=reverse)[0]

    part1_blur = _best(part1, "sharpness_laplacian")
    part1_flicker = _best(part1, "warp_error", reverse=True)
    basic_smooth = _best(basicvsrpp, "sharpness_laplacian")
    basic_flicker = _best(basicvsrpp, "edge_flicker", reverse=True)
    real_flicker = _best(realesrgan, "edge_flicker", reverse=True)

    lines = [
        "# Part 1/2 limitation analysis",
        "",
        "The numbers below were computed from the rendered videos/frames in `results/`.",
        "`warp_error` is measured after optical-flow alignment, so it is more relevant to flicker than raw frame difference.",
        "",
        "## Main observations",
        "",
    ]
    if part1_blur:
        lines.append(
            f"- Part 1 remains detail-limited: the strongest Part 1 sharpness score in this run is "
            f"`{part1_blur['name']}` with Laplacian variance `{float(part1_blur['sharpness_laplacian']):.2f}`, "
            "but its interpolation/SRCNN pipeline still works frame by frame and cannot recover stable fine texture."
        )
    if part1_flicker:
        lines.append(
            f"- Part 1 temporal averaging reduces noise but smears motion. The largest aligned temporal error appears in "
            f"`{part1_flicker['name']}` (`warp_error={float(part1_flicker['warp_error']):.2f}`), "
            f"with worst adjacent frame pairs `{part1_flicker['worst_pairs']}`."
        )
    if basic_smooth:
        lines.append(
            f"- BasicVSR++ is much more coherent than Part 1, but it is conservative on high-frequency regions. "
            f"The smoothest BasicVSR++ sample here is `{basic_smooth['name']}` "
            f"with Laplacian variance `{float(basic_smooth['sharpness_laplacian']):.2f}`."
        )
    if basic_flicker:
        lines.append(
            f"- BasicVSR++ still shows local temporal stress on moving REDS/Vimeo clips. "
            f"The highest BasicVSR++ edge flicker in the sampled set is `{basic_flicker['name']}` "
            f"(`edge_flicker={float(basic_flicker['edge_flicker']):.4f}`), "
            f"with worst aligned pairs `{basic_flicker['worst_pairs']}`."
        )
    if real_flicker:
        lines.append(
            f"- The single-frame Real-ESRGAN branch gives sharper local texture, but it is not temporally aware. "
            f"In the wild-video sample, `{real_flicker['name']}` has "
            f"`edge_flicker={float(real_flicker['edge_flicker']):.4f}` and "
            f"`warp_error={float(real_flicker['warp_error']):.2f}`."
        )
    lines.extend(
        [
            "",
            "## Part 3 response",
            "",
            "Part 3 therefore uses BasicVSR++/Real-ESRGAN outputs as candidates instead of replacing them.",
            "It adds detail only in stable textured areas, suppresses detail on high-motion or edge-unstable regions, and blends a motion-compensated previous output into the current frame.",
            "This directly targets the two visible failures above: over-smoothing in recurrent VSR and flickering hallucinated texture in single-frame enhancement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_part12_sources(project_root: Path, max_frames: int) -> list[tuple[str, list[np.ndarray]]]:
    results = project_root / "results"
    sources: list[tuple[str, list[np.ndarray]]] = []

    for video in sorted((results / "part1_videos").glob("*.avi")):
        frames = read_frames_from_video(video, max_frames=max_frames)
        if frames:
            sources.append((f"part1/{video.stem}", frames))

    wild_realesrgan = results / "wild_video_RealESRGAN_x4.mp4"
    if wild_realesrgan.exists():
        frames = read_frames_from_video(wild_realesrgan, max_frames=max_frames)
        if frames:
            sources.append(("part2/wild_RealESRGAN_x4", frames))

    reds_root = results / "reds_basicvsrpp"
    for seq_dir in sorted([p for p in reds_root.glob("*") if p.is_dir()])[:3]:
        frames = read_frames_from_dir(seq_dir, max_frames=max_frames)
        if frames:
            sources.append((f"part2/reds_basicvsrpp_{seq_dir.name}", frames))

    vimeo_root = results / "vimeo_basicvsrpp" / "images"
    for clip_dir in sorted([p for p in vimeo_root.glob("*") if p.is_dir()])[:3]:
        frames = read_frames_from_dir(clip_dir, max_frames=max_frames)
        if frames:
            sources.append((f"part2/vimeo_basicvsrpp_{clip_dir.name}", frames))

    return sources


def analyze_part12(project_root: Path, out_dir: Path, max_frames: int) -> list[dict[str, object]]:
    rows = [sequence_diagnostics(name, frames) for name, frames in collect_part12_sources(project_root, max_frames)]
    write_csv(rows, out_dir / "part12_limitations.csv")
    write_limitations_report(rows, out_dir / "part12_limitations.md")
    return rows


def _load_default_base(project_root: Path, dataset: str, seq: Optional[str], max_frames: int) -> tuple[str, list[np.ndarray], list[np.ndarray]]:
    results = project_root / "results"
    gt_frames: list[np.ndarray] = []
    if dataset == "reds":
        seq_name = seq or "000"
        base_dir = results / "reds_basicvsrpp" / seq_name
        base_frames = read_frames_from_dir(base_dir, max_frames=max_frames)
        gt_dir = project_root / "datasets" / "reds" / "val_sharp" / "val" / "val_sharp" / seq_name
        gt_frames = read_frames_from_dir(gt_dir, max_frames=len(base_frames))
        return f"reds_{seq_name}", base_frames, gt_frames
    if dataset == "vimeo":
        image_root = results / "vimeo_basicvsrpp" / "images"
        candidates = sorted([p for p in image_root.glob("*") if p.is_dir()])
        if not candidates:
            return "vimeo_missing", [], []
        base_dir = candidates[0] if seq is None else image_root / seq
        base_frames = read_frames_from_dir(base_dir, max_frames=max_frames)
        return f"vimeo_{base_dir.name}", base_frames, []
    if dataset == "wild":
        frame_dir = results / "temp_frames_output"
        base_frames = read_frames_from_dir(frame_dir, max_frames=max_frames)
        if not base_frames:
            video = results / "wild_video_RealESRGAN_x4.mp4"
            base_frames = read_frames_from_video(video, max_frames=max_frames)
        return "wild_realesrgan", base_frames, []
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_part3(project_root: Path, out_dir: Path, dataset: str, seq: Optional[str], config: TemporalHybridConfig) -> dict[str, object]:
    name, base_frames, gt_frames = _load_default_base(project_root, dataset, seq, config.max_frames)
    if not base_frames:
        raise FileNotFoundError(f"No base frames found for {dataset}. Run Part 2 first.")

    refined, frame_stats = refine_sequence(base_frames, config)
    target_dir = out_dir / dataset / name
    save_frames(refined, target_dir / "frames")
    save_video(refined, target_dir / f"{name}_part3_temporal_hybrid.mp4", config.fps)

    metrics = {
        "name": name,
        "dataset": dataset,
        "base": sequence_diagnostics(f"base/{name}", base_frames),
        "part3": sequence_diagnostics(f"part3/{name}", refined),
        "config": asdict(config),
    }
    gt_metrics = evaluate_against_gt(refined, gt_frames) if gt_frames else {}
    if gt_metrics:
        metrics["gt_metrics"] = gt_metrics
    write_json(metrics, target_dir / "metrics.json")
    write_csv(frame_stats, target_dir / "frame_stats.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 3 temporal hybrid VSR refinement")
    parser.add_argument("--project-root", default="/data/cyang690/vsr_project_2026")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--mode", choices=["analyze", "run", "all"], default="all")
    parser.add_argument("--dataset", choices=["reds", "vimeo", "wild", "all"], default="all")
    parser.add_argument("--seq", default=None)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--temporal-strength", type=float, default=0.62)
    parser.add_argument("--max-detail-alpha", type=float, default=0.58)
    parser.add_argument("--flow-max-side", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results" / "part3_temporal_hybrid"
    config = TemporalHybridConfig(
        max_frames=args.max_frames,
        fps=args.fps,
        temporal_strength=args.temporal_strength,
        max_detail_alpha=args.max_detail_alpha,
        flow_max_side=args.flow_max_side,
    )

    if args.mode in {"analyze", "all"}:
        rows = analyze_part12(project_root, out_dir, args.max_frames)
        print(f"Part 1/2 analysis rows: {len(rows)}")
        print(f"Analysis written to: {out_dir / 'part12_limitations.md'}")

    if args.mode in {"run", "all"}:
        datasets = ["reds", "vimeo", "wild"] if args.dataset == "all" else [args.dataset]
        for dataset in datasets:
            metrics = run_part3(project_root, out_dir, dataset, args.seq, config)
            base_warp = float(metrics["base"]["warp_error"])
            part3_warp = float(metrics["part3"]["warp_error"])
            print(
                f"{dataset}: warp_error {base_warp:.3f} -> {part3_warp:.3f}; "
                f"edge_flicker {float(metrics['base']['edge_flicker']):.4f} -> {float(metrics['part3']['edge_flicker']):.4f}"
            )


if __name__ == "__main__":
    main()
