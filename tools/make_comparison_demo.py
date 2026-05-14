#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def list_frames(path, limit):
    frames = sorted(Path(path).glob("*.png"))
    return frames[:limit]


def read_frame(path):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {path}")
    return frame


def fit_to_box(frame, width, height):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def put_text(img, text, org, scale=0.7, color=(245, 245, 245), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def make_pair_frame(base, part3, title, metric_text, frame_index, frame_count, side_w, side_h):
    left = fit_to_box(base, side_w, side_h)
    right = fit_to_box(part3, side_w, side_h)
    body = np.hstack([left, right])

    header_h = 92
    footer_h = 58
    canvas = np.full((header_h + side_h + footer_h, side_w * 2, 3), 28, dtype=np.uint8)
    canvas[header_h : header_h + side_h, :] = body
    canvas[:, side_w - 1 : side_w + 1] = (90, 90, 90)

    put_text(canvas, title, (28, 36), scale=0.8, color=(255, 255, 255), thickness=2)
    put_text(canvas, "Base output", (28, 76), scale=0.72, color=(210, 230, 255), thickness=2)
    put_text(canvas, "Part 3 temporal hybrid", (side_w + 28, 76), scale=0.72, color=(210, 255, 220), thickness=2)
    put_text(canvas, metric_text, (28, header_h + side_h + 38), scale=0.62, color=(245, 245, 245), thickness=2)
    put_text(canvas, f"frame {frame_index + 1}/{frame_count}", (side_w * 2 - 190, header_h + side_h + 38), scale=0.58, color=(220, 220, 220), thickness=1)
    return canvas


def append_segment(writer, base_paths, part3_paths, title, metric_text, repeat, side_w, side_h):
    count = min(len(base_paths), len(part3_paths))
    if count == 0:
        raise RuntimeError(f"No frames found for segment: {title}")
    for idx in range(count):
        base = read_frame(base_paths[idx])
        part3 = read_frame(part3_paths[idx])
        frame = make_pair_frame(base, part3, title, metric_text, idx, count, side_w, side_h)
        for _ in range(repeat):
            writer.write(frame)


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side comparison demo video")
    parser.add_argument("--project-root", default="/data/cyang690/vsr_project_2026")
    parser.add_argument("--output", default="results/comparison_videos/comparison_demo.mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--side-width", type=int, default=640)
    parser.add_argument("--side-height", type=int, default=360)
    parser.add_argument("--wild-frames", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.project_root)
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size = (args.side_width * 2, 92 + args.side_height + 58)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out_path}")

    vimeo_base = list_frames(root / "results/vimeo_basicvsrpp/images/00001_0266", 7)
    vimeo_part3 = list_frames(root / "results/part3_temporal_hybrid/vimeo/vimeo_00001_0266/frames", 7)
    append_segment(
        writer,
        vimeo_base,
        vimeo_part3,
        "Vimeo-90K 00001_0266",
        "warp_error: 2.187 -> 1.394    edge_flicker: 0.0213 -> 0.0170",
        args.repeat,
        args.side_width,
        args.side_height,
    )

    wild_base = list_frames(root / "results/temp_frames_output", args.wild_frames)
    wild_part3 = list_frames(root / "results/part3_temporal_hybrid/wild/wild_realesrgan/frames", args.wild_frames)
    append_segment(
        writer,
        wild_base,
        wild_part3,
        "Wild Video",
        "warp_error: 1.670 -> 1.487    edge_flicker: 0.0811 -> 0.0757",
        args.repeat,
        args.side_width,
        args.side_height,
    )

    writer.release()
    print(f"comparison video saved: {out_path}")


if __name__ == "__main__":
    main()
