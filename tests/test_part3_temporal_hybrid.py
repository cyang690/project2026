import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part3"))

from temporal_hybrid_vsr import (  # noqa: E402
    TemporalHybridConfig,
    build_adaptive_detail_mask,
    motion_compensated_smooth,
)
import temporal_hybrid_vsr as hybrid  # noqa: E402


def _gradient_frame(width=32, height=24, shift=0):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    frame = np.tile(x, (height, 1))
    frame = np.roll(frame, shift, axis=1)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_adaptive_detail_mask_suppresses_high_motion_regions():
    prev_base = _gradient_frame(shift=0)
    cur_base = _gradient_frame(shift=5)
    detail = cv2.convertScaleAbs(cur_base, alpha=1.25, beta=4)
    config = TemporalHybridConfig(max_detail_alpha=0.65)

    mask = build_adaptive_detail_mask(cur_base, detail, prev_base, config)

    assert mask.shape == cur_base.shape[:2]
    assert mask.dtype == np.float32
    assert 0.0 <= float(mask.min()) <= float(mask.max()) <= config.max_detail_alpha
    assert float(mask.mean()) < config.max_detail_alpha * 0.55


def test_motion_compensated_smooth_reduces_static_flicker():
    prev_out = np.full((24, 32, 3), 112, dtype=np.uint8)
    cur_frame = np.full((24, 32, 3), 148, dtype=np.uint8)
    prev_base = prev_out.copy()
    cur_base = cur_frame.copy()
    config = TemporalHybridConfig(temporal_strength=0.7)

    smoothed, stability = motion_compensated_smooth(
        cur_frame,
        prev_out,
        cur_base,
        prev_base,
        config,
    )

    before = np.mean(np.abs(cur_frame.astype(np.float32) - prev_out.astype(np.float32)))
    after = np.mean(np.abs(smoothed.astype(np.float32) - prev_out.astype(np.float32)))
    assert stability.mean() > 0.5
    assert after < before


def test_sequence_diagnostics_downscales_temporal_warp_work():
    frames = [
        np.zeros((120, 200, 3), dtype=np.uint8),
        np.full((120, 200, 3), 16, dtype=np.uint8),
    ]
    seen_shapes = []
    original = hybrid.warp_previous_to_current

    def fake_warp(prev_frame, cur_base, prev_base):
        seen_shapes.append(cur_base.shape[:2])
        h, w = cur_base.shape[:2]
        return prev_frame.copy(), np.zeros((h, w, 2), dtype=np.float32)

    hybrid.warp_previous_to_current = fake_warp
    try:
        hybrid.sequence_diagnostics("large", frames, diagnostic_max_side=50)
    finally:
        hybrid.warp_previous_to_current = original

    assert seen_shapes
    assert max(seen_shapes[0]) <= 50


def test_motion_compensation_can_compute_flow_on_downscaled_frames():
    cur_base = np.full((120, 200, 3), 80, dtype=np.uint8)
    prev_base = np.full((120, 200, 3), 80, dtype=np.uint8)
    cur_frame = np.full((120, 200, 3), 100, dtype=np.uint8)
    prev_out = np.full((120, 200, 3), 90, dtype=np.uint8)
    config = TemporalHybridConfig(temporal_strength=0.7, flow_max_side=40)

    seen_shapes = []
    original = hybrid._flow_cur_to_prev

    def fake_flow(cur_gray, prev_gray):
        seen_shapes.append(cur_gray.shape)
        h, w = cur_gray.shape
        return np.zeros((h, w, 2), dtype=np.float32)

    hybrid._flow_cur_to_prev = fake_flow
    try:
        smoothed, stability = motion_compensated_smooth(
            cur_frame,
            prev_out,
            cur_base,
            prev_base,
            config,
        )
    finally:
        hybrid._flow_cur_to_prev = original

    assert seen_shapes
    assert max(seen_shapes[0]) <= 40
    assert smoothed.shape == cur_frame.shape
    assert stability.shape == cur_frame.shape[:2]
