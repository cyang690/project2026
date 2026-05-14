from test_part3_temporal_hybrid import (
    test_adaptive_detail_mask_suppresses_high_motion_regions,
    test_motion_compensated_smooth_reduces_static_flicker,
    test_motion_compensation_can_compute_flow_on_downscaled_frames,
    test_sequence_diagnostics_downscales_temporal_warp_work,
)


if __name__ == "__main__":
    test_adaptive_detail_mask_suppresses_high_motion_regions()
    test_motion_compensated_smooth_reduces_static_flicker()
    test_motion_compensation_can_compute_flow_on_downscaled_frames()
    test_sequence_diagnostics_downscales_temporal_warp_work()
    print("manual-tests-pass")
