"""
Entry Point tại thư mục gốc:
Chuyển tiếp lệnh gọi vào chuongtrinhchay/doan.py
"""
import sys
from pathlib import Path

# Đảm bảo in tiếng Việt có dấu ổn định trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from chuongtrinhchay.doan import (
    CARD_LONG_MM,
    CARD_SHORT_MM,
    CARD_ASPECT_RATIO,
    HOMOGRAPHY_SCALE_PX_PER_MM,
    MAX_IMAGE_DIM,
    load_dataset,
    assess_image_quality,
    order_points,
    order_corners,
    detect_card_corners,
    detect_card,
    compare_scale_methods,
    rectify_perspective,
    segment_foot,
    classify_foot_shape,
    rotate_foot_vertical,
    measure_foot_pca,
    process_image_pipeline,
    run_dataset_loop,
    run_batch_pipeline,
    main,
)

if __name__ == "__main__":
    main()
