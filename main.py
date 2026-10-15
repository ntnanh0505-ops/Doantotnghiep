"""
Entry Point tại thư mục gốc:
Chuyển tiếp lệnh gọi vào chuongtrinhchay/doan.py
"""
import sys
import runpy
from pathlib import Path

# Đảm bảo in tiếng Việt có dấu ổn định trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from chuongtrinhchay.doan import (
    CARD_LONG_MM,
    CARD_SHORT_MM,
    CARD_ASPECT_RATIO,
    MAX_IMAGE_DIM,
    order_points,
    order_corners,
    detect_card_corners,
    detect_card,
    rectify_perspective,
    segment_foot,
    measure_foot_pca,
    process_image_pipeline,
    run_batch_pipeline,
    main,
)

if __name__ == "__main__":
    main()

