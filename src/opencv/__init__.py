"""
src/opencv — Giai đoạn 1: Đo kích thước bàn chân bằng OpenCV thuần
Đồ án: "Ứng dụng thị giác máy tính đo kích thước bàn chân qua ảnh điện thoại"

Các module:
    constants       — Hằng số chuẩn ISO/IEC 7810 ID-1 và cài đặt toàn cục
    card_detector   — Phát hiện thẻ ID-1 trong ảnh (Khối 1)
    perspective     — Nắn phối cảnh toàn cảnh Homography (Khối 2)
    foot_segmenter  — Tách mặt nạ bàn chân, cắt gót chân (Khối 3)
    foot_measurer   — Đo kích thước chân bằng PCA (Khối 4)
    pipeline        — Xử lý tuần tự và batch (Khối 5)
"""

from .constants import (
    CARD_LONG_MM,
    CARD_SHORT_MM,
    CARD_ASPECT_RATIO,
    MAX_IMAGE_DIM,
)
from .card_detector import detect_card, detect_card_corners
from .perspective import rectify_perspective, order_points, order_corners
from .foot_segmenter import segment_foot
from .foot_measurer import measure_foot_pca
from .pipeline import process_image_pipeline, run_batch_pipeline

__all__ = [
    "CARD_LONG_MM",
    "CARD_SHORT_MM",
    "CARD_ASPECT_RATIO",
    "MAX_IMAGE_DIM",
    "detect_card",
    "detect_card_corners",
    "rectify_perspective",
    "order_points",
    "order_corners",
    "segment_foot",
    "measure_foot_pca",
    "process_image_pipeline",
    "run_batch_pipeline",
]
