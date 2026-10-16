"""
perspective.py — Khối 2: Nắn phối cảnh toàn cảnh Homography (rectify_perspective).

Chức năng:
    - order_points(pts)        : Sắp xếp 4 điểm góc theo thứ tự chuẩn TL→TR→BR→BL.
    - rectify_perspective(...) : Nắn ảnh về mặt phẳng phẳng bằng Homography T @ H_card,
                                  bảo toàn toàn bộ bàn chân, tạo valid_mask.
"""

import cv2
import numpy as np
from .constants import CARD_LONG_MM, CARD_SHORT_MM, MAX_IMAGE_DIM


def order_points(pts):
    """
    Sắp xếp 4 tọa độ đỉnh theo thứ tự chuẩn OpenCV và hình học:
    - Điểm có tổng (x + y) nhỏ nhất là Top-Left (TL).
    - Điểm có tổng (x + y) lớn nhất là Bottom-Right (BR).
    - Điểm có hiệu (y - x) nhỏ nhất là Top-Right (TR).
    - Điểm có hiệu (y - x) lớn nhất là Bottom-Left (BL).
    Đảm bảo ảnh sau khi nắn (rectified) không bị lộn ngược 180 độ.
    """
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect).astype(np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    # Tổng (x + y): TL nhỏ nhất, BR lớn nhất
    s = pts[:, 0] + pts[:, 1]
    rect[0] = pts[np.argmin(s)]  # Top-Left (TL)
    rect[2] = pts[np.argmax(s)]  # Bottom-Right (BR)

    # Hiệu (x - y): TR lớn nhất (x lớn, y nhỏ), BL nhỏ nhất (x nhỏ, y lớn)
    diff = pts[:, 0] - pts[:, 1]
    rect[1] = pts[np.argmax(diff)]  # Top-Right (TR)
    rect[3] = pts[np.argmin(diff)]  # Bottom-Left (BL)

    return rect


# Alias tương thích ngược
order_corners = order_points


def rectify_perspective(image, card_corners, target_px_per_mm=None):
    """
    Khối 2 — Nắn phối cảnh:
    - Đưa thẻ về hình chữ nhật chuẩn ISO/IEC 7810 ID-1 (85.60 x 53.98 mm).
    - Tự động nhận diện cạnh ngang / cạnh dọc khớp với cạnh dài / cạnh ngắn của thẻ.
    - Biến đổi toàn bộ vùng ảnh qua phép tịnh tiến T @ H_card để bảo toàn trọn vẹn bàn chân.
    - Tạo valid_mask theo dõi vùng ảnh gốc hợp lệ.
    - Đo kích thước thẻ đã nắn để kiểm chứng tính chuẩn xác milimet.
    """
    h_orig, w_orig = image.shape[:2]
    tl, tr, br, bl = card_corners

    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    card_w_px = (w_top + w_bot) / 2.0
    card_h_px = (h_left + h_right) / 2.0

    # Xác định hướng thẻ (cạnh trên là cạnh dài hay ngắn)
    if card_w_px >= card_h_px:
        card_w_mm, card_h_mm = CARD_LONG_MM, CARD_SHORT_MM
    else:
        card_w_mm, card_h_mm = CARD_SHORT_MM, CARD_LONG_MM

    if target_px_per_mm is not None and target_px_per_mm > 0:
        px_per_mm = float(target_px_per_mm)
    else:
        scale_x = card_w_px / max(1.0, card_w_mm)
        scale_y = card_h_px / max(1.0, card_h_mm)
        px_per_mm = (scale_x + scale_y) / 2.0

    dst_card_w = card_w_mm * px_per_mm
    dst_card_h = card_h_mm * px_per_mm

    cx = float(np.mean(card_corners[:, 0]))
    cy = float(np.mean(card_corners[:, 1]))
    hw = dst_card_w / 2.0
    hh = dst_card_h / 2.0

    dst_card = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh]
    ], dtype=np.float32)

    H_card = cv2.getPerspectiveTransform(card_corners.astype(np.float32), dst_card)

    orig_corners = np.array([
        [0.0, 0.0],
        [float(w_orig), 0.0],
        [float(w_orig), float(h_orig)],
        [0.0, float(h_orig)]
    ], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(orig_corners, H_card).reshape(-1, 2)
    x_min, y_min = np.min(warped_corners, axis=0)
    x_max, y_max = np.max(warped_corners, axis=0)

    # Phép tịnh tiến dịch về góc dương
    T = np.array([
        [1.0, 0.0, -x_min],
        [0.0, 1.0, -y_min],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    H_total = T @ H_card
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))

    max_side = max(out_w, out_h)
    if max_side > MAX_IMAGE_DIM:
        scale_down = MAX_IMAGE_DIM / float(max_side)
        S = np.array([
            [scale_down, 0.0, 0.0],
            [0.0, scale_down, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        H_total = S @ H_total
        out_w = int(np.ceil(out_w * scale_down))
        out_h = int(np.ceil(out_h * scale_down))
        px_per_mm = px_per_mm * scale_down

    out_w = max(10, min(MAX_IMAGE_DIM, out_w))
    out_h = max(10, min(MAX_IMAGE_DIM, out_h))

    # Nắn ảnh
    rectified_img = cv2.warpPerspective(
        image,
        H_total,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    # Tạo valid_mask xác định chính xác phần ảnh gốc bên trong ảnh nắn
    orig_ones = np.ones((h_orig, w_orig), dtype=np.uint8) * 255
    valid_mask = cv2.warpPerspective(
        orig_ones,
        H_total,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    warped_card_corners = cv2.perspectiveTransform(
        card_corners.astype(np.float32).reshape(-1, 1, 2),
        H_total
    ).reshape(4, 2)

    mm_per_pixel = 1.0 / max(1e-5, px_per_mm)

    return {
        "rectified_image": rectified_img,
        "card_polygon": warped_card_corners,
        "pixels_per_mm": px_per_mm,
        "mm_per_pixel": mm_per_pixel,
        "card_length_mm": CARD_LONG_MM,
        "card_width_mm": CARD_SHORT_MM,
        "transform_matrix": H_total,
        "valid_mask": valid_mask
    }
