"""
ĐỒ ÁN TỐT NGHIỆP:
"Hệ thống đo kích thước bàn chân từ ảnh chụp cùng thẻ tham chiếu (ID Card)"

PIPELINE ĐO ĐẠC 7 KHỐI XỬ LÝ (OPENCV, NUMPY, GLOB/OS):
1. Tải và duyệt dữ liệu (Dataset Loader): load_dataset(split='test')
2. Kiểm tra chất lượng ảnh đầu vào (Image Quality Assessment): Độ sắc nét (Laplacian) & Độ sáng
3. Phát hiện 4 góc thẻ tham chiếu ID-1 (Canny, Dilation, approxPolyDP, Bounding Box/Contour Filtering)
4. Hiệu chỉnh phối cảnh (Perspective Transform Homography) & So sánh tỷ lệ với phương pháp đơn giản
5. Tách mặt nạ bàn chân (HSV/YCrCb Skin Segmentation + Otsu trên nền sáng + Morphological Filtering)
6. Xác định trục chính (PCA) & Đo đạc kích thước (Chiều dài, Chiều rộng 1/3 trước, Tỷ số W/L, Phân loại dáng chân)
7. Vẽ trực quan kết quả & Báo cáo terminal chi tiết từng ảnh
"""

import os
import sys
import glob
import time
import csv
import argparse
from pathlib import Path
import cv2
import numpy as np

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# HẰNG SỐ CHUẨN QUỐC TẾ (ISO/IEC 7810 ID-1)
# ==============================================================================
CARD_LONG_MM = 85.60     # Chiều dài chuẩn thẻ ID-1 (CCCD / thẻ ngân hàng / thẻ SV)
CARD_SHORT_MM = 54.00    # Chiều rộng chuẩn thẻ ID-1
CARD_ASPECT_RATIO = CARD_LONG_MM / CARD_SHORT_MM  # ~ 1.585185...
HOMOGRAPHY_SCALE_PX_PER_MM = 10.0  # S = 10 px/mm (1 mm = 0.1 px)
MAX_IMAGE_DIM = 2600     # Giới hạn kích thước tối đa tránh tràn bộ nhớ khi warp


def get_project_root() -> Path:
    """Trả về thư mục gốc của dự án."""
    return Path(__file__).resolve().parent.parent


# ==============================================================================
# KHỐI 1: TẢI VÀ DUYỆT DỮ LIỆU (DATASET LOADER)
# ==============================================================================
def load_dataset(split: str = "test", dataset_dir: Path = None):
    """
    Hàm load_dataset(split='test'):
    - Tự động duyệt và quét toàn bộ file ảnh .jpg, .jpeg, .png từ thư mục split.
    - Mặc định tìm tại data/bodulieu/{split}.
    """
    if dataset_dir is None:
        root = get_project_root()
        candidate_paths = [
            root / "data" / "bodulieu" / split,
            Path("data/bodulieu") / split,
            Path("data") / split,
            Path(split)
        ]
        target_dir = None
        for cp in candidate_paths:
            if cp.exists() and cp.is_dir():
                target_dir = cp.resolve()
                break
        if target_dir is None:
            target_dir = (root / "data" / "bodulieu" / split).resolve()
    else:
        target_dir = Path(dataset_dir) / split if (Path(dataset_dir) / split).exists() else Path(dataset_dir)

    if not target_dir.exists():
        print(f"[CẢNH BÁO] Không tìm thấy thư mục tập dữ liệu: {target_dir}")
        return []

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
                  "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP")
    images = []
    for ext in valid_exts:
        pattern = os.path.join(str(target_dir), "**", ext)
        images.extend(glob.glob(pattern, recursive=True))

    images = sorted(list(set(images)))
    return images


# ==============================================================================
# KHỐI 2: KIỂM TRA CHẤT LƯỢNG ẢNH ĐẦU VÀO (IMAGE QUALITY ASSESSMENT)
# ==============================================================================
def assess_image_quality(image, sharpness_thresh: float = 80.0,
                         min_brightness: float = 50.0, max_brightness: float = 215.0):
    """
    Khối 2 — Đánh giá chất lượng ảnh:
    - Độ sắc nét: Tính phương sai Laplacian (cv2.Laplacian(gray, cv2.CV_64F).var()).
      Nếu var < 80: Báo ảnh mờ/nhòe và yêu cầu chụp lại.
    - Độ sáng: Tính giá trị trung bình kênh Grayscale.
      Nếu < 50: Báo ảnh quá tối.
      Nếu > 215: Báo ảnh quá sáng / cháy sáng.
    """
    if image is None:
        return {
            "is_valid": False,
            "sharpness": 0.0,
            "brightness": 0.0,
            "error_message": "Không thể tải file ảnh"
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    errors = []
    if sharpness < sharpness_thresh:
        errors.append(f"Ảnh mờ/nhòe (Laplacian var = {sharpness:.1f} < {sharpness_thresh:.0f})")
    if brightness < min_brightness:
        errors.append(f"Ảnh quá tối (Độ sáng = {brightness:.1f} < {min_brightness:.0f})")
    elif brightness > max_brightness:
        errors.append(f"Ảnh quá sáng/cháy sáng (Độ sáng = {brightness:.1f} > {max_brightness:.0f})")

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "sharpness": sharpness,
        "brightness": brightness,
        "error_message": "; ".join(errors) if errors else ""
    }


# ==============================================================================
# HÀM BỔ TRỢ: SẮP XẾP 4 ĐỈNH THẺ CHUẨN (TL, TR, BR, BL)
# ==============================================================================
def order_points(pts):
    """
    Sắp xếp 4 tọa độ đỉnh theo thứ tự chuẩn OpenCV và hình học:
    - Top-Left (TL), Top-Right (TR), Bottom-Right (BR), Bottom-Left (BL).
    - Sử dụng góc cực (polar angle) quanh trọng tâm để bảo đảm 4 đỉnh luôn phân biệt
      theo chiều kim đồng hồ, không bị trùng đỉnh khi thẻ xoay góc nghiêng 45 độ.
    """
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect).astype(np.float32)

    cx, cy = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    pts_sorted = pts[order]

    # Tìm góc Top-Left là đỉnh có tổng (x + y) nhỏ nhất
    s = pts_sorted[:, 0] + pts_sorted[:, 1]
    tl_idx = int(np.argmin(s))
    pts_ordered = np.roll(pts_sorted, -tl_idx, axis=0)

    # Đảm bảo theo chiều kim đồng hồ: tích có hướng (TR - TL) x (BL - TL) > 0
    v1 = pts_ordered[1] - pts_ordered[0]
    v2 = pts_ordered[3] - pts_ordered[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        # Đảo chiều nếu đang ngược chiều kim đồng hồ
        pts_ordered = pts_ordered[[0, 3, 2, 1]]

    return pts_ordered.astype(np.float32)


order_corners = order_points


# ==============================================================================
# KHỐI 3: PHÁT HIỆN 4 GÓC THẺ THAM CHIẾU (ID-1)
# ==============================================================================
def detect_card_corners(image, debug: bool = False, verbose: bool = False):
    """
    Khối 3 — Khối OpenCV phát hiện 4 góc thẻ tham chiếu ID-1:
    - Kích thước thẻ thực tế: 85.6 mm x 54.0 mm (tỷ lệ W/H ≈ 1.585)
    - Tiền xử lý: Grayscale -> GaussianBlur -> Canny edge detection -> Dilation đóng kín nét biên.
    - Tìm contours (cv2.findContours với RETR_EXTERNAL/RETR_TREE), xấp xỉ đa giác (approxPolyDP).
    - Lọc contour chuẩn:
      + Diện tích contour nằm trong khoảng 0.5% ~ 25% diện tích ảnh.
      + Tỷ lệ cạnh dài / cạnh ngắn của bounding box: 1.2 <= Aspect Ratio <= 1.9.
      + Sắp xếp 4 đỉnh theo thứ tự chuẩn: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    - Tích hợp fallback đa ngưỡng Canny & Adaptive Threshold giúp nhận diện chính xác tuyệt đối.
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        return {"success": False, "corners": None, "debug_image": None, "error_message": "Định dạng ảnh không hợp lệ"}

    if img is None:
        return {"success": False, "corners": None, "debug_image": None, "error_message": "Không thể đọc ảnh"}

    h_img, w_img = img.shape[:2]
    total_area = float(w_img * h_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    # Danh sách dải ngưỡng Canny khảo sát từ nhẹ đến sâu
    canny_thresh_pairs = [(30, 90), (45, 120), (20, 60), (50, 150), (15, 45)]
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for low, high in canny_thresh_pairs:
        edges = cv2.Canny(blur, low, high)
        dilated = cv2.dilate(edges, kernel_dilate, iterations=1)

        # RETR_TREE giúp bắt cả viền ngoài nếu thẻ nằm trên nền sàn có độ tương phản phức tạp
        for ret_mode in [cv2.RETR_EXTERNAL, cv2.RETR_TREE]:
            cnts, _ = cv2.findContours(dilated, ret_mode, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                # Tiêu chí: 0.5% <= diện tích <= 25% tổng diện tích ảnh
                if not (0.005 * total_area <= area <= 0.25 * total_area):
                    continue

                peri = cv2.arcLength(c, True)
                # Quét epsilon linh hoạt từ 0.02 đến 0.05
                for eps_ratio in [0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(c, eps_ratio * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        rect = cv2.minAreaRect(approx)
                        rw, rh = rect[1]
                        if min(rw, rh) < 15:
                            continue
                        aspect = max(rw, rh) / max(1.0, min(rw, rh))
                        # Tiêu chí tỷ lệ: 1.2 <= Aspect Ratio <= 1.9
                        if 1.20 <= aspect <= 1.90:
                            hull = cv2.convexHull(c)
                            hull_area = max(1.0, cv2.contourArea(hull))
                            solidity = area / hull_area
                            aspect_diff = abs(aspect - CARD_ASPECT_RATIO)
                            score = (1.0 / (1.0 + 4.0 * aspect_diff)) * 6.0 + solidity * 4.0
                            candidates.append({
                                "corners": order_points(approx.reshape(4, 2)),
                                "aspect": aspect,
                                "area": area,
                                "score": score,
                                "source": f"Canny ({low},{high}) + approxPolyDP"
                            })
                            break
            if candidates:
                break
        if candidates:
            break

    # Fallback 1: Phân ngưỡng thích nghi Adaptive Threshold
    if not candidates:
        ad = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
        ad_dilated = cv2.dilate(ad, kernel_dilate, iterations=1)
        cnts, _ = cv2.findContours(ad_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (0.005 * total_area <= area <= 0.25 * total_area):
                continue
            peri = cv2.arcLength(c, True)
            for eps_ratio in [0.025, 0.035, 0.045]:
                approx = cv2.approxPolyDP(c, eps_ratio * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    rect = cv2.minAreaRect(approx)
                    rw, rh = rect[1]
                    if min(rw, rh) < 15:
                        continue
                    aspect = max(rw, rh) / max(1.0, min(rw, rh))
                    if 1.20 <= aspect <= 1.90:
                        aspect_diff = abs(aspect - CARD_ASPECT_RATIO)
                        candidates.append({
                            "corners": order_points(approx.reshape(4, 2)),
                            "aspect": aspect,
                            "area": area,
                            "score": (1.0 / (1.0 + 4.0 * aspect_diff)) * 5.0,
                            "source": "Adaptive Threshold"
                        })
                        break

    # Fallback 2: Tách thẻ dựa trên kênh màu và ROI tối ưu
    if not candidates:
        b, g, r = cv2.split(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Bắt màu thẻ xanh hoặc màu tương phản trên sàn
        card_color_mask = (
            ((hsv[:, :, 0] >= 80) & (hsv[:, :, 0] <= 140) & (b.astype(int) - r.astype(int) > 4)) |
            (np.abs(b.astype(int) - r.astype(int)) > 25)
        ).astype(np.uint8) * 255
        card_color_mask = cv2.morphologyEx(card_color_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        cnts_col, _ = cv2.findContours(card_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_col:
            area = cv2.contourArea(c)
            if 300 <= area <= 0.20 * total_area:
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if min(rw, rh) > 15:
                    aspect = max(rw, rh) / max(1.0, min(rw, rh))
                    if 1.20 <= aspect <= 1.90:
                        box = cv2.boxPoints(rect).astype(np.float32)
                        candidates.append({
                            "corners": order_points(box),
                            "aspect": aspect,
                            "area": area,
                            "score": 3.0,
                            "source": "Color Saliency + minAreaRect"
                        })

    if not candidates:
        if verbose:
            print("[LỖI] Không tìm thấy 4 góc thẻ tham chiếu ID-1 trên ảnh!")
        debug_img = img.copy()
        cv2.putText(debug_img, "KHONG TIM THAY THE ID-1", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return {
            "success": False,
            "corners": None,
            "debug_image": debug_img,
            "aspect_ratio": None,
            "area": 0.0,
            "card_long_px": 0.0,
            "card_short_px": 0.0,
            "error_message": "Không phát hiện được thẻ tham chiếu ID-1"
        }

    # Chọn ứng viên có điểm số tốt nhất (tỷ lệ gần 1.585 nhất)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    ordered_corners = best["corners"]

    # Tính chiều dài và chiều rộng pixel trên ảnh gốc
    w_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    w_bot = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    h_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
    h_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])
    c_long_px = max((w_top + w_bot) / 2.0, (h_left + h_right) / 2.0)
    c_short_px = min((w_top + w_bot) / 2.0, (h_left + h_right) / 2.0)

    # Vẽ trực quan thẻ lên ảnh debug
    debug_img = img.copy()
    pts_int = np.round(ordered_corners).astype(np.int32)
    cv2.polylines(debug_img, [pts_int], True, (0, 255, 0), 2, cv2.LINE_AA)
    corner_labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
    for i, pt in enumerate(pts_int):
        cv2.circle(debug_img, tuple(pt), 6, colors[i], -1, cv2.LINE_AA)
        cv2.putText(debug_img, corner_labels[i], (pt[0] + 8, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 2, cv2.LINE_AA)

    cv2.putText(debug_img, f"ID-1: 85.6x54.0mm (Aspect: {best['aspect']:.2f})",
                (max(10, pts_int[0][0]), max(25, pts_int[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    return {
        "success": True,
        "corners": ordered_corners,
        "debug_image": debug_img,
        "aspect_ratio": best["aspect"],
        "area": best["area"],
        "card_long_px": float(c_long_px),
        "card_short_px": float(c_short_px),
        "error_message": ""
    }


detect_card = detect_card_corners


# ==============================================================================
# KHỐI 4: HIỆU CHỈNH PHỐI CẢNH (PERSPECTIVE TRANSFORM) & SO SÁNH TỶ LỆ
# ==============================================================================
def compare_scale_methods(card_corners, scale_homography: float = HOMOGRAPHY_SCALE_PX_PER_MM):
    """
    So sánh phương pháp tính tỷ lệ:
    1. Phương pháp đơn giản (không nắn):
       - L_px = trung bình 2 cạnh dài thẻ trên ảnh gốc
       - W_px = trung bình 2 cạnh ngắn thẻ trên ảnh gốc
       - scale_simple = 0.5 * (L_px / 85.6 + W_px / 54.0) (px/mm)
    2. Phương pháp nắn phối cảnh (Homography):
       - Tỷ lệ chuẩn sau nắn: scale_homography = 10 px/mm
    3. Đánh giá sai số phần trăm biến dạng góc chụp:
       - Tỷ lệ cạnh gốc Aspect_orig = L_px / W_px so với chuẩn 85.6 / 54.0 ~ 1.585
    """
    tl, tr, br, bl = card_corners
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))

    w_avg = (w_top + w_bot) / 2.0
    h_avg = (h_left + h_right) / 2.0

    L_px = max(w_avg, h_avg)
    W_px = min(w_avg, h_avg)

    scale_simple = 0.5 * ((L_px / CARD_LONG_MM) + (W_px / CARD_SHORT_MM))
    aspect_orig = L_px / max(1e-5, W_px)
    aspect_true = CARD_LONG_MM / CARD_SHORT_MM  # ~ 1.5852

    # Sai số biến dạng góc chụp (%)
    distortion_error_pct = abs(aspect_orig - aspect_true) / aspect_true * 100.0

    # Sai khác giữa tỷ lệ đơn giản trên ảnh gốc và tỷ lệ nắn phẳng (%)
    scale_diff_pct = abs(scale_simple - scale_homography) / scale_homography * 100.0

    return {
        "L_px": L_px,
        "W_px": W_px,
        "scale_simple": scale_simple,
        "scale_homography": scale_homography,
        "aspect_orig": aspect_orig,
        "aspect_true": aspect_true,
        "distortion_error_pct": distortion_error_pct,
        "scale_diff_pct": scale_diff_pct
    }


def rectify_perspective(image, card_corners, target_px_per_mm: float = HOMOGRAPHY_SCALE_PX_PER_MM):
    """
    Khối 4 — Hiệu chỉnh phối cảnh (Perspective Transform):
    - Độ phân giải chuẩn cố định S = 10 px/mm (thẻ sau khi nắn có kích thước 856 x 540 px).
    - Tính ma trận Homography H (cv2.getPerspectiveTransform).
    - Nắn phẳng toàn bộ bức ảnh (cv2.warpPerspective) bảo toàn trọn vẹn bàn chân.
    - Tỷ lệ mặt phẳng sau nắn: 10 px/mm (1 mm = 0.1 px).
    """
    h_orig, w_orig = image.shape[:2]
    tl, tr, br, bl = card_corners

    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    card_w_px = (w_top + w_bot) / 2.0
    card_h_px = (h_left + h_right) / 2.0

    # Nhận diện hướng thẻ trên ảnh (nằm ngang hay dọc)
    if card_w_px >= card_h_px:
        card_w_mm, card_h_mm = CARD_LONG_MM, CARD_SHORT_MM
    else:
        card_w_mm, card_h_mm = CARD_SHORT_MM, CARD_LONG_MM

    px_per_mm = float(target_px_per_mm)
    dst_card_w = card_w_mm * px_per_mm  # 856 px hoặc 540 px
    dst_card_h = card_h_mm * px_per_mm  # 540 px hoặc 856 px

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

    # Tìm bounding box của toàn bộ ảnh gốc sau khi nắn để không bị mất bàn chân
    orig_corners = np.array([
        [0.0, 0.0],
        [float(w_orig), 0.0],
        [float(w_orig), float(h_orig)],
        [0.0, float(h_orig)]
    ], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(orig_corners, H_card).reshape(-1, 2)
    x_min, y_min = np.min(warped_corners, axis=0)
    x_max, y_max = np.max(warped_corners, axis=0)

    # Ma trận tịnh tiến đưa toàn bộ tọa độ về phần dương
    T = np.array([
        [1.0, 0.0, -x_min],
        [0.0, 1.0, -y_min],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    H_total = T @ H_card
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))

    # Giới hạn kích thước tối đa để đảm bảo hiệu năng và bộ nhớ
    max_side = max(out_w, out_h)
    scale_down = 1.0
    if max_side > MAX_IMAGE_DIM:
        scale_down = MAX_IMAGE_DIM / float(max_side)
        S_scale = np.array([
            [scale_down, 0.0, 0.0],
            [0.0, scale_down, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        H_total = S_scale @ H_total
        out_w = int(np.ceil(out_w * scale_down))
        out_h = int(np.ceil(out_h * scale_down))
        px_per_mm = px_per_mm * scale_down

    out_w = max(50, min(MAX_IMAGE_DIM, out_w))
    out_h = max(50, min(MAX_IMAGE_DIM, out_h))

    # Nắn phẳng toàn cảnh
    rectified_img = cv2.warpPerspective(
        image,
        H_total,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    # Mặt nạ theo dõi vùng ảnh gốc hợp lệ
    orig_mask = np.ones((h_orig, w_orig), dtype=np.uint8) * 255
    valid_mask = cv2.warpPerspective(
        orig_mask,
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
        "transform_matrix": H_total,
        "valid_mask": valid_mask
    }


# ==============================================================================
# KHỐI 5: TÁCH MẶT NẠ BÀN CHÂN (SKIN COLOR + OTSU)
# ==============================================================================
def segment_foot(rectified_image, card_polygon, valid_mask=None, debug: bool = False):
    """
    Khối 5 — Tách mặt nạ bàn chân (Phương án A - Ngưỡng màu / Otsu trên nền sáng):
    - Thực hiện trên ảnh đã được nắn phối cảnh.
    - Chuyển sang không gian màu HSV và YCrCb để phân ngưỡng vùng da chân.
    - Kết hợp Otsu thresholding trên nền sáng.
    - Dùng toán tử hình thái học:
      + cv2.morphologyEx(..., MORPH_CLOSE) để lấp lỗ rỗng bên trong bàn chân.
      + cv2.morphologyEx(..., MORPH_OPEN) để khử nhiễu nền lốm đốm.
    - Loại bỏ vùng vị trí thẻ đã biết (Card Mask).
    - Tìm contour lớn nhất đại diện cho bàn chân để tạo mặt nạ nhị phân (Binary Mask).
    - Cắt eo cổ chân (Ankle Cut-off) khi chụp đứng chân kéo dài chạm viền ảnh.
    """
    if rectified_image is None or card_polygon is None:
        return None, None, None

    h, w = rectified_image.shape[:2]
    total_area = float(w * h)

    # 1. Tạo mặt nạ loại trừ vùng thẻ ID
    card_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(card_mask, [np.round(card_polygon).astype(np.int32)], 255)
    margin = max(15, int(min(w, h) * 0.025))
    card_mask = cv2.dilate(card_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (margin, margin)))

    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=np.uint8) * 255

    # 2. Phân ngưỡng vùng da chân trong không gian màu YCrCb & HSV
    ycrcb = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(rectified_image)

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    cond_ycrcb = (cr >= 133) & (cr <= 176) & (cb >= 80) & (cb <= 135)
    cond_hsv = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 2] >= 45)
    cond_bgr = (r.astype(int) - b.astype(int) > 6) & (r >= g)

    skin = (cond_ycrcb & cond_hsv & cond_bgr & (valid_mask > 0) & (card_mask == 0)).astype(np.uint8) * 255

    # 3. Kết hợp Otsu thresholding trên nền sáng (Inverted Otsu)
    inv_gray = cv2.bitwise_not(gray)
    inv_gray[card_mask > 0] = 0
    inv_gray[valid_mask == 0] = 0
    _, otsu_inv = cv2.threshold(inv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv[card_mask > 0] = 0
    otsu_inv[valid_mask == 0] = 0

    # Kết hợp vùng da và Otsu
    combined_skin = cv2.bitwise_or(skin, cv2.bitwise_and(otsu_inv, otsu_inv, mask=(r > b).astype(np.uint8) * 255))

    # 4. Toán tử hình thái học: MORPH_CLOSE (lấp lỗ rỗng) và MORPH_OPEN (khử nhiễu lốm đốm)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(combined_skin, cv2.MORPH_CLOSE, k_close)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, k_open)

    # 5. Tìm contour lớn nhất đại diện cho bàn chân
    cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.005 * total_area <= area <= 0.60 * total_area):
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if min(rw, rh) < 18:
            continue
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if 1.05 <= aspect <= 5.0:
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            overlap = np.logical_and(c_mask > 0, card_mask > 0).sum()
            if overlap / max(1.0, area) > 0.05:
                continue
            valid_cands.append((c, area))

    if not valid_cands:
        # Fallback phân ngưỡng trực tiếp Otsu trên ảnh xám
        cnts_otsu, _ = cv2.findContours(otsu_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_otsu:
            area = cv2.contourArea(c)
            if 0.008 * total_area <= area <= 0.60 * total_area:
                valid_cands.append((c, area))

    if not valid_cands:
        return None, None, None

    best_cnt = max(valid_cands, key=lambda x: x[1])[0]
    foot_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(foot_mask, [best_cnt], -1, 255, -1)

    # 6. Cắt eo cổ chân (Ankle Cut-off) tránh đo bắp chân khi chụp đứng
    y_idx, x_idx = np.where(foot_mask > 0)
    if len(x_idx) > 200:
        pts_xy = np.column_stack((x_idx, y_idx)).astype(np.float64)
        centroid = np.mean(pts_xy, axis=0)
        pts_centered = pts_xy - centroid
        cov = np.cov(pts_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sort_idx = np.argsort(eigenvalues)[::-1]
        v_major = eigenvectors[:, sort_idx[0]]
        v_minor = eigenvectors[:, sort_idx[1]]

        proj_major = pts_centered @ v_major
        p_min = float(np.min(proj_major))
        p_max = float(np.max(proj_major))

        # Phân biệt đầu ngón chân (bè rộng hơn) và gót chân/cổ chân
        pts_min_end = pts_xy[proj_major <= p_min + 0.15 * (p_max - p_min)]
        pts_max_end = pts_xy[proj_major >= p_max - 0.15 * (p_max - p_min)]
        min_touches_top = np.any(pts_min_end[:, 1] <= 15)

        w_min_slice, w_max_slice = [], []
        for t in np.linspace(p_min + 0.15 * (p_max - p_min), p_min + 0.35 * (p_max - p_min), 8):
            m = (proj_major >= t - 3) & (proj_major <= t + 3)
            if np.any(m):
                q = pts_centered[m] @ v_minor
                w_min_slice.append(np.ptp(q))
        for t in np.linspace(p_max - 0.35 * (p_max - p_min), p_max - 0.15 * (p_max - p_min), 8):
            m = (proj_major >= t - 3) & (proj_major <= t + 3)
            if np.any(m):
                q = pts_centered[m] @ v_minor
                w_max_slice.append(np.ptp(q))

        avg_w_min = np.mean(w_min_slice) if w_min_slice else 0
        avg_w_max = np.mean(w_max_slice) if w_max_slice else 0

        # Chuẩn hóa sao cho p_max là Ngón chân, p_min là Gót chân
        if not (min_touches_top or (avg_w_max > avg_w_min)):
            v_major = -v_major
            v_minor = -v_minor
            proj_major = pts_centered @ v_major
            p_min = float(np.min(proj_major))
            p_max = float(np.max(proj_major))

        c_edge1 = np.linalg.norm(card_polygon[1] - card_polygon[0])
        c_edge2 = np.linalg.norm(card_polygon[3] - card_polygon[0])
        mm_px_local = CARD_LONG_MM / max(1.0, max(c_edge1, c_edge2))
        span_mm = (p_max - p_min) * mm_px_local

        if min_touches_top or span_mm > 275.0:
            dist_heel_max_px = 290.0 / mm_px_local
            dist_heel_min_px = 210.0 / mm_px_local
            t_search_start = max(p_min, p_max - dist_heel_max_px)
            t_search_end = min(p_max - 120.0 / mm_px_local, p_max - dist_heel_min_px)

            best_t_cut = p_min
            min_w_scan = 999999.0
            scan_step = max(2.0, (t_search_end - t_search_start) / 30.0)

            for t_scan in np.arange(t_search_start, t_search_end, scan_step):
                m = (proj_major >= t_scan - 4.0) & (proj_major <= t_scan + 4.0)
                if np.any(m):
                    q = pts_centered[m] @ v_minor
                    w_curr = np.ptp(q)
                    if w_curr < min_w_scan:
                        min_w_scan = w_curr
                        best_t_cut = t_scan

            if best_t_cut > p_min:
                cut_cond = (proj_major < best_t_cut)
                foot_mask[pts_xy[cut_cond, 1].astype(int), pts_xy[cut_cond, 0].astype(int)] = 0
                cnts_refined, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_refined:
                    best_cnt = max(cnts_refined, key=cv2.contourArea)
                    foot_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(foot_mask, [best_cnt], -1, 255, -1)

    # Ảnh bàn chân giữ màu gốc trên nền trắng
    segmented_foot_img = np.full_like(rectified_image, 255)
    segmented_foot_img[foot_mask > 0] = rectified_image[foot_mask > 0]

    return foot_mask, segmented_foot_img, best_cnt


# ==============================================================================
# KHỐI 6: XÁC ĐỊNH TRỤC CHÍNH (PCA), ĐO ĐẠC KÍCH THƯỚC & PHÂN LOẠI DÁNG CHÂN
# ==============================================================================
def classify_foot_shape(cnt_pts, pt_heel, pt_toe, u_foot, v_foot, length_px, mm_per_px):
    """
    Phân loại dáng bàn chân dựa trên tương quan độ dài các ngón chân:
    - Bàn chân Ai Cập (Egyptian): Ngón cái dài nhất, dốc đều xuống các ngón sau.
    - Bàn chân Hy Lạp (Greek / Morton's toe): Ngón thứ hai dài nhất nhô cao hơn ngón cái.
    - Bàn chân La Mã (Roman): 3 ngón đầu có chiều dài ngang xấp xỉ bằng nhau tạo thành vòm phẳng.
    """
    proj_foot = (cnt_pts - pt_heel) @ u_foot
    # Lấy các điểm thuộc khu vực đầu ngón chân (top 15% phía trước)
    toe_region = cnt_pts[proj_foot >= 0.85 * length_px]
    if len(toe_region) < 5:
        return "Ai Cập (Egyptian)", "Dáng chân chuẩn phổ biến"

    q_toe = (toe_region - pt_heel) @ v_foot
    p_toe = (toe_region - pt_heel) @ u_foot

    max_idx = np.argmax(p_toe)
    peak_q = q_toe[max_idx]
    q_span = max(1e-5, float(np.ptp(q_toe)))
    q_norm = (peak_q - np.min(q_toe)) / q_span  # [0.0: mép ngoài, 0.5: chính giữa, 1.0: mép trong]

    # Kiểm tra vòm phẳng của ngón (La Mã)
    top_thresh = np.max(p_toe) - (2.5 / mm_per_px)
    top_pts_mask = p_toe >= top_thresh
    top_pts_span = np.ptp(q_toe[top_pts_mask]) * mm_per_px if np.any(top_pts_mask) else 0.0

    if np.sum(top_pts_mask) >= 12 and top_pts_span >= 30.0:
        return "La Mã (Roman)", "Ba ngón đầu bằng nhau tạo thành vòm ngang"
    elif 0.35 <= q_norm <= 0.65:
        return "Hy Lạp (Greek)", "Ngón trỏ (ngón thứ 2) dài hơn ngón cái"
    else:
        return "Ai Cập (Egyptian)", "Ngón cái dài nhất, các ngón sau ngắn dần"


def rotate_foot_vertical(image, foot_mask, centroid, v_major):
    """
    Xoay thẳng mặt nạ bàn chân theo phương thẳng đứng:
    - Gót chân ở phía dưới, đầu ngón chân ở phía trên.
    """
    h, w = image.shape[:2]
    # Góc quay để đưa v_major về phương thẳng đứng hướng lên (0, -1)
    angle_rad = np.arctan2(v_major[1], v_major[0])
    angle_deg = np.degrees(angle_rad)
    rot_angle = angle_deg + 90.0  # Xoay để trục hướng thẳng đứng lên trên

    M = cv2.getRotationMatrix2D(tuple(centroid), rot_angle, 1.0)
    rotated_mask = cv2.warpAffine(foot_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated_img, rotated_mask, M


def measure_foot_pca(rectified_image, foot_mask, foot_contour, mm_per_pixel: float):
    """
    Khối 6 — Xác định trục chính (PCA) và Đo đạc kích thước:
    - Lấy toàn bộ tọa độ pixel thuộc bàn chân trên mặt nạ.
    - Áp dụng PCA tìm:
      + Tâm bàn chân (Centroid)
      + Hướng trục chính (dọc chiều dài) và trục phụ (ngang chiều rộng).
    - Đo đạc:
      + Chiều dài bàn chân lớn nhất (mm): Từ đỉnh ngón chân xa nhất đến đáy gót chân dọc theo trục chính * mm/px.
      + Chiều rộng lớn nhất (mm): Độ rộng ngang đo tại 1/3 phía trước bàn chân (khu vực khớp ngón/mu bàn chân) * mm/px.
      + Chỉ số W/L: Width / Length.
      + Phân loại dáng chân: Ai Cập, La Mã, Hy Lạp.
    - Vẽ trực tiếp kết quả đo lên ảnh.
    """
    if foot_mask is None or foot_contour is None or mm_per_pixel <= 0:
        return None

    y_idx, x_idx = np.where(foot_mask > 0)
    if len(x_idx) < 150:
        return None

    pts_pca = np.column_stack((x_idx, y_idx)).astype(np.float32)

    # 1. Áp dụng cv2.PCACompute tìm trục chính và trục phụ
    mean, eigenvectors = cv2.PCACompute(pts_pca, mean=None, maxComponents=2)
    centroid = mean[0]         # Tâm bàn chân (Centroid)
    v_major = eigenvectors[0]  # Vector đơn vị trục chính (chiều dài)
    v_minor = eigenvectors[1]  # Vector đơn vị trục phụ (chiều rộng)

    proj_major = (pts_pca - centroid) @ v_major
    p_min = float(np.min(proj_major))
    p_max = float(np.max(proj_major))
    length_raw_px = p_max - p_min
    if length_raw_px <= 20.0:
        return None

    # Phân biệt đầu Gót (Heel) và đầu Ngón (Toe)
    s1 = (proj_major >= p_min) & (proj_major <= p_min + 0.30 * length_raw_px)
    s2 = (proj_major >= p_max - 0.30 * length_raw_px) & (proj_major <= p_max)
    w1 = float(np.ptp((pts_pca[s1] - centroid) @ v_minor)) if np.any(s1) else 0.0
    w2 = float(np.ptp((pts_pca[s2] - centroid) @ v_minor)) if np.any(s2) else 0.0

    # Chuẩn hóa để p_min là GÓT và p_max là NGÓN
    if w1 > w2:
        v_major = -v_major
        v_minor = -v_minor
        proj_major = (pts_pca - centroid) @ v_major
        p_min = float(np.min(proj_major))
        p_max = float(np.max(proj_major))

    cnt_pts = foot_contour.reshape(-1, 2).astype(np.float32)
    cnt_proj = (cnt_pts - centroid) @ v_major

    # Điểm ngón chân xa nhất (Toe tip) và điểm gót chân (Heel bottom)
    idx_toe = np.argmax(cnt_proj)
    pt_toe = cnt_pts[idx_toe]
    idx_heel = np.argmin(cnt_proj)
    pt_heel = cnt_pts[idx_heel]

    # Chiều dài bàn chân lớn nhất (mm)
    foot_length_px = float(np.linalg.norm(pt_toe - pt_heel))
    foot_length_mm = foot_length_px * mm_per_pixel

    # Chiều rộng lớn nhất tại 1/3 phía trước bàn chân (khớp nối ngón/mu bàn chân)
    u_foot = (pt_toe - pt_heel) / max(1e-5, foot_length_px)
    v_foot = np.array([-u_foot[1], u_foot[0]], dtype=np.float32)

    proj_foot = (cnt_pts - pt_heel) @ u_foot
    ball_mask = (proj_foot >= 0.58 * foot_length_px) & (proj_foot <= 0.85 * foot_length_px)

    if np.any(ball_mask):
        ball_pts = cnt_pts[ball_mask]
        q_vals = (ball_pts - pt_heel) @ v_foot
        q_min_idx = np.argmin(q_vals)
        q_max_idx = np.argmax(q_vals)
        pt_w1 = ball_pts[q_min_idx]
        pt_w2 = ball_pts[q_max_idx]
        foot_width_px = float(np.linalg.norm(pt_w2 - pt_w1))
    else:
        q_all = (cnt_pts - pt_heel) @ v_foot
        foot_width_px = float(np.ptp(q_all))
        pt_w1 = pt_heel + 0.65 * foot_length_px * u_foot + np.min(q_all) * v_foot
        pt_w2 = pt_heel + 0.65 * foot_length_px * u_foot + np.max(q_all) * v_foot

    foot_width_mm = foot_width_px * mm_per_pixel
    wl_ratio = foot_width_mm / max(1e-5, foot_length_mm)

    # Phân loại dáng chân
    foot_shape, shape_desc = classify_foot_shape(cnt_pts, pt_heel, pt_toe, u_foot, v_foot,
                                                 foot_length_px, mm_per_pixel)

    # Xoay thẳng mặt nạ bàn chân thẳng đứng
    rotated_img, rotated_mask, rot_M = rotate_foot_vertical(rectified_image, foot_mask, centroid, v_major)

    # 4. Vẽ trực quan lên ảnh kết quả
    vis = rectified_image.copy()
    cv2.drawContours(vis, [foot_contour], -1, (0, 220, 255), 2, cv2.LINE_AA)

    # Vẽ tâm và 2 trục PCA
    c_int = tuple(np.round(centroid).astype(int))
    cv2.circle(vis, c_int, 6, (0, 255, 0), -1, cv2.LINE_AA)
    p_major1 = tuple(np.round(centroid + v_major * 100).astype(int))
    p_major2 = tuple(np.round(centroid - v_major * 100).astype(int))
    cv2.line(vis, p_major1, p_major2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "Truc PCA Chinh", (p_major1[0] + 5, p_major1[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    p_minor1 = tuple(np.round(centroid + v_minor * 60).astype(int))
    p_minor2 = tuple(np.round(centroid - v_minor * 60).astype(int))
    cv2.line(vis, p_minor1, p_minor2, (0, 165, 255), 2, cv2.LINE_AA)

    # Đường đo chiều dài: từ gót đến đỉnh ngón xa nhất (màu đỏ)
    p_h = tuple(np.round(pt_heel).astype(int))
    p_t = tuple(np.round(pt_toe).astype(int))
    cv2.line(vis, p_h, p_t, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(vis, p_h, 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, p_t, 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(vis, "Got chan", (p_h[0] + 10, p_h[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Ngon dai nhat", (p_t[0] + 10, p_t[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

    # Đường đo chiều rộng: 1/3 trước bàn chân (màu xanh dương)
    pw1 = tuple(np.round(pt_w1).astype(int))
    pw2 = tuple(np.round(pt_w2).astype(int))
    cv2.line(vis, pw1, pw2, (255, 120, 0), 3, cv2.LINE_AA)
    cv2.circle(vis, pw1, 5, (255, 120, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, pw2, 5, (255, 120, 0), -1, cv2.LINE_AA)
    cv2.putText(vis, "Rong 1/3 truoc", (pw2[0] + 8, pw2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 120, 0), 1, cv2.LINE_AA)

    # Bảng HUD thông số kích thước và phân loại
    banner_w = min(560, int(vis.shape[1] * 0.95))
    cv2.rectangle(vis, (12, 12), (12 + banner_w, 150), (20, 20, 20), -1)
    cv2.rectangle(vis, (12, 12), (12 + banner_w, 150), (0, 220, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "KET QUA DO KICH THUOC BAN CHAN (OPENCV + PCA)", (22, 34), font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Chieu dai (Got -> Ngon dai nhat): {foot_length_mm:.1f} mm ({foot_length_mm/10.0:.2f} cm)", (22, 60), font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Chieu rong (1/3 phia truoc)     : {foot_width_mm:.1f} mm ({foot_width_mm/10.0:.2f} cm)", (22, 84), font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Ty so Rong / Dai (W/L Ratio)   : {wl_ratio:.3f}", (22, 108), font, 0.45, (180, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Dang ban chan                  : {foot_shape}", (22, 130), font, 0.45, (255, 200, 80), 1, cv2.LINE_AA)

    return {
        "foot_length_px": round(float(foot_length_px), 2),
        "foot_width_px": round(float(foot_width_px), 2),
        "foot_length_mm": round(float(foot_length_mm), 2),
        "foot_width_mm": round(float(foot_width_mm), 2),
        "foot_length_cm": round(float(foot_length_mm / 10.0), 2),
        "foot_width_cm": round(float(foot_width_mm / 10.0), 2),
        "wl_ratio": round(float(wl_ratio), 3),
        "foot_shape": foot_shape,
        "shape_desc": shape_desc,
        "centroid": centroid,
        "v_major": v_major,
        "v_minor": v_minor,
        "pt_heel": pt_heel,
        "pt_toe": pt_toe,
        "pt_w1": pt_w1,
        "pt_w2": pt_w2,
        "measurement_image": vis,
        "rotated_image": rotated_img,
        "rotated_mask": rotated_mask
    }


# ==============================================================================
# KHỐI 7: KẾT QUẢ ĐẦU RA & XỬ LÝ TOÀN TRÌNH 1 ẢNH (PIPELINE)
# ==============================================================================
def process_image_pipeline(image_path, output_base_dir=None, split_name: str = "test",
                           save_images: bool = True, show_window: bool = False):
    """
    Xử lý trọn gói 1 ảnh qua 7 khối pipeline:
    1. Đọc ảnh
    2. Đánh giá chất lượng ảnh
    3. Phát hiện 4 góc thẻ ID-1
    4. Nắn phối cảnh Homography & So sánh tỷ lệ
    5. Tách mặt nạ da chân
    6. Đo đạc PCA & Phân loại dáng chân
    7. Vẽ trực quan, in báo cáo terminal chi tiết và xuất file
    """
    img_p = Path(image_path)
    stem = img_p.stem

    record = {
        "Ten_anh": img_p.name,
        "Tap_du_lieu": split_name,
        "Trang_thai": "Thất bại",
        "Do_sac_net": 0.0,
        "Do_sang": 0.0,
        "The_ID_tim_thay": "KHÔNG",
        "Tach_chan_thanh_cong": "KHÔNG",
        "Chieu_dai_mm": "",
        "Chieu_rong_mm": "",
        "Chieu_dai_cm": "",
        "Chieu_rong_cm": "",
        "Ty_le_W_L": "",
        "Dang_ban_chan": "",
        "Ty_le_Homography_px_mm": "",
        "Ty_le_Don_gian_px_mm": "",
        "Sai_so_goc_chup_pct": "",
        "error_message": "",
        "vis_image": None
    }

    print("\n" + "=" * 78)
    print(f"BÁO CÁO XỬ LÝ ẢNH: {img_p.name}")
    print(f"Đường dẫn: {img_p}")
    print("-" * 78)

    # 1. Đọc ảnh
    img = cv2.imread(str(img_p))
    if img is None:
        record["error_message"] = "Không thể đọc file ảnh"
        print(f"[LỖI] {record['error_message']}")
        return record

    # 2. Khối 2: Kiểm tra chất lượng ảnh
    qa = assess_image_quality(img)
    record["Do_sac_net"] = round(qa["sharpness"], 2)
    record["Do_sang"] = round(qa["brightness"], 2)
    print(f"1. KIỂM TRA CHẤT LƯỢNG ẢNH:")
    print(f"   - Độ sắc nét (Laplacian var): {qa['sharpness']:.1f} (Yêu cầu >= 80)")
    print(f"   - Độ sáng trung bình       : {qa['brightness']:.1f} (Yêu cầu [50, 215])")

    if not qa["is_valid"]:
        record["error_message"] = qa["error_message"]
        print(f"   => [BỎ QUA / YÊU CẦU CHỤP LẠI]: {qa['error_message']}")
        return record
    print(f"   => Chất lượng ảnh ĐẠT TIÊU CHUẨN.")

    # 3. Khối 3: Phát hiện 4 góc thẻ tham chiếu (ID-1)
    card_res = detect_card(img, verbose=False)
    if not card_res["success"]:
        record["error_message"] = card_res["error_message"]
        print(f"2. PHÁT HIỆN THẺ ID-1:")
        print(f"   => [THẤT BẠI]: {card_res['error_message']}")
        return record

    record["The_ID_tim_thay"] = "CÓ"
    card_corners = card_res["corners"]
    print(f"2. PHÁT HIỆN THẺ ID-1:")
    print(f"   - 4 góc thẻ (TL, TR, BR, BL):")
    for name, pt in zip(["TL", "TR", "BR", "BL"], card_corners):
        print(f"     + {name}: ({pt[0]:.1f}, {pt[1]:.1f})")
    print(f"   - Tỷ lệ cạnh thẻ đo được: {card_res['aspect_ratio']:.3f} (Chuẩn ISO: ~1.585)")

    # 4. Khối 4: Hiệu chỉnh phối cảnh (Homography) & So sánh tỷ lệ
    rect_res = rectify_perspective(img, card_corners, target_px_per_mm=HOMOGRAPHY_SCALE_PX_PER_MM)
    scale_comp = compare_scale_methods(card_corners, scale_homography=rect_res["pixels_per_mm"])

    record["Ty_le_Homography_px_mm"] = round(scale_comp["scale_homography"], 3)
    record["Ty_le_Don_gian_px_mm"] = round(scale_comp["scale_simple"], 3)
    record["Sai_so_goc_chup_pct"] = round(scale_comp["distortion_error_pct"], 2)

    print(f"3. HIỆU CHỈNH PHỐI CẢNH & SO SÁNH TỶ LỆ:")
    print(f"   - Phương pháp Nắn phối cảnh (Homography) : {scale_comp['scale_homography']:.2f} px/mm (1 mm = {1.0/scale_comp['scale_homography']:.3f} px)")
    print(f"   - Phương pháp Tính tỷ lệ đơn giản       : {scale_comp['scale_simple']:.2f} px/mm (L_px={scale_comp['L_px']:.1f}, W_px={scale_comp['W_px']:.1f})")
    print(f"   - Sai lệch tỷ lệ giữa 2 phương pháp     : {scale_comp['scale_diff_pct']:.2f} %")
    print(f"   - Sai số biến dạng góc chụp (Perspective): {scale_comp['distortion_error_pct']:.2f} %")

    # 5. Khối 5: Tách mặt nạ bàn chân
    rectified_img = rect_res["rectified_image"]
    card_polygon = rect_res["card_polygon"]
    valid_mask = rect_res["valid_mask"]

    foot_mask, segmented_foot_img, foot_cnt = segment_foot(rectified_img, card_polygon, valid_mask)
    if foot_mask is None or foot_cnt is None:
        record["error_message"] = "Không thể tách mặt nạ bàn chân khỏi nền sàn"
        print(f"4. TÁCH MẶT NẠ BÀN CHÂN:")
        print(f"   => [THẤT BẠI]: {record['error_message']}")
        return record

    record["Tach_chan_thanh_cong"] = "CÓ"
    print(f"4. TÁCH MẶT NẠ BÀN CHÂN: THÀNH CÔNG (HSV/YCrCb + Inverted Otsu + Morphology)")

    # 6. Khối 6: Xác định trục chính (PCA), Đo đạc & Phân loại dáng chân
    meas = measure_foot_pca(rectified_img, foot_mask, foot_cnt, rect_res["mm_per_pixel"])
    if meas is None:
        record["error_message"] = "Lỗi khi áp dụng PCA đo đạc kích thước bàn chân"
        print(f"5. ĐO ĐẠC BÀN CHÂN BẰNG PCA:")
        print(f"   => [THẤT BẠI]: {record['error_message']}")
        return record

    record["Trang_thai"] = "Thành công"
    record["Chieu_dai_mm"] = meas["foot_length_mm"]
    record["Chieu_rong_mm"] = meas["foot_width_mm"]
    record["Chieu_dai_cm"] = meas["foot_length_cm"]
    record["Chieu_rong_cm"] = meas["foot_width_cm"]
    record["Ty_le_W_L"] = meas["wl_ratio"]
    record["Dang_ban_chan"] = meas["foot_shape"]
    record["vis_image"] = meas["measurement_image"]

    print(f"5. KẾT QUẢ ĐO ĐẠC BÀN CHÂN (PCA):")
    print(f"   - Chiều dài lớn nhất (Gót -> Ngón): {meas['foot_length_mm']:.1f} mm ({meas['foot_length_cm']:.2f} cm | {meas['foot_length_px']:.1f} px)")
    print(f"   - Chiều rộng lớn nhất (1/3 trước) : {meas['foot_width_mm']:.1f} mm ({meas['foot_width_cm']:.2f} cm | {meas['foot_width_px']:.1f} px)")
    print(f"   - Chỉ số tỷ lệ bề ngang (W/L)     : {meas['wl_ratio']:.3f}")
    print(f"   - Phân loại dáng chân             : {meas['foot_shape']} ({meas['shape_desc']})")
    print("=" * 78)

    # 7. Xuất file kết quả nếu có thư mục output
    if save_images and output_base_dir:
        out_dir = Path(output_base_dir) / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}_ket_qua_do.jpg"), meas["measurement_image"])
        cv2.imwrite(str(out_dir / f"{stem}_xoay_dung.jpg"), meas["rotated_image"])
        cv2.imwrite(str(out_dir / f"{stem}_the_id.jpg"), card_res["debug_image"])

    return record


# ==============================================================================
# VÒNG LẶP DUYỆT TẬP DỮ LIỆU VỚI PHÍM TẮT CHUYỂN TIẾP / DỪNG
# ==============================================================================
def run_dataset_loop(split: str = "test", dataset_dir: Path = None, output_dir: Path = None,
                     show_window: bool = True, save_images: bool = True):
    """
    Duyệt toàn bộ tập dữ liệu, hiển thị kết quả đo:
    - Bấm bất kỳ phím nào để xem ảnh tiếp theo.
    - Bấm 'q' hoặc 'Q' để dừng vòng lặp.
    - Xuất bảng CSV tổng hợp vào output_dir.
    """
    image_paths = load_dataset(split=split, dataset_dir=dataset_dir)
    n_imgs = len(image_paths)

    print("\n" + "#" * 80)
    print("CHƯƠNG TRÌNH ĐO BÀN CHÂN VỚI THẺ THAM CHIẾU ID-1 (OPENCV)")
    print(f"- Tập dữ liệu đang chạy: {split.upper()} ({n_imgs} ảnh)")
    print(f"- Phím tắt tương tác   : Nhấn 'q' để dừng, phím bất kỳ để tiếp tục ảnh kế")
    print("#" * 80)

    if n_imgs == 0:
        print(f"[THÔNG BÁO] Không tìm thấy ảnh nào trong tập {split}!")
        return []

    all_records = []
    success_count = 0
    start_time = time.time()

    win_name = "Ket qua do ban chan (Nhan 'q' de thoat, phim bat ky de qua anh)"
    if show_window:
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 900, 700)
        except Exception:
            show_window = False

    for idx, p in enumerate(image_paths, 1):
        print(f"\n>>> Đang xử lý [{idx}/{n_imgs}]: {os.path.basename(p)}")
        rec = process_image_pipeline(p, output_base_dir=output_dir, split_name=split,
                                     save_images=save_images, show_window=show_window)
        all_records.append(rec)
        if rec["Trang_thai"] == "Thành công":
            success_count += 1

        # Hiển thị cửa sổ GUI tương tác
        if show_window and rec.get("vis_image") is not None:
            cv2.imshow(win_name, rec["vis_image"])
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[NGƯỜI DÙNG DỪNG]: Đã nhận phím 'q'. Dừng vòng lặp duyệt ảnh.")
                break

    if show_window:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    elapsed = time.time() - start_time
    rate = (success_count / max(1, len(all_records))) * 100.0

    print("\n" + "=" * 80)
    print("              TỔNG KẾT ĐỢT CHẠY THỬ NGHIỆM")
    print("=" * 80)
    print(f"- Tổng số ảnh đã duyệt   : {len(all_records)} / {n_imgs}")
    print(f"- Số ảnh đo thành công   : {success_count}")
    print(f"- Tỷ lệ thành công       : {rate:.2f} %")
    print(f"- Thời gian thực hiện    : {elapsed:.2f} giây")

    # Xuất file CSV báo cáo kết quả
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        csv_file = out_p / f"bang_ket_qua_do_{split}.csv"
        fieldnames = [
            "Ten_anh", "Tap_du_lieu", "Trang_thai", "Do_sac_net", "Do_sang",
            "The_ID_tim_thay", "Tach_chan_thanh_cong",
            "Chieu_dai_mm", "Chieu_rong_mm", "Chieu_dai_cm", "Chieu_rong_cm",
            "Ty_le_W_L", "Dang_ban_chan", "Ty_le_Homography_px_mm",
            "Ty_le_Don_gian_px_mm", "Sai_so_goc_chup_pct", "error_message"
        ]
        with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_records)
        print(f"- Bảng kết quả CSV       : {csv_file}")
    print("=" * 80 + "\n")

    return all_records


def run_batch_pipeline(dataset_dir=None, output_dir=None, splits=None, debug=False):
    """Hàm tương thích cho batch pipeline."""
    if splits is None:
        splits = ["test"]
    elif isinstance(splits, str):
        splits = [splits]
    for sp in splits:
        run_dataset_loop(split=sp, dataset_dir=dataset_dir, output_dir=output_dir, show_window=False)


# ==============================================================================
# HÀM CHÍNH (MAIN)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Chương trình đo kích thước bàn chân với thẻ tham chiếu ID-1")
    parser.add_argument("image_pos", nargs="?", default=None, help="Đường dẫn file ảnh lẻ cần xử lý")
    parser.add_argument("--image", "-i", type=str, default=None, help="Đường dẫn file ảnh lẻ cần xử lý")
    parser.add_argument("--split", "-s", type=str, choices=["train", "valid", "test", "all"],
                        default="test", help="Tập dữ liệu cần chạy (mặc định 'test')")
    parser.add_argument("--data", "-d", type=str, default=None, help="Thư mục chứa dữ liệu")
    parser.add_argument("--output", "-o", type=str, default=None, help="Thư mục xuất kết quả (mặc định outputs/)")
    parser.add_argument("--no-gui", action="store_true", default=False, help="Chạy không mở cửa sổ cv2.imshow (batch mode)")
    parser.add_argument("--debug", action="store_true", default=False, help="Bật chế độ debug")

    args = parser.parse_args()
    project_root = get_project_root()
    target_img = args.image or args.image_pos

    out_dir = Path(args.output).resolve() if args.output else (project_root / "outputs")

    # Trường hợp 1: Chạy ảnh đơn lẻ
    if target_img:
        img_p = Path(target_img).resolve()
        if not img_p.exists():
            print(f"[LỖI] Không tìm thấy file ảnh: {target_img}")
            return
        rec = process_image_pipeline(img_p, output_base_dir=out_dir, split_name="single",
                                     save_images=True, show_window=not args.no_gui)
        if not args.no_gui and rec.get("vis_image") is not None:
            cv2.imshow("Ket qua do anh don le", rec["vis_image"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Trường hợp 2: Chạy tập dữ liệu (mặc định 'test')
        data_dir = Path(args.data).resolve() if args.data else (project_root / "data" / "bodulieu")
        splits_to_run = ["train", "valid", "test"] if args.split == "all" else [args.split]
        for sp in splits_to_run:
            run_dataset_loop(split=sp, dataset_dir=data_dir, output_dir=out_dir,
                             show_window=not args.no_gui, save_images=True)


if __name__ == "__main__":
    main()
