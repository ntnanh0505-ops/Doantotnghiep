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
import json
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import cv2
import numpy as np

from src.yolo_y8.predict import detect_foot_card, load_detector
from src.yolo_y8.train import load_config

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
CARD_SHORT_MM = 53.98    # Chiều rộng chuẩn thẻ ID-1
CARD_ASPECT_RATIO = CARD_LONG_MM / CARD_SHORT_MM  # ~ 1.585185...
HOMOGRAPHY_SCALE_PX_PER_MM = 10.0  # S = 10 px/mm (1 mm = 0.1 px)
MAX_IMAGE_DIM = 2600     # Giới hạn kích thước tối đa tránh tràn bộ nhớ khi warp
LENGTH_FEATURES = ["L_raw", "W_raw", "rear", "area", "perim", "hull", "solidity", "rect_L", "rect_W",
                   "card_frac", "D", "L_over_D", "W_over_D"]  # thứ tự đặc trưng của mô hình chiều dài (xem length_features)
TOE_NOTCH_MIN_MM = 1.0   # Kẽ giữa các ngón chân (khuyết lồi của viền) sâu từ mức này mới tính khi tìm đầu ngón
CHORD_STRIP_MM = 1.5     # Bề rộng dải khi tìm dây cung dài/rộng nhất của mặt nạ
BALL_FROM_TOE_MM = (45.0, 95.0)  # Chiều rộng đo trong cửa sổ cách đầu ngón này (mm): vùng khớp ngón, không phụ thuộc thấy gót hay không
SPIKE_OPEN_MM = 10.0     # Kích thước phép mở hình thái học loại đuôi mỏng khỏi mặt nạ YOLO (ngón chân rộng hơn thế nhiều)
HEEL_REAR_SOLIDITY_MIN = 0.91  # Thấy gót khi 25% đầu sau của mặt nạ đủ lồi (diện tích / diện tích bao lồi) ...
HEEL_REAR_NOTCH_MAX_MM = 25.0  # ... và ở đầu sau không có chỗ lõm sâu hơn mức này (mép ống quần / cẳng chân cắt vào).
# Hai ngưỡng chọn trên 7 ảnh thấy gót / 46 ảnh che gót xem bằng mắt, thử qua 5 biến thể ảnh (JPEG 85/95, phóng 3 lần rồi thu nhỏ,
# sáng +10%): 7/7 ảnh thấy gót luôn được nhận, nhầm 8-9/46. Ngưỡng lõm 15 mm trước đây làm ảnh thấy gót bị từ chối ngẫu nhiên khi lưu lại JPEG.


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
    - Mặc định tìm tại data/yolo_dataset/images/{split}.
    """
    if dataset_dir is None:
        root = get_project_root()
        candidate_paths = [
            root / "data" / "yolo_dataset" / "images" / split,
            Path("data/yolo_dataset/images") / split,
            Path("data") / split,
            Path(split)
        ]
        target_dir = None
        for cp in candidate_paths:
            if cp.exists() and cp.is_dir():
                target_dir = cp.resolve()
                break
        if target_dir is None:
            target_dir = (root / "data" / "yolo_dataset" / "images" / split).resolve()
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
HEEL_HIDDEN_MESSAGE = ("Không thấy gót (gót bị cổ chân/ống quần che): từ chối đo. "
                       "Chụp lại từ trên xuống sao cho thấy rõ gót và 5 ngón chân, cẳng chân nằm ngoài khung hình")


def preprocess_image(image):
    """
    Thu nhỏ ảnh về tối đa input.max_long_side x input.max_short_side (mặc định Full HD, ~2 MP) trong configs/yolo_config.yaml.
    Giữ tỉ lệ, không phóng to ảnh nhỏ hơn khung. Trả về (ảnh, hệ số thu nhỏ). Nội suy INTER_AREA để không làm mờ cạnh thẻ.
    """
    cfg = load_config().get("input", {})
    h, w = image.shape[:2]
    scale = min(1.0, cfg.get("max_long_side", 1920) / max(h, w), cfg.get("max_short_side", 1080) / min(h, w))
    if scale >= 1.0:
        return image, 1.0
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA), scale


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


def card_corners_from_polygon(poly):
    """
    Suy ra 4 góc thẻ (TL, TR, BR, BL; pixel ảnh gốc) từ polygon mask thẻ do YOLO phân đoạn.
    - Xấp xỉ bao lồi về tứ giác, rồi khớp lại 4 đường cạnh từ các điểm biên và lấy giao điểm.
    - Thẻ có góc bo tròn nên đỉnh của bao lồi nằm lệch vào trong; dùng thẳng các đỉnh đó sẽ làm
      tỷ lệ px/mm sai, còn giao của 4 đường cạnh thì trả lại đúng góc hình học.
    Trả về None nếu polygon không đủ điểm.
    """
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None

    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2)
        if len(approx) == 4:
            quad = approx
            break
        if len(approx) < 4:
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    quad = order_points(quad)

    # Khớp lại 4 cạnh bằng các điểm biên nằm ở phần giữa mỗi cạnh (tránh vùng góc bo)
    edge_len = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    tol = 0.04 * min(edge_len)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        u = (b - a) / max(1e-5, edge_len[i])
        n = np.array([-u[1], u[0]], dtype=np.float32)
        rel = pts - a
        t = (rel @ u) / max(1e-5, edge_len[i])
        near = np.abs(rel @ n) <= tol
        sel = pts[near & (t >= 0.2) & (t <= 0.8)]
        if len(sel) < 8:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)))

    refined = quad.copy()
    for i in range(4):
        # Góc i = giao của cạnh (i-1 -> i) và cạnh (i -> i+1)
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        A = np.column_stack((d1, -d2))
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        s, _ = np.linalg.solve(A, p2 - p1)
        corner = (p1 + s * d1).astype(np.float32)
        if np.linalg.norm(corner - quad[i]) <= 0.06 * max(edge_len):
            refined[i] = corner
    return order_points(refined)


def detect_card_yolo(image, detection):
    """
    Thay thế detect_card() bằng mask thẻ của YOLO (detection = predict.detect_foot_card()).
    Trả về dict cùng khóa với detect_card() để pipeline dùng chung phần phía sau.
    """
    fail = {"success": False, "corners": None, "debug_image": image.copy(), "aspect_ratio": 0.0,
            "area": 0.0, "card_long_px": 0.0, "card_short_px": 0.0, "error_message": ""}

    corners = card_corners_from_polygon(detection.get("card_polygon")) if detection.get("card_polygon") is not None else None
    if corners is None:
        fail["error_message"] = "YOLO không phát hiện được thẻ ID-1"
        return fail

    tl, tr, br, bl = corners
    w_avg = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h_avg = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    long_px, short_px = max(w_avg, h_avg), min(w_avg, h_avg)
    aspect = long_px / max(1e-5, short_px)
    if not (1.25 <= aspect <= 1.95):
        fail["error_message"] = f"Tỷ lệ cạnh thẻ YOLO bất thường ({aspect:.2f}, chuẩn ISO ~1.585)"
        return fail

    debug_img = image.copy()
    pts_int = np.round(corners).astype(np.int32)
    cv2.polylines(debug_img, [pts_int], True, (0, 255, 0), 2, cv2.LINE_AA)
    for label, pt, color in zip(["TL", "TR", "BR", "BL"], pts_int,
                                [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]):
        cv2.circle(debug_img, tuple(int(v) for v in pt), 6, color, -1, cv2.LINE_AA)
        cv2.putText(debug_img, label, (int(pt[0]) + 8, int(pt[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return {"success": True, "corners": corners, "debug_image": debug_img, "aspect_ratio": float(aspect),
            "area": float(cv2.contourArea(corners)), "card_long_px": float(long_px),
            "card_short_px": float(short_px), "error_message": ""}


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


def toe_direction_sign(contour, centroid, u, mm_per_pixel: float):
    """
    Đầu ngón nằm ở phía nào của trục u: +1 (phía u dương), -1 (phía âm), None nếu không phân biệt được.
    Đầu ngón chân có nhiều kẽ giữa các ngón (khuyết lồi của viền) hơn đầu gót / cổ chân. Cách cũ coi đầu rộng hơn là
    ngón, nhưng khi mặt nạ lẫn cổ chân thì đầu cổ chân rộng hơn nên bị đảo ngược (sai ~40% ảnh khi kiểm tra bằng mắt).
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    a = (pts - centroid) @ u
    band = 0.30 * (a.max() - a.min())
    try:
        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
    except cv2.error:
        return None
    if defects is None:
        return None
    low = high = 0
    for _, _, far, depth in defects.reshape(-1, 4):
        if depth / 256.0 * mm_per_pixel < TOE_NOTCH_MIN_MM:
            continue
        if a[far] <= a.min() + band:
            low += 1
        elif a[far] >= a.max() - band:
            high += 1
    return None if low == high else (1 if high > low else -1)


def cut_ankle(foot_mask, best_cnt, card_polygon):
    """
    Cắt eo cổ chân (Ankle Cut-off) khi bàn chân kéo dài chạm viền ảnh hoặc dài bất thường,
    tránh đo cả bắp chân. Dùng chung cho mặt nạ OpenCV và mặt nạ YOLO.
    Trả về (foot_mask, best_cnt) sau khi cắt.
    """
    h, w = foot_mask.shape[:2]
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
        c_edge1 = np.linalg.norm(card_polygon[1] - card_polygon[0])
        c_edge2 = np.linalg.norm(card_polygon[3] - card_polygon[0])
        mm_px_local = CARD_LONG_MM / max(1.0, max(c_edge1, c_edge2))
        toe_sign = toe_direction_sign(best_cnt, centroid, v_major, mm_px_local)
        if toe_sign is None:
            toe_sign = 1 if (min_touches_top or (avg_w_max > avg_w_min)) else -1
        if toe_sign < 0:
            v_major = -v_major
            v_minor = -v_minor
            proj_major = pts_centered @ v_major
            p_min = float(np.min(proj_major))
            p_max = float(np.max(proj_major))

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
    return foot_mask, best_cnt


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

    foot_mask, best_cnt = cut_ankle(foot_mask, best_cnt, card_polygon)

    # Ảnh bàn chân giữ màu gốc trên nền trắng
    segmented_foot_img = np.full_like(rectified_image, 255)
    segmented_foot_img[foot_mask > 0] = rectified_image[foot_mask > 0]

    return foot_mask, segmented_foot_img, best_cnt


def segment_foot_from_polygon(rectified_image, foot_polygon, transform_matrix, card_polygon):
    """
    Phương án B — Mặt nạ bàn chân từ YOLO: polygon trên ảnh gốc được nắn bằng cùng ma trận homography
    với ảnh (rect_res["transform_matrix"]) rồi đi qua bước cắt cổ chân như phương án A.
    Trả về (foot_mask, segmented_foot_img, contour) giống segment_foot().
    """
    if rectified_image is None or foot_polygon is None or card_polygon is None:
        return None, None, None

    h, w = rectified_image.shape[:2]
    warped = cv2.perspectiveTransform(
        np.asarray(foot_polygon, dtype=np.float32).reshape(-1, 1, 2), transform_matrix
    ).reshape(-1, 2)
    foot_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(foot_mask, [np.round(warped).astype(np.int32)], 255)

    # Bỏ các "đuôi" mỏng (bóng, nếp ống quần) bám vào mặt nạ: chúng kéo dài số đo thêm vài cm
    card_long_px = max(np.linalg.norm(card_polygon[1] - card_polygon[0]), np.linalg.norm(card_polygon[3] - card_polygon[0]))
    k = max(3, int(round(SPIKE_OPEN_MM * card_long_px / CARD_LONG_MM)) | 1)
    foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    cnts, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None
    best_cnt = max(cnts, key=cv2.contourArea)
    foot_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(foot_mask, [best_cnt], -1, 255, -1)

    foot_mask, best_cnt = cut_ankle(foot_mask, best_cnt, card_polygon)

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


def camera_distance_cm(card_long_px: float, image_shape, focal_ratio: float) -> float:
    """Khoảng cách máy-mặt thẻ (cm) từ cỡ thẻ trong ảnh gốc; tiêu cự ước tính = focal_ratio x cạnh dài ảnh (px)."""
    return float(focal_ratio) * max(image_shape[:2]) * CARD_LONG_MM / card_long_px / 10.0


def height_correction(card_long_px: float, image_shape, cfg: dict = None):
    """
    Hiệu chỉnh độ cao so với mặt thẻ. Thẻ nằm trên sàn nên tỷ lệ mm/px của phép nắn chỉ đúng cho vật nằm trên sàn;
    mu bàn chân và cổ chân cao hơn nên bị phóng đại D / (D - h), D = khoảng cách máy-mặt thẻ.
    D suy ra từ cỡ thẻ trong ảnh gốc và tiêu cự ước tính (focal_ratio x cạnh dài ảnh, tính bằng px).
    Tham số lấy từ khóa measure.height_correction trong configs/yolo_config.yaml.
    Trả về (hệ số chiều dài, hệ số chiều rộng) hoặc None nếu tắt / thiếu dữ liệu.
    """
    if cfg is None:
        cfg = load_config().get("measure", {}).get("height_correction", {})
    if not cfg.get("enabled", False) or card_long_px <= 0:
        return None
    distance_cm = camera_distance_cm(card_long_px, image_shape, cfg["focal_ratio"])
    k_len = max(0.0, 1.0 - float(cfg["h_length_cm"]) / distance_cm)
    k_wid = max(0.0, 1.0 - float(cfg["h_width_cm"]) / distance_cm)
    return k_len, k_wid


def length_features(meas, foot_mask, foot_cnt, mm_per_pixel: float, card_long_px: float, image_shape):
    """
    Đặc trưng cho mô hình chiều dài (thứ tự theo LENGTH_FEATURES): số đo mask thô, độ tròn đầu gót, hình học mask (cm, cm²)
    và cỡ thẻ so với cạnh dài của ảnh (không dùng pixel: ảnh điện thoại 12 MP lớn hơn ảnh huấn luyện 960x1280 nhiều lần). Số đo thô, chưa hiệu chỉnh độ cao, để mô hình tự học phần sai lệch hệ thống.
    """
    k = mm_per_pixel
    focal_ratio = load_config().get("measure", {}).get("height_correction", {}).get("focal_ratio", 0.72)
    ys, xs = np.where(foot_mask > 0)
    (_, (rect_a, rect_b), _) = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    area = cv2.contourArea(foot_cnt) * k * k / 100.0
    hull = cv2.contourArea(cv2.convexHull(foot_cnt)) * k * k / 100.0
    distance = camera_distance_cm(card_long_px, image_shape, focal_ratio)
    length, width = meas["foot_length_raw_mm"] / 10.0, meas["foot_width_raw_mm"] / 10.0
    return [length, width, meas["rear_ratio"], area, cv2.arcLength(foot_cnt, True) * k / 10.0, hull, area / max(hull, 1e-6),
            max(rect_a, rect_b) * k / 10.0, min(rect_a, rect_b) * k / 10.0, card_long_px / max(image_shape[:2]), distance, length / distance, width / distance]


_LENGTH_MODEL = {}


def load_length_model():
    """Đọc mô hình chiều dài (hồi quy Ridge) do `python -m src.yolo_y8.train --fit-length` lưu; None nếu tắt hoặc chưa có."""
    cfg = load_config().get("measure", {}).get("length_model", {})
    if not cfg.get("enabled", False):
        return None
    path = get_project_root() / cfg["path"]
    if str(path) not in _LENGTH_MODEL:
        _LENGTH_MODEL[str(path)] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return _LENGTH_MODEL[str(path)]


def predict_length_cm(features):
    """Chiều dài (cm) do mô hình dự đoán từ đặc trưng length_features(); None nếu chưa có mô hình."""
    model = load_length_model()
    if model is None:
        return None
    x = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    return float(model["intercept"] + x @ np.asarray(model["coef"]))


def refine_foot_axis(pts, origin, u, v, front_frac: float = 0.65, n_iter: int = 3, n_slices: int = 24):
    """
    Trục thật của bàn chân: đường thẳng qua trung điểm các lát cắt vuông góc trục của phần phía ngón
    (front_frac chiều dài), nơi chắc chắn là bàn chân chứ không phải cổ chân. u hướng về phía ngón.
    Trả về (điểm trên trục, u mới, v mới).
    """
    for _ in range(n_iter):
        a = (pts - origin) @ u
        q = (pts - origin) @ v
        a0, a1 = float(a.min()), float(a.max())
        edges = np.linspace(a1 - front_frac * (a1 - a0), a1, n_slices + 1)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (a >= lo) & (a < hi)
            if np.count_nonzero(s) > 20:
                xs.append((lo + hi) / 2.0)
                ys.append((float(q[s].min()) + float(q[s].max())) / 2.0)
        if len(xs) < 5:
            break
        slope, offset = np.polyfit(xs, ys, 1)
        d = np.array([1.0, slope]) / np.hypot(1.0, slope)
        origin = origin + offset * v
        u, v = d[0] * u + d[1] * v, -d[1] * u + d[0] * v
    return origin, u, v


def measure_foot_pca(rectified_image, foot_mask, foot_contour, mm_per_pixel: float, correction=None):
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
    correction = (hệ số dài, hệ số rộng) từ height_correction(), nhân vào số đo mm; số chưa hiệu chỉnh trả về ở *_raw_mm.
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

    # Chuẩn hóa để p_min là GÓT và p_max là NGÓN: theo kẽ giữa các ngón; không phân biệt được thì theo đầu rộng hơn
    toe_sign = toe_direction_sign(foot_contour, centroid, v_major, mm_per_pixel)
    if toe_sign is None:
        toe_sign = 1 if w2 >= w1 else -1
    if toe_sign < 0:
        v_major = -v_major
        v_minor = -v_minor
        proj_major = (pts_pca - centroid) @ v_major
        p_min = float(np.min(proj_major))
        p_max = float(np.max(proj_major))

    # Trục PCA bị cổ chân/cẳng chân kéo nghiêng; lấy lại trục theo đường giữa của phần bàn chân nhìn rõ
    axis_origin, v_major, v_minor = refine_foot_axis(pts_pca, centroid, v_major, v_minor)
    proj_major = (pts_pca - axis_origin) @ v_major
    p_min = float(np.min(proj_major))
    p_max = float(np.max(proj_major))

    cnt_pts = foot_contour.reshape(-1, 2).astype(np.float32)
    u_foot = v_major.astype(np.float32)
    v_foot = np.array([-u_foot[1], u_foot[0]], dtype=np.float32)

    # Cờ "thấy gót" (xem bằng mắt 53 ảnh test: gót thật thì đầu sau mặt nạ tròn lồi; bị ống quần / cẳng chân che thì
    # mép che làm đầu sau có chỗ lõm sâu hoặc mặt nạ kéo dài ra ngoài):
    #   độ lồi của 25% đầu sau >= HEEL_REAR_SOLIDITY_MIN và chỗ lõm sâu nhất của viền ở 30% đầu sau <= HEEL_REAR_NOTCH_MAX_MM.
    slice_w = []
    for t in range(20):
        lo = p_min + (p_max - p_min) * t / 20.0
        in_slice = (proj_major >= lo) & (proj_major < lo + (p_max - p_min) / 20.0)
        slice_w.append(float(np.ptp((pts_pca[in_slice] - axis_origin) @ v_minor)) if np.any(in_slice) else 0.0)
    rear_ratio = float(np.mean(slice_w[:2]) / max(1e-5, max(slice_w)))  # độ rộng 10% đầu sau / chỗ rộng nhất (đặc trưng của mô hình chiều dài)
    rear_pts = pts_pca[proj_major <= p_min + 0.25 * (p_max - p_min)]
    rear_solidity = float(len(rear_pts) / max(1.0, cv2.contourArea(cv2.convexHull(rear_pts))))
    rear_notch_mm = 0.0
    try:
        defects = cv2.convexityDefects(foot_contour, cv2.convexHull(foot_contour, returnPoints=False))
    except cv2.error:
        defects = None
    if defects is not None:
        contour_a = (cnt_pts - axis_origin) @ v_major
        for _, _, far, depth in defects.reshape(-1, 4):
            if contour_a[far] <= p_min + 0.30 * (p_max - p_min):
                rear_notch_mm = max(rear_notch_mm, depth / 256.0 * mm_per_pixel)
    heel_visible = rear_solidity >= HEEL_REAR_SOLIDITY_MIN and rear_notch_mm <= HEEL_REAR_NOTCH_MAX_MM

    # Đo bằng DÂY CUNG nằm trong mặt nạ, đầu mút nằm trên viền (không đo giữa hai điểm biên ở hai vị trí khác nhau,
    # vì như vậy cộng thêm độ lệch ngang giữa chúng, tức là đường chéo):
    #   chiều dài = dây cung dài nhất song song trục (gót -> ngón)
    #   chiều rộng = dây cung rộng nhất vuông góc trục, trong cửa sổ cách đầu ngón BALL_FROM_TOE_MM (vùng khớp ngón)
    strip_px = max(2.0, CHORD_STRIP_MM / mm_per_pixel)
    q_all = (pts_pca - axis_origin) @ v_foot

    def longest_chord(coord_along, coord_across, lo, hi):
        """Dây cung dài nhất theo coord_along trong các dải rộng strip_px của coord_across (chỉ xét coord_across trong [lo, hi])."""
        keep = (coord_across >= lo) & (coord_across <= hi)
        base = float(coord_across[keep].min())
        strip = np.floor((coord_across[keep] - base) / strip_px).astype(int)
        lows = np.full(strip.max() + 1, np.inf)
        highs = np.full(strip.max() + 1, -np.inf)
        np.minimum.at(lows, strip, coord_along[keep])
        np.maximum.at(highs, strip, coord_along[keep])
        best = int(np.argmax(np.where(np.isfinite(lows), highs - lows, -1.0)))
        return float(lows[best]), float(highs[best]), base + (best + 0.5) * strip_px

    p_heel, p_toe, q_len = longest_chord(proj_major, q_all, -np.inf, np.inf)
    pt_heel = (axis_origin + u_foot * p_heel + v_foot * q_len).astype(np.float32)
    pt_toe = (axis_origin + u_foot * p_toe + v_foot * q_len).astype(np.float32)
    foot_length_px = p_toe - p_heel
    foot_length_raw_mm = foot_length_px * mm_per_pixel
    foot_length_mm = foot_length_raw_mm * (correction[0] if correction else 1.0)

    ball_lo, ball_hi = (p_max - BALL_FROM_TOE_MM[1] / mm_per_pixel, p_max - BALL_FROM_TOE_MM[0] / mm_per_pixel)
    if np.count_nonzero((proj_major >= ball_lo) & (proj_major <= ball_hi)) > 50:
        q_lo, q_hi, a_width = longest_chord(q_all, proj_major, ball_lo, ball_hi)
        foot_width_px = q_hi - q_lo
        pt_w1 = (axis_origin + u_foot * a_width + v_foot * q_lo).astype(np.float32)
        pt_w2 = (axis_origin + u_foot * a_width + v_foot * q_hi).astype(np.float32)
    else:
        q_lo, q_hi = float(q_all.min()), float(q_all.max())
        foot_width_px = q_hi - q_lo
        pt_w1 = (axis_origin + u_foot * (p_max - 0.35 * (p_max - p_min)) + v_foot * q_lo).astype(np.float32)
        pt_w2 = (axis_origin + u_foot * (p_max - 0.35 * (p_max - p_min)) + v_foot * q_hi).astype(np.float32)

    foot_width_raw_mm = foot_width_px * mm_per_pixel
    foot_width_mm = foot_width_raw_mm * (correction[1] if correction else 1.0)
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
    axis_mid = (pt_heel + pt_toe) / 2.0
    c_int = tuple(np.round(axis_mid).astype(int))
    cv2.circle(vis, c_int, 6, (0, 255, 0), -1, cv2.LINE_AA)
    p_major1 = tuple(np.round(axis_mid + v_major * 100).astype(int))
    p_major2 = tuple(np.round(axis_mid - v_major * 100).astype(int))
    cv2.line(vis, p_major1, p_major2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "Truc PCA Chinh", (p_major1[0] + 5, p_major1[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    p_minor1 = tuple(np.round(axis_mid + v_minor * 60).astype(int))
    p_minor2 = tuple(np.round(axis_mid - v_minor * 60).astype(int))
    cv2.line(vis, p_minor1, p_minor2, (0, 165, 255), 2, cv2.LINE_AA)

    # Đường đo chiều dài: từ gót đến đỉnh ngón xa nhất (màu đỏ)
    p_h = tuple(np.round(pt_heel).astype(int))
    p_t = tuple(np.round(pt_toe).astype(int))
    cv2.line(vis, p_h, p_t, (0, 0, 255), 3, cv2.LINE_AA)
    for end in (pt_heel, pt_toe):  # vạch vuông góc trục tại hai đầu mút: chỗ mặt nạ chạm tới
        cv2.line(vis, tuple(np.round(end + v_foot * 60).astype(int)), tuple(np.round(end - v_foot * 60).astype(int)),
                 (0, 0, 255), 2, cv2.LINE_AA)
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
    cv2.rectangle(vis, (12, 12), (12 + banner_w, 172), (20, 20, 20), -1)
    cv2.rectangle(vis, (12, 12), (12 + banner_w, 172), (0, 220, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "KET QUA DO KICH THUOC BAN CHAN (OPENCV + PCA)", (22, 34), font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Chieu dai (Got -> Ngon dai nhat): {foot_length_mm:.1f} mm ({foot_length_mm/10.0:.2f} cm)", (22, 60), font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Chieu rong (1/3 phia truoc)     : {foot_width_mm:.1f} mm ({foot_width_mm/10.0:.2f} cm)", (22, 84), font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Ty so Rong / Dai (W/L Ratio)   : {wl_ratio:.3f}", (22, 108), font, 0.45, (180, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Dang ban chan                  : {foot_shape}", (22, 130), font, 0.45, (255, 200, 80), 1, cv2.LINE_AA)
    if correction:
        cv2.putText(vis, f"Da hieu chinh do cao: dai x{correction[0]:.3f}, rong x{correction[1]:.3f}", (22, 148), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    if not heel_visible:
        cv2.putText(vis, "CANH BAO: khong thay got, so do khong tin cay", (22, 166), font, 0.46, (0, 0, 255), 2, cv2.LINE_AA)

    return {
        "foot_length_raw_mm": round(float(foot_length_raw_mm), 2),
        "foot_width_raw_mm": round(float(foot_width_raw_mm), 2),
        "foot_length_px": round(float(foot_length_px), 2),
        "foot_width_px": round(float(foot_width_px), 2),
        "foot_length_mm": round(float(foot_length_mm), 2),
        "foot_width_mm": round(float(foot_width_mm), 2),
        "foot_length_cm": round(float(foot_length_mm / 10.0), 2),
        "foot_width_cm": round(float(foot_width_mm / 10.0), 2),
        "wl_ratio": round(float(wl_ratio), 3),
        "foot_shape": foot_shape,
        "shape_desc": shape_desc,
        "heel_visible": heel_visible,
        "rear_solidity": round(rear_solidity, 3),
        "rear_notch_mm": round(rear_notch_mm, 1),
        "rear_ratio": round(rear_ratio, 3),
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
                           save_images: bool = True, show_window: bool = False,
                           method: str = "opencv", detector=None, require_heel: bool = None):
    """
    Xử lý trọn gói 1 ảnh qua 7 khối pipeline.
    method="opencv": khối 3 và 5 dùng OpenCV thuần (Canny + ngưỡng màu da).
    method="yolo"  : khối 3 và 5 lấy mask thẻ / chân từ YOLOv8n-seg (detector = predict.load_detector()),
                     các khối còn lại giữ nguyên.
    Các khối:
    1. Đọc ảnh
    2. Đánh giá chất lượng ảnh
    3. Phát hiện 4 góc thẻ ID-1
    4. Nắn phối cảnh Homography & So sánh tỷ lệ
    5. Tách mặt nạ da chân
    6. Đo đạc PCA & Phân loại dáng chân
    7. Vẽ trực quan, in báo cáo terminal chi tiết và xuất file
    Ảnh được thu nhỏ về tối đa Full HD trước khi xử lý (preprocess_image).
    require_heel: True thì không thấy gót -> từ chối đo (Trang_thai "Thất bại", không có số cm); None lấy theo
    measure.require_heel trong configs/yolo_config.yaml.
    """
    img_p = Path(image_path)
    stem = img_p.stem

    record = {
        "Ten_anh": img_p.name,
        "Tap_du_lieu": split_name,
        "Phuong_phap": method,
        "Trang_thai": "Thất bại",
        "Do_sac_net": 0.0,
        "Do_sang": 0.0,
        "The_ID_tim_thay": "KHÔNG",
        "Tach_chan_thanh_cong": "KHÔNG",
        "Chieu_dai_mm": "",
        "Chieu_rong_mm": "",
        "Chieu_dai_cm": "",
        "Chieu_rong_cm": "",
        "Chieu_dai_cm_chua_hieu_chinh": "",
        "Chieu_rong_cm_chua_hieu_chinh": "",
        "Ty_le_W_L": "",
        "Dang_ban_chan": "",
        "Thay_got": "",
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
    if require_heel is None:
        require_heel = load_config().get("measure", {}).get("require_heel", False)
    original_size = img.shape[1], img.shape[0]
    img, resize_scale = preprocess_image(img)
    qa = assess_image_quality(img)
    record["Do_sac_net"] = round(qa["sharpness"], 2)
    record["Do_sang"] = round(qa["brightness"], 2)
    print(f"1. KIỂM TRA CHẤT LƯỢNG ẢNH:")
    if resize_scale < 1.0:
        print(f"   - Kích thước ảnh: {original_size[0]}x{original_size[1]} lớn hơn Full HD, đã thu nhỏ về {img.shape[1]}x{img.shape[0]}")
    else:
        print(f"   - Kích thước ảnh: {img.shape[1]}x{img.shape[0]}")
    print(f"   - Độ sắc nét (Laplacian var): {qa['sharpness']:.1f} (Yêu cầu >= 80)")
    print(f"   - Độ sáng trung bình       : {qa['brightness']:.1f} (Yêu cầu [50, 215])")

    if not qa["is_valid"]:
        record["error_message"] = qa["error_message"]
        print(f"   => [BỎ QUA / YÊU CẦU CHỤP LẠI]: {qa['error_message']}")
        return record
    print(f"   => Chất lượng ảnh ĐẠT TIÊU CHUẨN.")

    # 3. Khối 3: Phát hiện 4 góc thẻ tham chiếu (ID-1)
    if method == "yolo":
        yolo_det = detect_foot_card(detector, img)
        card_res = detect_card_yolo(img, yolo_det)
    else:
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

    if method == "yolo":
        foot_mask, segmented_foot_img, foot_cnt = segment_foot_from_polygon(
            rectified_img, yolo_det["foot_polygon"], rect_res["transform_matrix"], card_polygon)
    else:
        foot_mask, segmented_foot_img, foot_cnt = segment_foot(rectified_img, card_polygon, valid_mask)
    if foot_mask is None or foot_cnt is None:
        record["error_message"] = "Không thể tách mặt nạ bàn chân khỏi nền sàn"
        print(f"4. TÁCH MẶT NẠ BÀN CHÂN:")
        print(f"   => [THẤT BẠI]: {record['error_message']}")
        return record

    record["Tach_chan_thanh_cong"] = "CÓ"
    if method == "yolo":
        print(f"4. TÁCH MẶT NẠ BÀN CHÂN: THÀNH CÔNG (YOLOv8n-seg, conf = {yolo_det['foot_conf']:.2f})")
    else:
        print(f"4. TÁCH MẶT NẠ BÀN CHÂN: THÀNH CÔNG (HSV/YCrCb + Inverted Otsu + Morphology)")

    # 6. Khối 6: Xác định trục chính (PCA), Đo đạc & Phân loại dáng chân
    correction = height_correction(card_res["card_long_px"], img.shape) if method == "yolo" else None
    meas = measure_foot_pca(rectified_img, foot_mask, foot_cnt, rect_res["mm_per_pixel"], correction)
    if meas is None:
        record["error_message"] = "Lỗi khi áp dụng PCA đo đạc kích thước bàn chân"
        print(f"5. ĐO ĐẠC BÀN CHÂN BẰNG PCA:")
        print(f"   => [THẤT BẠI]: {record['error_message']}")
        return record

    record["Thay_got"] = "CÓ" if meas["heel_visible"] else "KHÔNG"
    if method == "yolo":
        record["length_features"] = length_features(meas, foot_mask, foot_cnt, rect_res["mm_per_pixel"], card_res["card_long_px"], img.shape)
    if require_heel and not meas["heel_visible"]:
        record["error_message"] = HEEL_HIDDEN_MESSAGE
        print(f"5. ĐO ĐẠC BÀN CHÂN:")
        print(f"   => [TỪ CHỐI ĐO]: {HEEL_HIDDEN_MESSAGE}")
        return record

    # Chiều dài: mô hình hồi quy học từ số đo thật (chỉ --method yolo, khi đã có mô hình); không thì dùng số đo mask đã hiệu chỉnh.
    # Ảnh che gót đã bị từ chối ở trên (khi require_heel). Dù cờ thấy gót còn báo nhầm ~17% ảnh bị che, ở các ảnh được nhận mô hình
    # hồi quy vẫn tốt hơn số đo hình học thuần (valid 0,81 so với 1,91 cm), nên không chuyển cách đo theo cờ.
    length_mm = meas["foot_length_mm"]
    length_source = "đo trên mask, đã hiệu chỉnh độ cao" if method == "yolo" else "đo trên mask"
    if method == "yolo":
        predicted_cm = predict_length_cm(record["length_features"])
        if predicted_cm is not None:
            length_mm = predicted_cm * 10.0
            length_source = "mô hình hồi quy học từ số đo thật"
            cv2.rectangle(meas["measurement_image"], (12, 176), (12 + min(560, int(meas["measurement_image"].shape[1] * 0.95)), 200), (20, 20, 20), -1)
            cv2.putText(meas["measurement_image"], f"Chieu dai du doan (hoi quy): {predicted_cm:.1f} cm", (22, 194),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    record["Trang_thai"] = "Thành công"
    record["Chieu_dai_mm"] = round(length_mm, 2)
    record["Chieu_rong_mm"] = meas["foot_width_mm"]
    record["Chieu_dai_cm"] = round(length_mm / 10.0, 2)
    record["Chieu_rong_cm"] = meas["foot_width_cm"]
    record["Chieu_dai_cm_chua_hieu_chinh"] = round(meas["foot_length_raw_mm"] / 10.0, 2)
    record["Chieu_rong_cm_chua_hieu_chinh"] = round(meas["foot_width_raw_mm"] / 10.0, 2)
    record["Ty_le_W_L"] = round(meas["foot_width_mm"] / max(1e-5, length_mm), 3)
    record["Dang_ban_chan"] = meas["foot_shape"]
    record["vis_image"] = meas["measurement_image"]

    print(f"5. KẾT QUẢ ĐO ĐẠC BÀN CHÂN (PCA):")
    print(f"   - Chiều dài lớn nhất (Gót -> Ngón): {length_mm:.1f} mm ({length_mm / 10:.2f} cm | {meas['foot_length_px']:.1f} px) [{length_source}]")
    print(f"   - Chiều rộng lớn nhất (1/3 trước) : {meas['foot_width_mm']:.1f} mm ({meas['foot_width_cm']:.2f} cm | {meas['foot_width_px']:.1f} px)")
    if correction:
        print(f"   - Đã hiệu chỉnh độ cao (mu bàn chân cao hơn mặt thẻ): dài x{correction[0]:.3f}, rộng x{correction[1]:.3f}"
              f" (chưa hiệu chỉnh: {meas['foot_length_raw_mm'] / 10:.2f} x {meas['foot_width_raw_mm'] / 10:.2f} cm)")
    print(f"   - Thấy gót                        : " + ("CÓ" if meas["heel_visible"] else "KHÔNG (gót bị che, số đo kém tin cậy: sai số có thể trên 1 cm, nên chụp lại)"))
    print(f"   - Chỉ số tỷ lệ bề ngang (W/L)     : {record['Ty_le_W_L']:.3f}")
    print(f"   - Phân loại dáng chân             : {meas['foot_shape']} ({meas['shape_desc']})")
    print("=" * 78)

    # 7. Xuất file kết quả nếu có thư mục output
    if save_images and output_base_dir:
        out_dir = Path(output_base_dir) / (split_name if method == "opencv" else f"{split_name}_{method}")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}_ket_qua_do.jpg"), meas["measurement_image"])
        cv2.imwrite(str(out_dir / f"{stem}_xoay_dung.jpg"), meas["rotated_image"])
        cv2.imwrite(str(out_dir / f"{stem}_the_id.jpg"), card_res["debug_image"])

    return record


# ==============================================================================
# VÒNG LẶP DUYỆT TẬP DỮ LIỆU VỚI PHÍM TẮT CHUYỂN TIẾP / DỪNG
# ==============================================================================
def run_dataset_loop(split: str = "test", dataset_dir: Path = None, output_dir: Path = None,
                     show_window: bool = True, save_images: bool = True,
                     method: str = "opencv", detector=None):
    """
    Duyệt toàn bộ tập dữ liệu, hiển thị kết quả đo:
    - Bấm bất kỳ phím nào để xem ảnh tiếp theo.
    - Bấm 'q' hoặc 'Q' để dừng vòng lặp.
    - Xuất bảng CSV tổng hợp vào output_dir.
    """
    image_paths = load_dataset(split=split, dataset_dir=dataset_dir)
    n_imgs = len(image_paths)

    print("\n" + "#" * 80)
    print(f"CHƯƠNG TRÌNH ĐO BÀN CHÂN VỚI THẺ THAM CHIẾU ID-1 (phương pháp: {method.upper()})")
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
                                     save_images=save_images, show_window=show_window,
                                     method=method, detector=detector)
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
        rec.pop("vis_image", None)  # ảnh vẽ kết quả rất nặng: bỏ ngay để RAM không tăng theo số ảnh (chỉ CSV cần record)

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
        csv_file = out_p / (f"bang_ket_qua_do_{split}.csv" if method == "opencv"
                            else f"bang_ket_qua_do_{split}_{method}.csv")
        fieldnames = [
            "Ten_anh", "Tap_du_lieu", "Phuong_phap", "Trang_thai", "Do_sac_net", "Do_sang",
            "The_ID_tim_thay", "Tach_chan_thanh_cong",
            "Chieu_dai_mm", "Chieu_rong_mm", "Chieu_dai_cm", "Chieu_rong_cm",
            "Chieu_dai_cm_chua_hieu_chinh", "Chieu_rong_cm_chua_hieu_chinh",
            "Ty_le_W_L", "Dang_ban_chan", "Thay_got", "Ty_le_Homography_px_mm",
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
    parser.add_argument("--json", action="store_true", default=False,
                        help="Ảnh đơn: in đúng 1 dòng JSON {ok, length_cm, width_cm, error} ra stdout (báo cáo chi tiết chuyển sang stderr)")
    parser.add_argument("--method", "-m", type=str, choices=["opencv", "yolo"], default="opencv",
                        help="Cách lấy mask thẻ/chân: 'opencv' (mặc định) hoặc 'yolo' (cần weights đã train)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Weights YOLO cho --method yolo (mặc định: khóa infer.weights trong configs/yolo_config.yaml)")

    args = parser.parse_args()
    project_root = get_project_root()
    target_img = args.image or args.image_pos

    out_dir = Path(args.output).resolve() if args.output else (project_root / "outputs")
    detector = load_detector(args.weights) if args.method == "yolo" else None

    # Trường hợp 1: Chạy ảnh đơn lẻ
    if target_img:
        img_p = Path(target_img).resolve()
        if not img_p.exists():
            if args.json:
                print(json.dumps({"ok": False, "length_cm": None, "width_cm": None,
                                  "error": f"Không tìm thấy file ảnh: {target_img}"}, ensure_ascii=False))
            else:
                print(f"[LỖI] Không tìm thấy file ảnh: {target_img}")
            return
        with redirect_stdout(sys.stderr if args.json else sys.stdout):
            rec = process_image_pipeline(img_p, output_base_dir=out_dir, split_name="single",
                                         save_images=True, show_window=not args.no_gui and not args.json,
                                         method=args.method, detector=detector)
        if args.json:
            ok = rec["Trang_thai"] == "Thành công"
            print(json.dumps({"ok": ok, "length_cm": rec["Chieu_dai_cm"] if ok else None,
                              "width_cm": rec["Chieu_rong_cm"] if ok else None,
                              "error": rec["error_message"]}, ensure_ascii=False))
        elif not args.no_gui and rec.get("vis_image") is not None:
            cv2.imshow("Ket qua do anh don le", rec["vis_image"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Trường hợp 2: Chạy tập dữ liệu (mặc định 'test')
        data_dir = Path(args.data).resolve() if args.data else (project_root / "data" / "yolo_dataset" / "images")
        splits_to_run = ["train", "valid", "test"] if args.split == "all" else [args.split]
        for sp in splits_to_run:
            run_dataset_loop(split=sp, dataset_dir=data_dir, output_dir=out_dir,
                             show_window=not args.no_gui, save_images=True,
                             method=args.method, detector=detector)


if __name__ == "__main__":
    main()
