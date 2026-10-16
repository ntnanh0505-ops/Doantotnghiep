"""
card_detector.py — Khối 1: Phát hiện thẻ ID-1 trong ảnh (detect_card).

Chức năng:
    - refine_card_corners(cnt, approx_corners) : Tinh chỉnh góc thẻ bằng Line Fitting.
    - detect_card_corners(image, ...)          : Phát hiện 4 góc thẻ ID-1 theo chuẩn ISO 7810.
    - detect_card                              : Alias của detect_card_corners.

Chiến lược nhận diện:
    1. Khóa vị trí thẻ bằng dải màu xanh sinh viên đặc thù (HSV + B-R channel diff).
    2. Mở rộng ROI 2.5× và áp dụng Multi-scale Canny + BilateralFilter + AdaptiveThreshold.
    3. Lọc ứng viên: tỷ lệ cạnh [1.35, 1.80], độ lồi ≥ 0.60.
    4. Tính điểm tổng hợp (solidity + aspect_diff + area) và chọn ứng viên tốt nhất.
    5. Fallback: dùng minAreaRect trên vỏ lồi khi viền thẻ hòa lẫn sàn.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

from .constants import CARD_LONG_MM, CARD_SHORT_MM, CARD_ASPECT_RATIO
from .perspective import order_points


def refine_card_corners(cnt, approx_corners):
    """
    Làm mịn và xác định 4 góc thẻ ID-1 với độ chính xác milimet:
    - Thẻ ID-1 có 4 góc bo tròn (bán kính R = 3.18mm theo chuẩn ISO 7810).
    - cv2.approxPolyDP thông thường dễ cắt góc hoặc lệch biên.
    - Phương pháp: Khớp 4 đường thẳng (cv2.fitLine) vào 4 cạnh thẳng của contour
      (bỏ qua 15% phần bo tròn ở 2 đầu mỗi cạnh), sau đó tìm giao điểm toán học
      của 4 đường thẳng đó để tái tạo 4 đỉnh vuông góc chuẩn xác tuyệt đối.
    """
    try:
        pts = approx_corners.astype(np.float32)
        lines = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-3:
                return approx_corners
            u_edge = edge_vec / edge_len
            n_edge = np.array([-u_edge[1], u_edge[0]])

            c_pts = cnt.reshape(-1, 2).astype(np.float32)
            d_along = (c_pts - p1) @ u_edge
            d_perp = np.abs((c_pts - p1) @ n_edge)

            mask = (d_along >= 0.15 * edge_len) & (d_along <= 0.85 * edge_len) & (d_perp <= max(3.0, 0.08 * edge_len))
            if np.sum(mask) >= 5:
                edge_pts = c_pts[mask]
                [vx, vy, x0, y0] = cv2.fitLine(edge_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                lines.append((float(vx), float(vy), float(x0), float(y0)))
            else:
                lines.append((float(u_edge[0]), float(u_edge[1]), float(p1[0]), float(p1[1])))

        intersections = []
        for i in range(4):
            l1 = lines[(i - 1) % 4]
            l2 = lines[i]
            A1, B1 = l1[1], -l1[0]
            C1 = l1[1] * l1[2] - l1[0] * l1[3]
            A2, B2 = l2[1], -l2[0]
            C2 = l2[1] * l2[2] - l2[0] * l2[3]
            det = A1 * B2 - A2 * B1
            if abs(det) > 1e-4:
                ix = (C1 * B2 - C2 * B1) / det
                iy = (A1 * C2 - A2 * C1) / det
                intersections.append([ix, iy])
            else:
                intersections.append(pts[i].tolist())
        return np.array(intersections, dtype=np.float32)
    except Exception:
        return approx_corners


def detect_card_corners(image, debug=False, verbose=True):
    """
    Khối 1 — Phát hiện 4 góc thẻ ID-1 theo chuẩn Đề cương Mốc 1:
    - Tiền xử lý đa ngưỡng (Multi-scale Canny / BilateralFilter / Morphological Closing)
      để hàn kín viền trắng ngoài cùng của thẻ với mặt sàn, không bị đứt quãng.
    - Tìm contours ngoài cùng (cv2.RETR_EXTERNAL), loại bỏ biên dính mép ảnh (quần, ron gạch).
    - Dùng cv2.approxPolyDP với epsilon linh hoạt (0.02 - 0.04 * chu vi) và cv2.minAreaRect.
    - Kiểm tra tỷ lệ cạnh: chấp nhận dải thực tế do góc chụp nghiêng là 1.35 <= L/W <= 1.80
      (ưu tiên ứng viên có tỷ lệ gần nhất với chuẩn ISO ID-1 ~ 1.586).
    - Loại bỏ triệt để việc bắt nhầm vào khối hoa văn/chân dung/logo bên trong thẻ làm tỷ lệ tụt xuống 1.20.
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        return {
            "success": False,
            "corners": None,
            "debug_image": None,
            "aspect_ratio": None,
            "area": 0.0,
            "card_long_px": 0.0,
            "card_short_px": 0.0,
            "error_message": "Định dạng ảnh đầu vào không hợp lệ",
            "debug_maps": {}
        }

    if img is None:
        return {
            "success": False,
            "corners": None,
            "debug_image": None,
            "aspect_ratio": None,
            "area": 0.0,
            "card_long_px": 0.0,
            "card_short_px": 0.0,
            "error_message": "Không thể giải mã hoặc đọc file ảnh",
            "debug_maps": {}
        }

    h_img, w_img = img.shape[:2]
    total_area = float(w_img * h_img)

    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Giai đoạn 1: Khóa chặt vị trí thẻ bằng dải màu xanh sinh viên đặc thù
    cond_blue = (
        (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 135) &
        (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 30) &
        (b.astype(int) - r.astype(int) > 8)
    )
    mask_blue = (cond_blue * 255).astype(np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, k5)
    cnts_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_candidates = []
    for c in cnts_blue:
        area = cv2.contourArea(c)
        if 350 <= area <= 0.15 * total_area:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Thẻ nằm hoàn toàn trong mặt sàn, tuyệt đối không dính 4 mép viền ảnh (loại trừ ống quần)
            if bx > 8 and by > 8 and (bx + bw) < w_img - 8 and (by + bh) < h_img - 8:
                rect = cv2.minAreaRect(c)
                (cx, cy), (bw_m, bh_m), angle = rect
                if min(bw_m, bh_m) >= 12:
                    blue_candidates.append({
                        "contour": c,
                        "area": area,
                        "center": (cx, cy),
                        "rect": rect,
                        "size": (bw_m, bh_m),
                        "bbox": (bx, by, bw, bh)
                    })

    # Fallback dải màu xanh rộng nếu ảnh thiếu sáng hoặc lệch sáng
    if not blue_candidates:
        cond_blue_w = ((hsv[:, :, 0] >= 80) & (hsv[:, :, 0] <= 140) & (b.astype(int) - r.astype(int) > 4))
        mask_w = cv2.morphologyEx((cond_blue_w * 255).astype(np.uint8), cv2.MORPH_CLOSE, k5)
        cnts_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_w:
            area = cv2.contourArea(c)
            if 350 <= area <= 0.15 * total_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bx > 8 and by > 8 and (bx + bw) < w_img - 8 and (by + bh) < h_img - 8:
                    rect = cv2.minAreaRect(c)
                    (cx, cy), (bw_m, bh_m), angle = rect
                    if min(bw_m, bh_m) >= 12:
                        blue_candidates.append({
                            "contour": c,
                            "area": area,
                            "center": (cx, cy),
                            "rect": rect,
                            "size": (bw_m, bh_m),
                            "bbox": (bx, by, bw, bh)
                        })

    if not blue_candidates:
        if verbose:
            print("[CẢNH BÁO] Không tìm thấy dải màu xanh đặc trưng của thẻ ID!")
        debug_img = img.copy()
        cv2.putText(debug_img, "KHONG TIM THAY THE ID-1", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return {
            "success": False,
            "corners": None,
            "debug_image": debug_img,
            "aspect_ratio": None,
            "area": 0.0,
            "card_long_px": 0.0,
            "card_short_px": 0.0,
            "error_message": "card_not_found: Không tìm thấy vùng thẻ ID trên mặt sàn",
            "debug_maps": {}
        }

    # Sắp xếp theo diện tích hợp lý
    blue_candidates.sort(key=lambda item: item["area"], reverse=True)
    best_candidate = None
    best_score = -1.0

    for cand in blue_candidates[:3]:
        cx, cy = cand["center"]
        bw, bh = cand["size"]
        blue_area = cand["area"]

        # Mở rộng ROI ra 2.4 lần để bao trọn mép ngoài viền trắng
        roi_dim = int(max(bw, bh) * 2.5)
        roi_dim = max(180, min(roi_dim, 340))
        rx1, ry1 = max(0, int(cx - roi_dim // 2)), max(0, int(cy - roi_dim // 2))
        rx2, ry2 = min(w_img, int(cx + roi_dim // 2)), min(h_img, int(cy + roi_dim // 2))

        roi = img[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue

        roi_cx = cx - rx1
        roi_cy = cy - ry1
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Tiền xử lý BilateralFilter làm mượt hoa văn nội bộ nhưng giữ sắc nét biên thẻ với mặt sàn
        roi_bilateral = cv2.bilateralFilter(roi_gray, 9, 75, 75)
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        roi_maps = []
        # Multi-scale Canny trên BilateralFilter + Morphological Closing 7x7
        for low, high in [(15, 50), (25, 75), (35, 105), (45, 135)]:
            edges = cv2.Canny(roi_bilateral, low, high)
            roi_maps.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k7))

        # Phân ngưỡng thích nghi Adaptive Threshold
        for bs in [21, 31, 41]:
            ad = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 4)
            roi_maps.append(cv2.morphologyEx(ad, cv2.MORPH_CLOSE, k7))

        # Morphological Gradient
        grad = cv2.morphologyEx(roi_bilateral, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, grad_th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        roi_maps.append(cv2.morphologyEx(grad_th, cv2.MORPH_CLOSE, k7))

        found_in_roi = False
        for b_map in roi_maps:
            cnts_roi, _ = cv2.findContours(b_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c_roi in cnts_roi:
                area_c = cv2.contourArea(c_roi)
                if area_c < blue_area * 0.90 or area_c > 35000:
                    continue
                if cv2.pointPolygonTest(c_roi, (roi_cx, roi_cy), False) < 0:
                    continue

                hull = cv2.convexHull(c_roi)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                solidity = area_c / hull_area
                if solidity < 0.60:
                    continue

                rect_c = cv2.minAreaRect(hull)
                rw, rh = rect_c[1]
                if min(rw, rh) < 25:
                    continue

                aspect = max(rw, rh) / max(1.0, min(rw, rh))
                # KIỂM TRA TỶ LỆ CẠNH: Chấp nhận dải thực tế [1.35, 1.80]
                if not (1.35 <= aspect <= 1.80):
                    continue

                # Dùng cv2.approxPolyDP linh hoạt (0.02 - 0.04 * chu vi)
                peri = cv2.arcLength(hull, True)
                corners_roi = None
                for eps in np.linspace(0.02, 0.04, 7):
                    approx = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        corners_roi = approx.reshape(4, 2).astype(np.float32)
                        break

                if corners_roi is None:
                    corners_roi = cv2.boxPoints(rect_c).astype(np.float32)

                corners_orig = corners_roi + np.array([rx1, ry1], dtype=np.float32)
                aspect_diff = abs(aspect - CARD_ASPECT_RATIO)
                # Chấm điểm: ưu tiên tỷ lệ sát 1.586 và độ lồi cao
                score = (solidity * 3.0) + (1.0 / (1.0 + 5.0 * aspect_diff)) * 5.0 + (area_c / 10000.0)

                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "corners": order_points(corners_orig),
                        "aspect": aspect,
                        "area": area_c,
                        "score": score,
                        "source": "Bilateral + Multi-scale Canny + ConvexHull"
                    }
                    found_in_roi = True

        # Fallback: nếu viền trắng hòa lẫn màu sàn, dùng minAreaRect trên vỏ lồi của vùng thẻ mở rộng
        if not found_in_roi and cand["area"] >= 1500:
            rect_b = cand["rect"]
            (cx_b, cy_b), (bw_b, bh_b), angle_b = rect_b
            long_side = max(bw_b, bh_b)
            w_est = max(long_side * 1.15, 130.0)
            h_est = w_est / CARD_ASPECT_RATIO
            card_box = cv2.boxPoints(((cx_b, cy_b), (w_est, h_est), angle_b)).astype(np.float32)
            score_b = 2.0
            if score_b > best_score:
                best_score = score_b
                best_candidate = {
                    "corners": order_points(card_box),
                    "aspect": CARD_ASPECT_RATIO,
                    "area": w_est * h_est,
                    "score": score_b,
                    "source": "Expanded Card Hull Anchor"
                }

        if best_candidate is not None and best_candidate["score"] > 4.5:
            break

    if best_candidate is None:
        if verbose:
            print("[CẢNH BÁO] Không tìm thấy tứ giác thẻ ID đạt tỷ lệ [1.35 - 1.80]!")
        debug_img = img.copy()
        cv2.putText(debug_img, "KHONG TIM THAY THE ID-1", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return {
            "success": False,
            "corners": None,
            "debug_image": debug_img,
            "aspect_ratio": None,
            "area": 0.0,
            "card_long_px": 0.0,
            "card_short_px": 0.0,
            "error_message": "card_not_found: Không tìm thấy ứng viên thẻ ID đạt tỷ lệ [1.35 - 1.80]",
            "debug_maps": {}
        }

    ordered_final = best_candidate["corners"]
    w_top = np.linalg.norm(ordered_final[1] - ordered_final[0])
    w_bot = np.linalg.norm(ordered_final[2] - ordered_final[3])
    h_left = np.linalg.norm(ordered_final[3] - ordered_final[0])
    h_right = np.linalg.norm(ordered_final[2] - ordered_final[1])
    c_long_px = max((w_top + w_bot) / 2.0, (h_left + h_right) / 2.0)
    c_short_px = min((w_top + w_bot) / 2.0, (h_left + h_right) / 2.0)

    if verbose:
        print(f"[CHỌN ỨNG VIÊN THẺ]:")
        print(f"  - Cơ chế nhận diện: {best_candidate['source']}")
        print(f"  - Tỷ lệ cạnh      : {best_candidate['aspect']:.3f} (Chuẩn ID-1: ~1.586)")
        print(f"  - Kích thước thẻ  : {c_long_px:.1f} x {c_short_px:.1f} px")
        print(f"  - Điểm tổng hợp   : {best_candidate['score']:.2f}")

    debug_img = img.copy()
    pts_int = np.round(ordered_final).astype(np.int32)
    cv2.polylines(debug_img, [pts_int], True, (0, 255, 0), 3, cv2.LINE_AA)
    corner_names = ["TL (1)", "TR (2)", "BR (3)", "BL (4)"]
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
    for i, pt in enumerate(pts_int):
        cv2.circle(debug_img, tuple(pt), 7, colors[i], -1, cv2.LINE_AA)
        cv2.circle(debug_img, tuple(pt), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_img, corner_names[i], (pt[0] + 8, pt[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 2, cv2.LINE_AA)

    label_txt = f"The ID-1: {CARD_LONG_MM:.1f} x {CARD_SHORT_MM:.1f} mm (Aspect: {best_candidate['aspect']:.2f})"
    cv2.putText(debug_img, label_txt, (max(10, pts_int[0][0]), max(25, pts_int[0][1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)

    return {
        "success": True,
        "corners": ordered_final,
        "debug_image": debug_img,
        "aspect_ratio": best_candidate["aspect"],
        "area": best_candidate["area"],
        "card_long_px": round(float(c_long_px), 2),
        "card_short_px": round(float(c_short_px), 2),
        "error_message": "",
        "debug_maps": {}
    }


# Alias tương thích ngược
detect_card = detect_card_corners
