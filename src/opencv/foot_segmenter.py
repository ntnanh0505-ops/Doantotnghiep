"""
foot_segmenter.py — Khối 3: Tách mặt nạ bàn chân và cắt tại gót chân (segment_foot).

Chiến lược phân đoạn:
    1. Loại bỏ vùng thẻ ID và vùng ngoài ảnh hợp lệ.
    2. Nhận diện da chân qua đa không gian màu:
       - YCrCb: Cr [132-178], Cb [80-135]
       - BGR: R - B > 8, R >= G >= B-15, R > 50
       - HSV: H <= 28 hoặc H >= 165, S [20-220]
    3. Hỗ trợ Otsu nghịch đảo khi da không đủ nổi bật.
    4. CẮT TẠI GÓT CHÂN (Heel Truncation):
       Phát hiện eo cổ chân ở khoảng 210-285 mm tính từ ngón chân,
       loại bỏ phần bắp chân kéo dài ra khỏi khung đo.
"""

import cv2
import numpy as np
from .constants import CARD_LONG_MM


def segment_foot(rectified_image, card_polygon, valid_mask=None, debug=False):
    """
    Khối 3 — Tách mặt nạ bàn chân chuẩn xác:
    - Loại bỏ hoàn toàn vùng thẻ và lề đệm bên ngoài ảnh nắn.
    - Nhận diện vùng da chân qua kết hợp đa không gian màu: YCrCb (Cr [132-178], Cb [80-135])
      + HSV + BGR contrast (R - B > 8).
    - Lọc ứng viên contour dựa trên diện tích nhân trắc học milimet [3000 mm² - 55000 mm²].
    - CẮT TẠI GÓT CHÂN (Heel Truncation):
      Nếu ảnh chụp đứng có phần cổ chân/bắp chân kéo dài lên trên, quét tìm vị trí thắt eo cổ chân
      (vùng thắt eo tự nhiên ở khoảng cách nhân trắc học 210 - 285 mm tính từ ngón chân) và cắt bỏ phần bắp chân.
      Đảm bảo mặt nạ chỉ chứa phần bàn chân từ GÓT CHÂN đến ĐẦU NGÓN CHÂN DÀI NHẤT.
    - Trả về: (foot_mask, segmented_foot_img, foot_contour)
    """
    if rectified_image is None or card_polygon is None:
        return None, None, None

    h, w = rectified_image.shape[:2]
    total_area = float(w * h)

    # 1. Tạo mặt nạ loại trừ vùng thẻ ID
    card_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(card_mask, [np.round(card_polygon).astype(np.int32)], 255)
    margin = max(15, int(min(w, h) * 0.03))
    card_mask = cv2.dilate(card_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (margin, margin)))
    card_cx = float(np.mean(card_polygon[:, 0]))
    card_cy = float(np.mean(card_polygon[:, 1]))

    # Mặt nạ vùng ảnh hợp lệ (loại trừ viền đệm biên)
    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=np.uint8) * 255

    # 2. Trích xuất đặc trưng sắc tố da người
    ycrcb = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(rectified_image)

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    cond_ycrcb = (cr >= 132) & (cr <= 178) & (cb >= 80) & (cb <= 135)
    cond_bgr = (r.astype(np.int32) - b.astype(np.int32) > 8) & (r >= g) & (g >= b - 15) & (r > 50)
    cond_hsv = ((hsv[:, :, 0] <= 28) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 1] <= 220)

    skin = (cond_ycrcb & cond_bgr & cond_hsv & (valid_mask > 0) & (card_mask == 0)).astype(np.uint8) * 255

    # Phân ngưỡng Otsu hỗ trợ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    inv_cl = cv2.bitwise_not(cl)
    inv_cl[card_mask > 0] = 0
    inv_cl[valid_mask == 0] = 0
    _, inv_otsu = cv2.threshold(inv_cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_otsu[card_mask > 0] = 0
    inv_otsu[valid_mask == 0] = 0

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    skin_clean = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k_close)
    skin_clean = cv2.morphologyEx(skin_clean, cv2.MORPH_OPEN, k_open)

    cnts, _ = cv2.findContours(skin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Nếu bản đồ da quá nhỏ, thử kết hợp Otsu tại vùng có sắc tố da
    if not cnts or max([cv2.contourArea(c) for c in cnts], default=0) < 0.01 * total_area:
        combined = cv2.bitwise_or(skin_clean, cv2.bitwise_and(inv_otsu, inv_otsu, mask=(r > b).astype(np.uint8) * 255))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open)
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.008 * total_area <= area <= 0.50 * total_area):
            continue

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if min(rw, rh) < 18:
            continue

        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if not (1.10 <= aspect <= 4.5):
            continue

        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        overlap_card = np.logical_and(c_mask > 0, card_mask > 0).sum()
        if overlap_card / max(1.0, area) > 0.05:
            continue

        cx, cy = rect[0]
        dist_to_card = np.linalg.norm(np.array([cx, cy]) - np.array([card_cx, card_cy]))
        skin_ratio = (skin[c_mask > 0] > 0).sum() / max(1.0, area)

        score = (skin_ratio * 4.0) + (area / total_area) * 8.0 + (1.0 / (1.0 + dist_to_card / 300.0))
        valid_cands.append((c, score))

    # Fallback: Cơ chế ROI thích ứng quang học quanh thẻ ID (hỗ trợ nền sàn gỗ hoặc sàn tối màu)
    if not valid_cands:
        try:
            c_edge1 = np.linalg.norm(card_polygon[1] - card_polygon[0])
            c_edge2 = np.linalg.norm(card_polygon[3] - card_polygon[0])
            c_long_px = max(c_edge1, c_edge2)
            # Bán kính tìm kiếm bàn chân quanh thẻ (~3.5 đến 4.0 lần chiều dài thẻ)
            r_px = int(max(c_long_px * 3.8, min(w, h) * 0.40))
            x1 = max(0, int(card_cx - r_px))
            x2 = min(w, int(card_cx + r_px))
            y1 = max(0, int(card_cy - r_px))
            y2 = min(h, int(card_cy + r_px))

            roi_gray = gray[y1:y2, x1:x2]
            roi_valid = (valid_mask[y1:y2, x1:x2] > 0) & (card_mask[y1:y2, x1:x2] == 0)
            if np.sum(roi_valid) > 500:
                valid_px = roi_gray[roi_valid]
                th_val, _ = cv2.threshold(valid_px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                for inv_mode in [True, False]:
                    if inv_mode:
                        _, roi_bin = cv2.threshold(roi_gray, th_val, 255, cv2.THRESH_BINARY_INV)
                    else:
                        _, roi_bin = cv2.threshold(roi_gray, th_val, 255, cv2.THRESH_BINARY)
                    roi_bin[~roi_valid] = 0
                    roi_clean = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, k_close)
                    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, k_open)

                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = roi_clean
                    cnts_roi, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cr in cnts_roi:
                        ar = cv2.contourArea(cr)
                        if 0.008 * total_area <= ar <= 0.45 * total_area:
                            rect_r = cv2.minAreaRect(cr)
                            rw, rh = rect_r[1]
                            asp = max(rw, rh) / max(1.0, min(rw, rh))
                            if 1.15 <= asp <= 4.5:
                                valid_cands.append((cr, ar / total_area * 10.0))
        except Exception:
            pass

    if not valid_cands:
        return None, None, None

    best_cnt = max(valid_cands, key=lambda x: x[1])[0]
    foot_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(foot_mask, [best_cnt], -1, 255, -1)

    # 3. CẮT BỎ CỔ CHÂN / BẮP CHÂN ĐỂ MẶT NẠ DỪNG LẠI CHUẨN XÁC TẠI GÓT CHÂN
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

        # Xác định đầu nào là ngón chân và đầu nào là cổ chân / gót
        pts_min_end = pts_xy[proj_major <= p_min + 0.15 * (p_max - p_min)]
        pts_max_end = pts_xy[proj_major >= p_max - 0.15 * (p_max - p_min)]
        min_touches_top = np.any(pts_min_end[:, 1] <= 15)
        max_touches_top = np.any(pts_max_end[:, 1] <= 15)

        w_min_slice, w_max_slice = [], []
        for t in np.linspace(p_min + 0.15 * (p_max - p_min), p_min + 0.35 * (p_max - p_min), 10):
            m = (proj_major >= t - 3) & (proj_major <= t + 3)
            if np.any(m):
                q = pts_centered[m] @ v_minor
                w_min_slice.append(np.max(q) - np.min(q))
        for t in np.linspace(p_max - 0.35 * (p_max - p_min), p_max - 0.15 * (p_max - p_min), 10):
            m = (proj_major >= t - 3) & (proj_major <= t + 3)
            if np.any(m):
                q = pts_centered[m] @ v_minor
                w_max_slice.append(np.max(q) - np.min(q))

        avg_w_min = np.mean(w_min_slice) if w_min_slice else 0
        avg_w_max = np.mean(w_max_slice) if w_max_slice else 0

        # Chuẩn hóa v_major sao cho p_max là NGÓN CHÂN và p_min là GÓT / CỔ CHÂN
        if not (min_touches_top or (avg_w_max > avg_w_min and not max_touches_top)):
            v_major = -v_major
            v_minor = -v_minor
            proj_major = pts_centered @ v_major
            p_min = float(np.min(proj_major))
            p_max = float(np.max(proj_major))

        # Tính mm_per_pixel nội bộ từ thẻ đã nắn
        c_edge1 = np.linalg.norm(card_polygon[1] - card_polygon[0])
        c_edge2 = np.linalg.norm(card_polygon[3] - card_polygon[0])
        mm_px_local = CARD_LONG_MM / max(1.0, max(c_edge1, c_edge2))

        span_px = p_max - p_min
        span_mm = span_px * mm_px_local

        # Nếu chiều dài vượt quá chiều dài bàn chân người chuẩn (> 275 mm) hoặc chân chạm mép trên ảnh
        if min_touches_top or span_mm > 275.0:
            # Vùng tìm kiếm gót chân theo nhân trắc học: từ 210 mm đến 290 mm tính từ đỉnh ngón chân lùi lại
            dist_heel_max_px = 290.0 / mm_px_local
            dist_heel_min_px = 210.0 / mm_px_local

            t_search_start = max(p_min, p_max - dist_heel_max_px)
            t_search_end = min(p_max - 120.0 / mm_px_local, p_max - dist_heel_min_px)

            best_t_cut = p_min
            min_w_scan = 999999.0
            scan_step = max(2.0, (t_search_end - t_search_start) / 35.0)

            for t_scan in np.arange(t_search_start, t_search_end, scan_step):
                m = (proj_major >= t_scan - 4.0) & (proj_major <= t_scan + 4.0)
                if np.any(m):
                    q = pts_centered[m] @ v_minor
                    w_curr = np.max(q) - np.min(q)
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

    # Tạo ảnh bàn chân giữ nguyên màu gốc trên nền trắng
    segmented_foot_img = np.full_like(rectified_image, 255)
    segmented_foot_img[foot_mask > 0] = rectified_image[foot_mask > 0]

    return foot_mask, segmented_foot_img, best_cnt
