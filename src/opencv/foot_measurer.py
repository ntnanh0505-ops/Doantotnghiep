"""
foot_measurer.py — Khối 4: Đo kích thước bàn chân bằng PCA (measure_foot_pca).

Thuật toán:
    1. cv2.PCACompute xác định trục chính (major axis) của mặt nạ bàn chân.
    2. Phân biệt Heel ↔ Toe bằng độ bè ngang ở 30% hai đầu.
    3. Chiều dài = khoảng cách Euclid từ Heel đến Toe × mm_per_pixel.
    4. Chiều rộng = đoạn ngang lớn nhất ở vùng khớp bàn ngón (58%-85% từ gót).
    5. Vẽ HUD thông số (bảng đen, đường đo màu đỏ/xanh) lên ảnh kết quả.
"""

import cv2
import numpy as np


def measure_foot_pca(rectified_image, foot_mask, foot_contour, mm_per_pixel):
    """
    Khối 4 — Đo kích thước bàn chân bằng PCA theo chuẩn Mốc 1:
    - Tìm trục chính (major axis) của mặt nạ chân bằng cv2.PCACompute.
    - Chiều dài chân (mm) = Chiều dài hình chiếu cực đại dọc theo trục chính * mm_per_pixel.
    - Chiều rộng chân (mm) = Chiều rộng lớn nhất vuông góc trục chính ở khu vực 1/3 phía trước
      bàn chân (khớp bàn ngón chân / metatarsal) * mm_per_pixel.
    - Tính tỷ số W/L.
    - Vẽ trực quan các đường đo và bảng HUD thông số lên ảnh kết quả.
    """
    if foot_mask is None or foot_contour is None or mm_per_pixel <= 0:
        return None

    y_idx, x_idx = np.where(foot_mask > 0)
    if len(x_idx) < 150:
        return None

    # Tọa độ các điểm ảnh bàn chân
    pts_pca = np.column_stack((x_idx, y_idx)).astype(np.float32)

    # 1. Áp dụng cv2.PCACompute tìm trục chính và trục phụ
    mean, eigenvectors = cv2.PCACompute(pts_pca, mean=None, maxComponents=2)
    v_major = eigenvectors[0]  # Vector đơn vị trục chính
    v_minor = eigenvectors[1]  # Vector đơn vị trục phụ vuông góc

    proj_major = (pts_pca - mean) @ v_major
    p_min = float(np.min(proj_major))
    p_max = float(np.max(proj_major))
    length_raw_px = p_max - p_min

    if length_raw_px <= 20.0:
        return None

    # Phân biệt đầu gót (Heel) và đầu ngón (Toe) qua độ bè ngang
    s1 = (proj_major >= p_min) & (proj_major <= p_min + 0.30 * length_raw_px)
    s2 = (proj_major >= p_max - 0.30 * length_raw_px) & (proj_major <= p_max)
    w1 = float(np.ptp((pts_pca[s1] - mean) @ v_minor)) if np.any(s1) else 0.0
    w2 = float(np.ptp((pts_pca[s2] - mean) @ v_minor)) if np.any(s2) else 0.0

    # Chuẩn hóa để p_min là GÓT và p_max là NGÓN
    if w1 > w2:
        v_major = -v_major
        v_minor = -v_minor
        proj_major = (pts_pca - mean) @ v_major
        p_min = float(np.min(proj_major))
        p_max = float(np.max(proj_major))

    cnt_pts = foot_contour.reshape(-1, 2).astype(np.float32)
    cnt_proj = (cnt_pts - mean) @ v_major

    # Điểm ngón chân dài nhất (Acropodion)
    idx_toe = np.argmax(cnt_proj)
    pt_toe = cnt_pts[idx_toe]

    # Điểm gót chân (Pternion)
    idx_heel = np.argmin(cnt_proj)
    pt_heel = cnt_pts[idx_heel]

    # Chiều dài chân (mm) = khoảng cách Euclid dọc trục chính * mm_per_pixel
    foot_length_px = float(np.linalg.norm(pt_toe - pt_heel))
    foot_length_mm = foot_length_px * mm_per_pixel

    # Chiều rộng chân: đo vuông góc trục chính ở khu vực 1/3 phía trước bàn chân
    u_foot = (pt_toe - pt_heel) / max(1e-5, foot_length_px)
    v_foot = np.array([-u_foot[1], u_foot[0]], dtype=np.float32)

    proj_foot = (cnt_pts - pt_heel) @ u_foot
    # Khu vực 1/3 phía trước: từ 0.58 đến 0.85 khoảng cách từ gót
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

    foot_length_cm = round(foot_length_mm / 10.0, 2)
    foot_width_cm = round(foot_width_mm / 10.0, 2)
    wl_ratio = round(foot_width_mm / max(1e-5, foot_length_mm), 3)

    # 4. Vẽ trực quan lên ảnh kết quả
    vis = rectified_image.copy()
    cv2.drawContours(vis, [foot_contour], -1, (0, 220, 255), 2, cv2.LINE_AA)

    # Đường đo chiều dài: từ gót đến ngón dài nhất (màu đỏ)
    p_h = tuple(np.round(pt_heel).astype(int))
    p_t = tuple(np.round(pt_toe).astype(int))
    cv2.line(vis, p_h, p_t, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(vis, p_h, 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, p_t, 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(vis, "Got chan (Heel)", (p_h[0] + 10, p_h[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Ngon dai nhat (Toe)", (p_t[0] + 10, p_t[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

    # Đường đo chiều rộng: khớp bàn ngón (màu xanh dương)
    pw1 = tuple(np.round(pt_w1).astype(int))
    pw2 = tuple(np.round(pt_w2).astype(int))
    cv2.line(vis, pw1, pw2, (255, 120, 0), 3, cv2.LINE_AA)
    cv2.circle(vis, pw1, 5, (255, 120, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, pw2, 5, (255, 120, 0), -1, cv2.LINE_AA)

    # Bảng HUD thông số đo Mốc 1
    banner_w = min(540, int(vis.shape[1] * 0.92))
    cv2.rectangle(vis, (10, 10), (10 + banner_w, 135), (25, 25, 25), -1)
    cv2.rectangle(vis, (10, 10), (10 + banner_w, 135), (0, 220, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "KET QUA DO BAN CHAN BANG OPENCV (MOC 1)", (20, 32), font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Chieu dai (Got -> Ngon dai nhat): {foot_length_cm:.2f} cm ({foot_length_mm:.1f} mm)", (20, 58), font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Chieu rong (Khop ban ngon)     : {foot_width_cm:.2f} cm ({foot_width_mm:.1f} mm)", (20, 82), font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Ty so Rong / Dai (W/L Ratio)   : {wl_ratio:.3f}", (20, 104), font, 0.44, (200, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(vis, f"The ID-1 chuan: 85.60x53.98mm | Ti le: {1.0/mm_per_pixel:.2f} px/mm", (20, 126), font, 0.42, (0, 240, 100), 1, cv2.LINE_AA)

    return {
        "foot_length_px": round(float(foot_length_px), 2),
        "foot_width_px": round(float(foot_width_px), 2),
        "foot_length_mm": round(float(foot_length_mm), 2),
        "foot_width_mm": round(float(foot_width_mm), 2),
        "foot_length_cm": foot_length_cm,
        "foot_width_cm": foot_width_cm,
        "wl_ratio": wl_ratio,
        "measurement_image": vis
    }
