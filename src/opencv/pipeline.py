"""
pipeline.py — Khối 5: Xử lý pipeline tuần tự và batch (OpenCV giai đoạn 1).

Chức năng:
    - process_image_pipeline(image_path, output_base_dir, ...) :
        Xử lý 1 ảnh qua 4 khối (detect → rectify → segment → measure),
        lưu ảnh kết quả, trả về dict record.
    - run_batch_pipeline(dataset_dir, output_dir, splits, debug) :
        Quét đệ quy train/valid/test, chạy hàng loạt, xuất 2 file CSV:
          outputs/opencv/bang_sai_so_moc_1.csv
          outputs/opencv/bang_ket_qua_do_opencv.csv
"""

import csv
import time
from pathlib import Path

import cv2

from .card_detector import detect_card
from .perspective import rectify_perspective
from .foot_segmenter import segment_foot
from .foot_measurer import measure_foot_pca


def _get_project_root() -> Path:
    """Trả về thư mục gốc của dự án (2 cấp trên src/opencv/)."""
    return Path(__file__).resolve().parent.parent.parent


def _find_images_recursive(folder_path):
    """
    Tìm kiếm đệ quy toàn bộ ảnh trong thư mục với các định dạng:
    JPG, JPEG, PNG, BMP, WEBP (hỗ trợ cả chữ hoa và chữ thường).
    """
    folder = Path(folder_path)
    if not folder.exists():
        return []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(images)


def process_image_pipeline(image_path, output_base_dir, split_name="single", save_images=True, debug=False):
    """
    Xử lý tuần tự 1 ảnh qua 4 khối:
    1. detect_card     → lưu ảnh detected    (_01_the_id.jpg)
    2. rectify_perspective → lưu ảnh rectified (_02_topdown.jpg)
    3. segment_foot    → lưu mask            (_03_mat_na_chan.png)
    4. measure_foot_pca → cập nhật kết quả đo (_04_ket_qua_do_ban_chan.jpg)
    Hỗ trợ chế độ debug lưu các ảnh trung gian chi tiết.
    """
    img_path = Path(image_path)
    stem = img_path.stem
    record = {
        "Ten_anh": img_path.name,
        "Trang_thai_CV": "Thất bại",
        "Dai_OpenCV_cm": "",
        "Rong_OpenCV_cm": "",
        "So_do_thuc_te_cm": "",
        "Sai_so_tuyet_doi_cm": "",
        "Ty_le_W_L": "",
        "Tap_du_lieu": split_name,
        "Trang_thai": "Thất bại",
        "Chieu_dai_cm": "",
        "Chieu_rong_cm": "",
        "Chieu_dai_mm": "",
        "Chieu_rong_mm": "",
        "The_ID_tim_thay": "KHÔNG",
        "Tach_chan_thanh_cong": "KHÔNG",
        "image_name": img_path.name,
        "split": split_name,
        "card_detected": False,
        "segmentation_success": False,
        "foot_length_px": "",
        "foot_width_px": "",
        "foot_length_mm": "",
        "foot_width_mm": "",
        "status": "FAIL",
        "error_message": ""
    }

    img = cv2.imread(str(img_path))
    if img is None:
        record["error_message"] = "Không thể đọc file ảnh"
        return record

    out_base = Path(output_base_dir) / split_name
    eval_dir = Path(output_base_dir) / f"eval_{split_name}"
    dir_detected = out_base / "detected"
    dir_rectified = out_base / "rectified"
    dir_masks = out_base / "masks"
    dir_segmented = out_base / "segmented"
    dir_debug = out_base / "debug"

    if save_images:
        eval_dir.mkdir(parents=True, exist_ok=True)
        for d in [dir_detected, dir_rectified, dir_masks, dir_segmented]:
            d.mkdir(parents=True, exist_ok=True)
        if debug:
            dir_debug.mkdir(parents=True, exist_ok=True)

    # 1. Khối 1: Phát hiện thẻ ID-1
    card_res = detect_card(img, debug=debug)
    if not card_res["success"]:
        record["error_message"] = card_res["error_message"]
        if save_images and card_res["debug_image"] is not None:
            cv2.imwrite(str(dir_detected / f"{stem}_detected.jpg"), card_res["debug_image"])
        return record

    record["card_detected"] = True
    record["The_ID_tim_thay"] = "CÓ"
    if save_images:
        cv2.imwrite(str(dir_detected / f"{stem}_detected.jpg"), card_res["debug_image"])
        cv2.imwrite(str(eval_dir / f"{stem}_01_the_id.jpg"), card_res["debug_image"])
        if debug and card_res.get("debug_maps"):
            if "clahe" in card_res["debug_maps"]:
                cv2.imwrite(str(dir_debug / f"{stem}_debug_01_clahe.jpg"), card_res["debug_maps"]["clahe"])
            if "diff_b" in card_res["debug_maps"]:
                cv2.imwrite(str(dir_debug / f"{stem}_debug_02_diff_b.jpg"), card_res["debug_maps"]["diff_b"])

    # 2. Khối 2: Nắn phối cảnh toàn cảnh Homography
    rect_res = rectify_perspective(img, card_res["corners"])
    rect_img = rect_res["rectified_image"]
    warped_card = rect_res["card_polygon"]
    valid_mask = rect_res["valid_mask"]
    mm_per_pixel = rect_res["mm_per_pixel"]

    if save_images:
        cv2.imwrite(str(dir_rectified / f"{stem}_rectified.jpg"), rect_img)
        cv2.imwrite(str(eval_dir / f"{stem}_02_topdown.jpg"), rect_img)

    # 3. Khối 3: Tách bàn chân bằng OpenCV
    foot_mask, segmented_foot, foot_contour = segment_foot(rect_img, warped_card, valid_mask=valid_mask, debug=debug)
    if foot_mask is None:
        record["error_message"] = "segmentation_failed: Không thể phân đoạn bàn chân bằng thuật toán phân ngưỡng OpenCV"
        return record

    record["segmentation_success"] = True
    record["Tach_chan_thanh_cong"] = "CÓ"
    if save_images:
        cv2.imwrite(str(dir_masks / f"{stem}_mask.png"), foot_mask)
        cv2.imwrite(str(dir_segmented / f"{stem}_segmented.jpg"), segmented_foot)
        cv2.imwrite(str(eval_dir / f"{stem}_03_mat_na_chan.png"), foot_mask)

    # 4. Khối 4: Đo kích thước bằng PCA
    meas_res = measure_foot_pca(rect_img, foot_mask, foot_contour, mm_per_pixel)
    if meas_res is not None:
        record["foot_length_px"] = meas_res["foot_length_px"]
        record["foot_width_px"] = meas_res["foot_width_px"]
        record["foot_length_mm"] = meas_res["foot_length_mm"]
        record["foot_width_mm"] = meas_res["foot_width_mm"]
        record["Chieu_dai_mm"] = meas_res["foot_length_mm"]
        record["Chieu_rong_mm"] = meas_res["foot_width_mm"]
        record["Chieu_dai_cm"] = meas_res["foot_length_cm"]
        record["Chieu_rong_cm"] = meas_res["foot_width_cm"]
        record["Dai_OpenCV_cm"] = meas_res["foot_length_cm"]
        record["Rong_OpenCV_cm"] = meas_res["foot_width_cm"]
        record["Ty_le_W_L"] = meas_res["wl_ratio"]
        record["Trang_thai"] = "Thành công"
        record["Trang_thai_CV"] = "Thành công"
        record["status"] = "SUCCESS"
        record["error_message"] = ""

        if save_images and meas_res["measurement_image"] is not None:
            cv2.imwrite(str(dir_segmented / f"{stem}_segmented.jpg"), meas_res["measurement_image"])
            cv2.imwrite(str(eval_dir / f"{stem}_04_ket_qua_do_ban_chan.jpg"), meas_res["measurement_image"])
            if debug:
                overlay = rect_img.copy()
                cv2.drawContours(overlay, [foot_contour], -1, (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.35, rect_img, 0.65, 0, overlay)
                cv2.imwrite(str(dir_debug / f"{stem}_debug_08_overlay.jpg"), overlay)
    else:
        record["status"] = "FAIL"
        record["error_message"] = "measurement_failed: Không thể tính toán kích thước PCA trên mặt nạ bàn chân"

    return record


def run_batch_pipeline(dataset_dir=None, output_dir=None, splits=None, debug=False):
    """
    Khối 5 — Chạy hàng loạt (Batch Processing):
    - Quét ảnh đệ quy trong các thư mục split: train, valid, test.
    - Lưu ảnh vào đúng cấu trúc outputs/opencv/{split}/...
    - Xuất file CSV tổng hợp vào outputs/opencv/
    """
    project_root = _get_project_root()
    if dataset_dir is None:
        dataset_dir = project_root / "data" / "processed"
    else:
        dataset_dir = Path(dataset_dir)

    if output_dir is None:
        output_dir = project_root / "outputs" / "opencv"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["train", "valid", "test"]
    elif isinstance(splits, str):
        splits = [splits]

    all_records = []
    stats = {}
    t_start_all = time.time()

    print("\n" + "=" * 80)
    print("CHƯƠNG TRÌNH ĐO BÀN CHÂN BẰNG OPENCV THUẦN (MỐC 1)")
    print(f"- Thư mục dữ liệu gốc : {dataset_dir}")
    print(f"- Thư mục xuất kết quả: {output_dir}")
    print(f"- Các tập xử lý      : {', '.join(splits)}")
    print(f"- Chế độ Debug        : {'BẬT' if debug else 'TẮT'}")
    print("=" * 80)

    total_images_all = 0
    total_success_all = 0

    for split in splits:
        split_path = dataset_dir / split
        if not split_path.exists():
            print(f"[CẢNH BÁO] Không tìm thấy thư mục split: {split_path}")
            continue

        images = _find_images_recursive(split_path)
        n_imgs = len(images)
        total_images_all += n_imgs

        print(f"\n--- ĐANG XỬ LÝ TẬP: {split.upper()} ({n_imgs} ảnh) ---")
        card_ok_count = 0
        seg_ok_count = 0
        meas_ok_count = 0
        t_split_start = time.time()

        for idx, img_p in enumerate(images, 1):
            t_img_start = time.time()
            rec = process_image_pipeline(img_p, output_base_dir=output_dir, split_name=split, save_images=True, debug=debug)
            t_img_ms = (time.time() - t_img_start) * 1000.0
            all_records.append(rec)

            if rec["card_detected"]:
                card_ok_count += 1
            if rec["segmentation_success"]:
                seg_ok_count += 1
            if rec["status"] == "SUCCESS":
                meas_ok_count += 1
                total_success_all += 1

            status_str = "Thành công" if rec["status"] == "SUCCESS" else "Thất bại"
            l_str = f"{rec['foot_length_mm']:.1f} mm" if rec["foot_length_mm"] != "" else "N/A"
            w_str = f"{rec['foot_width_mm']:.1f} mm" if rec["foot_width_mm"] != "" else "N/A"
            short_name = img_p.name[:25] + "..." if len(img_p.name) > 28 else img_p.name

            print(f"[{idx:03d}/{n_imgs:03d}] {short_name:<30} -> {status_str:<12} | Dài: {l_str:<10} | Rộng: {w_str:<10} | {t_img_ms:5.1f}ms")

        elapsed_split = time.time() - t_split_start
        stats[split] = {
            "total": n_imgs,
            "card_ok": card_ok_count,
            "seg_ok": seg_ok_count,
            "success": meas_ok_count,
            "rate": (meas_ok_count / max(1, n_imgs)) * 100.0,
            "time_s": elapsed_split
        }

    # 1. Ghi file CSV theo đúng chuẩn Đề cương Mốc 1: bang_sai_so_moc_1.csv
    csv_moc1 = output_dir / "bang_sai_so_moc_1.csv"
    fieldnames_moc1 = [
        "Ten_anh", "Trang_thai_CV", "Dai_OpenCV_cm", "Rong_OpenCV_cm",
        "So_do_thuc_te_cm", "Sai_so_tuyet_doi_cm", "Ty_le_W_L",
        "Tap_du_lieu", "The_ID_tim_thay", "Tach_chan_thanh_cong", "error_message"
    ]
    with open(csv_moc1, mode="w", newline="", encoding="utf-8-sig") as f:
        writer1 = csv.DictWriter(f, fieldnames=fieldnames_moc1, extrasaction="ignore")
        writer1.writeheader()
        writer1.writerows(all_records)

    # 2. Ghi song song file bang_ket_qua_do_opencv.csv
    csv_file = output_dir / "bang_ket_qua_do_opencv.csv"
    fieldnames_full = [
        "Tap_du_lieu", "Ten_anh", "Trang_thai",
        "Chieu_dai_cm", "Chieu_rong_cm", "Chieu_dai_mm", "Chieu_rong_mm",
        "The_ID_tim_thay", "Tach_chan_thanh_cong",
        "image_name", "split", "card_detected", "segmentation_success",
        "foot_length_px", "foot_width_px", "foot_length_mm", "foot_width_mm",
        "status", "error_message"
    ]
    with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer2 = csv.DictWriter(f, fieldnames=fieldnames_full, extrasaction="ignore")
        writer2.writeheader()
        writer2.writerows(all_records)

    elapsed_all = time.time() - t_start_all
    overall_rate = (total_success_all / max(1, total_images_all)) * 100.0

    print("\n" + "=" * 80)
    print("      TỔNG KẾT KẾT QUẢ THỰC NGHIỆM OPENCV THUẦN (MỐC 1)")
    print("=" * 80)
    print(f"{'Tập dữ liệu':<15} | {'Tổng ảnh':<10} | {'Tìm thẻ OK':<12} | {'Tách chân OK':<14} | {'Thành công':<12} | {'Tỷ lệ':<8}")
    print("-" * 80)
    for sp, s in stats.items():
        print(f"{sp:<15} | {s['total']:<10} | {s['card_ok']:<12} | {s['seg_ok']:<14} | {s['success']:<12} | {s['rate']:6.2f} %")
    print("-" * 80)
    print(f"{'TỔNG CỘNG':<15} | {total_images_all:<10} | {'-':<12} | {'-':<14} | {total_success_all:<12} | {overall_rate:6.2f} %")
    print("=" * 80)
    print(f"[BẢNG SAI SỐ MỐC 1]: {csv_moc1}")
    print(f"[BẢNG TỔNG HỢP CSV ]: {csv_file}")
    print(f"[THỜI GIAN CHẠY]: {elapsed_all:.2f} giây")
    print("=" * 80 + "\n")

    return {
        "csv_file": str(csv_file),
        "stats": stats,
        "total_images": total_images_all,
        "total_success": total_success_all,
        "overall_rate": overall_rate
    }
