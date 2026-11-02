"""
evaluate.py — Đánh giá model YOLOv8n-seg và so sánh với OpenCV (Giai đoạn 2).

Sử dụng:
    python -m src.yolo_y8.evaluate --weights models/trained/yolov8n_seg_foot/weights/best.pt
                                   --data configs/yolo_config.yaml

    So sánh số đo cm (OpenCV và YOLO) với số đo thật trong data/yolo_dataset/subjects_clean.csv:
    python -m src.yolo_y8.evaluate --measure --split test
"""

import argparse
import csv
import io
import re
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np

from .train import dataset_yaml, get_project_root, load_config, resolve_device


def load_ground_truth() -> dict:
    """Số đo thật {(số người, 'L'|'R'): (dài_cm, rộng_cm)} từ data/yolo_dataset/subjects_clean.csv."""
    path = get_project_root() / "data" / "yolo_dataset" / "subjects_clean.csv"
    gt = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["length_cm_raw"] and r["width_cm_raw"]:
                gt[(int(r["person_id"].strip().lstrip("P")), r["foot"].strip())] = (
                    float(r["length_cm_raw"]), float(r["width_cm_raw"]))
    return gt


def load_heel_labels() -> dict:
    """{tên ảnh: 'CÓ'|'KHÔNG'} từ cột thay_got của data/yolo_dataset/images_meta.csv (xem bằng mắt, hiện chỉ có cho ảnh test)."""
    path = get_project_root() / "data" / "yolo_dataset" / "images_meta.csv"
    with open(path, encoding="utf-8-sig") as f:
        return {r["file"]: r["thay_got"] for r in csv.DictReader(f) if r.get("thay_got")}


def evaluate_measurements(split: str = "test", methods=("opencv", "yolo"), weights: str = None, device: str = None,
                          heel_only: bool = False, include_hidden: bool = False):
    """
    Chạy pipeline đo trên các ảnh của `split` có số đo thật (ảnh tên P###_{L|R}_##.jpg) và báo sai số cm.
    Ảnh kiểu timestamp (Roboflow) không có mã người nên bị bỏ qua.
    Lưu từng ảnh vào outputs/yolo_y8/measure_<split>.csv.
    heel_only=True: chỉ lấy ảnh được xem bằng mắt là thấy gót (cột thay_got trong images_meta.csv).
    Mặc định ảnh che gót bị từ chối như khi triển khai (không có số cm, tính vào số ảnh bị từ chối); include_hidden=True thì vẫn đo.

    Lưu ý: số đo thật chỉ ghi đến ~0,5 cm nên MAE dưới ~0,25 cm không còn ý nghĩa phân biệt.
    """
    from .. import doan
    from .predict import load_detector

    root = get_project_root()
    gt = load_ground_truth()
    detector = load_detector(weights, device) if "yolo" in methods else None

    items = []
    for p in doan.load_dataset(split=split):
        m = re.match(r"P(\d+)_([LR])_", p.replace("\\", "/").split("/")[-1])
        if m and (int(m.group(1)), m.group(2)) in gt:
            items.append((p, gt[(int(m.group(1)), m.group(2))]))
    if heel_only:
        heel = load_heel_labels()
        items = [(p, g) for p, g in items if heel.get(Path(p).name) == "CÓ"]
    print(f"[ĐO CM] split={split}{' (chỉ ảnh thấy gót)' if heel_only else ''}: {len(items)} ảnh có số đo thật, phương pháp: {', '.join(methods)}")

    rows = []
    for method in methods:
        for i, (p, (gl, gw)) in enumerate(items, 1):
            with redirect_stdout(io.StringIO()):
                rec = doan.process_image_pipeline(p, split_name=split, save_images=False,
                                                  method=method, detector=detector, require_heel=not include_hidden)
            ok = rec["Trang_thai"] == "Thành công"
            rows.append({
                "method": method, "file": rec["Ten_anh"], "ok": ok, "error": rec["error_message"],
                "heel": rec["Thay_got"],
                "length_cm": rec["Chieu_dai_cm"] if ok else "", "width_cm": rec["Chieu_rong_cm"] if ok else "",
                "gt_length_cm": gl, "gt_width_cm": gw,
                "err_length_cm": round(rec["Chieu_dai_cm"] - gl, 2) if ok else "",
                "err_width_cm": round(rec["Chieu_rong_cm"] - gw, 2) if ok else "",
            })

    out = root / "outputs" / "yolo_y8" / f"measure_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'Phương pháp':<16} {'Đo được':>9} | {'Dài MAE':>8} {'bias':>7} {'|e|≤0.5':>8} {'|e|≤1.0':>8} | {'Rộng MAE':>8} {'bias':>7} {'|e|≤0.5':>8}")
    for method in methods:
        rs = [r for r in rows if r["method"] == method]
        for label, ok in ((method, [r for r in rs if r["ok"]]),
                          (f"{method}/thấy gót", [r for r in rs if r["ok"] and r["heel"] == "CÓ"])):
            if not ok:
                print(f"{label:<16} {0:>4}/{len(rs):<4}")
                continue
            el = np.array([r["err_length_cm"] for r in ok])
            ew = np.array([r["err_width_cm"] for r in ok])
            print(f"{label:<16} {len(ok):>4}/{len(rs):<4} | {np.abs(el).mean():>8.2f} {el.mean():>+7.2f} {(np.abs(el) <= 0.5).mean():>8.0%} {(np.abs(el) <= 1.0).mean():>8.0%}"
                  f" | {np.abs(ew).mean():>8.2f} {ew.mean():>+7.2f} {(np.abs(ew) <= 0.5).mean():>8.0%}")
    for method in methods:
        rejected = [r for r in rows if r["method"] == method and not r["ok"] and r["error"].startswith("Không thấy gót")]
        print(f"[{method}] từ chối vì che gót: {len(rejected)}/{len(items)} ảnh")
    print(f"\nChi tiết từng ảnh: {out}")
    return rows


def evaluate(weights: str, data_cfg: str = None, imgsz: int = 640, device: str = None):
    """
    Đánh giá model YOLOv8n-seg trên tập test.

    Metrics đầu ra:
        - mAP50, mAP50-95 (theo lớp: foot, card)
        - Precision, Recall
        - IoU phân đoạn instance
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics chưa được cài đặt. Chạy: pip install ultralytics")

    root = get_project_root()
    data_yaml = dataset_yaml(load_config(data_cfg))
    device = resolve_device(device)

    print(f"\n{'='*60}")
    print("  EVALUATE YOLOv8n-seg — Giai đoạn 2")
    print(f"{'='*60}")
    print(f"  Weights : {weights}")
    print(f"  Data    : {data_yaml}")
    print(f"  Imgsz   : {imgsz}  |  Device: {device}")
    print(f"{'='*60}\n")

    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        split="test",
        project=str(root / "outputs" / "yolo_y8"),
        name="eval",
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )

    print("\n[KẾT QUẢ ĐÁNH GIÁ]")
    print(f"  mAP50        : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95     : {metrics.seg.map:.4f}")
    print(f"  Precision    : {metrics.seg.mp:.4f}")
    print(f"  Recall       : {metrics.seg.mr:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Đánh giá YOLOv8n-seg trên tập test")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn weights (.pt); mặc định theo infer.weights trong config")
    parser.add_argument("--measure", action="store_true", help="So sánh số đo cm của OpenCV và YOLO với số đo thật")
    parser.add_argument("--heel-only", action="store_true", help="Với --measure: chỉ ảnh được xem bằng mắt là thấy gót (cột thay_got)")
    parser.add_argument("--include-hidden", action="store_true", help="Với --measure: vẫn đo cả ảnh che gót (mặc định từ chối như khi triển khai)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"], help="Tập dùng cho --measure")
    parser.add_argument("--data", type=str, default=None, help="Đường dẫn config (mặc định: configs/yolo_config.yaml)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị: '0', 'mps', 'cpu'; mặc định tự chọn")
    args = parser.parse_args()

    if args.measure:
        evaluate_measurements(split=args.split, weights=args.weights, device=args.device, heel_only=args.heel_only,
                             include_hidden=args.include_hidden)
        return
    weights = args.weights or load_config(args.data).get("infer", {}).get("weights")
    evaluate(weights=str(get_project_root() / weights), data_cfg=args.data, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
