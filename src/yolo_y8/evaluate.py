"""
evaluate.py — Đánh giá model YOLOv8n-seg và so sánh với OpenCV (Giai đoạn 2).

Sử dụng:
    python -m src.yolo_y8.evaluate --weights models/trained/yolov8n_seg_foot/weights/best.pt
                                   --data configs/yolo_config.yaml
"""

import argparse
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def evaluate(weights: str, data_cfg: str = None, imgsz: int = 640, device: str = "0"):
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
    if data_cfg is None:
        data_cfg = str(root / "configs" / "yolo_config.yaml")

    print(f"\n{'='*60}")
    print("  EVALUATE YOLOv8n-seg — Giai đoạn 2")
    print(f"{'='*60}")
    print(f"  Weights : {weights}")
    print(f"  Data    : {data_cfg}")
    print(f"  Imgsz   : {imgsz}  |  Device: {device}")
    print(f"{'='*60}\n")

    model = YOLO(weights)
    metrics = model.val(
        data=data_cfg,
        imgsz=imgsz,
        device=device,
        split="test",
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
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn weights (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Đường dẫn data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị")
    args = parser.parse_args()

    evaluate(weights=args.weights, data_cfg=args.data, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
