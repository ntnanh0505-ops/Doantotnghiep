"""
predict.py — Inference với YOLOv8n-seg model đã train (Giai đoạn 2).

Sử dụng:
    python -m src.yolo_y8.predict --weights models/trained/yolov8n_seg_foot/weights/best.pt
                                  --source data/raw/foot_photos/
                                  --output outputs/yolo_y8/predictions/
"""

import argparse
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def predict(
    weights: str,
    source: str,
    output_dir: str = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "0",
    save_txt: bool = True,
    save_conf: bool = True,
):
    """
    Chạy inference YOLOv8n-seg trên ảnh / thư mục ảnh.

    Args:
        weights   : Đường dẫn file weights best.pt
        source    : File ảnh, thư mục, hoặc glob pattern
        output_dir: Thư mục lưu kết quả (mặc định outputs/yolo_y8/predictions/)
        conf      : Ngưỡng confidence
        iou       : Ngưỡng IoU cho NMS
        imgsz     : Kích thước ảnh inference
        device    : Thiết bị
        save_txt  : Lưu labels dạng txt
        save_conf : Lưu confidence score trong txt
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics chưa được cài đặt. Chạy: pip install ultralytics")

    root = get_project_root()
    if output_dir is None:
        output_dir = str(root / "outputs" / "yolo_y8" / "predictions")

    print(f"\n{'='*60}")
    print("  INFERENCE YOLOv8n-seg — Giai đoạn 2")
    print(f"{'='*60}")
    print(f"  Weights  : {weights}")
    print(f"  Source   : {source}")
    print(f"  Output   : {output_dir}")
    print(f"  Conf     : {conf}  |  IoU: {iou}")
    print(f"{'='*60}\n")

    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        project=output_dir,
        name="run",
        stream=False,
    )
    return results


def main():
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Inference YOLOv8n-seg phân đoạn bàn chân và thẻ ID")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Ảnh, thư mục hoặc glob pattern")
    parser.add_argument("--output", type=str, default=None, help="Thư mục lưu kết quả")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence (mặc định: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng IoU NMS (mặc định: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh (mặc định: 640)")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị: '0' hoặc 'cpu'")
    args = parser.parse_args()

    predict(
        weights=args.weights,
        source=args.source,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
