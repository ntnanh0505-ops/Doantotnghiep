"""
train.py — Huấn luyện model YOLOv8n-seg (Giai đoạn 2).

Sử dụng:
    python -m src.yolo_y8.train [--data configs/yolo_config.yaml] [--epochs 100] [--device cpu]
    # hoặc
    python scripts/train_yolo.py

Workflow:
    1. Load YOLOv8n-seg pretrained weights từ models/pretrained/yolov8n-seg.pt
    2. Train trên dataset data/yolo_dataset/ với cấu hình configs/yolo_config.yaml
    3. Lưu kết quả vào models/trained/runs/segment/
"""

import argparse
from pathlib import Path


def get_project_root() -> Path:
    """Trả về thư mục gốc dự án."""
    return Path(__file__).resolve().parent.parent.parent


def train(
    data_cfg: str = None,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    pretrained: str = None,
    project: str = None,
    name: str = "yolov8n_seg_foot",
    resume: bool = False,
):
    """
    Khởi chạy training YOLOv8n-seg.

    Args:
        data_cfg  : Đường dẫn tới data.yaml (mặc định configs/yolo_config.yaml)
        epochs    : Số epoch huấn luyện
        imgsz     : Kích thước ảnh đầu vào (px)
        batch     : Batch size (-1 = auto)
        device    : Thiết bị ('0' GPU / 'cpu')
        pretrained: Đường dẫn weights khởi đầu (mặc định models/pretrained/yolov8n-seg.pt)
        project   : Thư mục lưu kết quả training
        name      : Tên run
        resume    : Tiếp tục từ checkpoint cuối
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics chưa được cài đặt. Chạy: pip install ultralytics"
        )

    root = get_project_root()

    if data_cfg is None:
        data_cfg = str(root / "configs" / "yolo_config.yaml")
    if pretrained is None:
        pretrained = str(root / "models" / "pretrained" / "yolov8n-seg.pt")
    if project is None:
        project = str(root / "models" / "trained")

    print(f"\n{'='*60}")
    print("  TRAINING YOLOv8n-seg — Giai đoạn 2")
    print(f"{'='*60}")
    print(f"  Dataset config : {data_cfg}")
    print(f"  Pretrained     : {pretrained}")
    print(f"  Epochs         : {epochs}")
    print(f"  Imgsz          : {imgsz}")
    print(f"  Batch          : {batch}")
    print(f"  Device         : {device}")
    print(f"  Output project : {project}/{name}")
    print(f"{'='*60}\n")

    model = YOLO(pretrained)
    results = model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n-seg để phân đoạn bàn chân và thẻ ID")
    parser.add_argument("--data", type=str, default=None, help="Đường dẫn data.yaml (mặc định: configs/yolo_config.yaml)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epoch (mặc định: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh (mặc định: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (mặc định: 16)")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị: '0' (GPU) hoặc 'cpu'")
    parser.add_argument("--name", type=str, default="yolov8n_seg_foot", help="Tên run")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint cuối")
    args = parser.parse_args()

    train(
        data_cfg=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
