"""
train.py — Huấn luyện model YOLOv8n-seg (Giai đoạn 2).

Sử dụng:
    python -m src.yolo_y8.train [--data configs/yolo_config.yaml] [--epochs 100] [--device mps]

Workflow:
    1. Đọc configs/yolo_config.yaml (dataset + siêu tham số trong train_args)
    2. Load YOLOv8n-seg pretrained weights (khóa `model` trong config)
    3. Train trên dataset data/yolo_dataset/, lưu kết quả vào models/trained/<name>/
"""

import argparse
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Trả về thư mục gốc dự án."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_device(device: str = None) -> str:
    """Chọn thiết bị: giá trị người dùng nếu có, không thì CUDA > MPS (Mac) > CPU."""
    if device:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(data_cfg: str = None) -> dict:
    """Đọc configs/yolo_config.yaml (hoặc file được chỉ định)."""
    cfg_path = Path(data_cfg) if data_cfg else get_project_root() / "configs" / "yolo_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_yaml(cfg: dict) -> str:
    """
    Ghi file dataset yaml với `path` tuyệt đối rồi trả về đường dẫn file.

    Ultralytics không hiểu `path` tương đối theo gốc dự án, còn config thì phải giữ đường dẫn
    tương đối. File sinh ra nằm trong models/trained/ (đã gitignore) và bị ghi đè mỗi lần chạy.
    """
    root = get_project_root()
    ds = {k: cfg[k] for k in ("train", "val", "test", "names") if k in cfg}
    ds["path"] = str((root / cfg["path"]).resolve())
    out = root / "models" / "trained" / "dataset_resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(ds, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(out)


def train(
    data_cfg: str = None,
    epochs: int = None,
    imgsz: int = None,
    batch: int = None,
    device: str = None,
    pretrained: str = None,
    project: str = None,
    name: str = None,
    resume: bool = False,
):
    """
    Khởi chạy training YOLOv8n-seg. Tham số để None sẽ lấy từ configs/yolo_config.yaml.

    Args:
        data_cfg  : Đường dẫn config (mặc định configs/yolo_config.yaml)
        epochs    : Số epoch huấn luyện
        imgsz     : Kích thước ảnh đầu vào (px)
        batch     : Batch size (-1 = auto)
        device    : Thiết bị ('0' GPU / 'mps' / 'cpu'); mặc định tự chọn
        pretrained: Đường dẫn weights khởi đầu (mặc định khóa `model` trong config)
        project   : Thư mục lưu kết quả training
        name      : Tên run
        resume    : Tiếp tục từ checkpoint last.pt của run `name`
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics chưa được cài đặt. Chạy: pip install -r requirements.txt"
        )

    root = get_project_root()
    cfg = load_config(data_cfg)
    args = dict(cfg.get("train_args", {}))
    for key, value in (("epochs", epochs), ("imgsz", imgsz), ("batch", batch), ("name", name), ("project", project)):
        if value is not None:
            args[key] = value
    args["project"] = str((root / args.get("project", "models/trained")).resolve())
    args.setdefault("name", "yolov8n_seg_foot")
    args["device"] = resolve_device(device)
    args["data"] = dataset_yaml(cfg)

    pretrained = str((root / (pretrained or cfg["model"])).resolve())
    if resume:
        pretrained = str(Path(args["project"]) / args["name"] / "weights" / "last.pt")
        args["resume"] = True
    if not Path(pretrained).exists():
        raise FileNotFoundError(f"Không tìm thấy weights: {pretrained}")

    print(f"\n{'='*60}")
    print("  TRAINING YOLOv8n-seg — Giai đoạn 2")
    print(f"{'='*60}")
    print(f"  Dataset        : {args['data']}")
    print(f"  Pretrained     : {pretrained}")
    print(f"  Epochs         : {args.get('epochs')}")
    print(f"  Imgsz          : {args.get('imgsz')}")
    print(f"  Batch          : {args.get('batch')}")
    print(f"  Device         : {args['device']}")
    print(f"  Output project : {args['project']}/{args['name']}")
    print(f"{'='*60}\n")

    model = YOLO(pretrained)
    return model.train(**args)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n-seg để phân đoạn bàn chân và thẻ ID")
    parser.add_argument("--data", type=str, default=None, help="Đường dẫn config (mặc định: configs/yolo_config.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch (mặc định: theo config)")
    parser.add_argument("--imgsz", type=int, default=None, help="Kích thước ảnh (mặc định: theo config)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (mặc định: theo config)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị: '0' (GPU), 'mps' (Mac), 'cpu'; mặc định tự chọn")
    parser.add_argument("--name", type=str, default=None, help="Tên run (mặc định: theo config)")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint cuối của run --name")
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
