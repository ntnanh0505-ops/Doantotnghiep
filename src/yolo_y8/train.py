"""
train.py — Huấn luyện model YOLOv8n-seg (Giai đoạn 2).

Sử dụng:
    python -m src.yolo_y8.train [--data configs/yolo_config.yaml] [--epochs 100] [--device mps]

    python -m src.yolo_y8.train --fit-length     # học mô hình chiều dài từ số đo thật (cần model YOLO đã train)

Workflow:
    1. Đọc configs/yolo_config.yaml (dataset + siêu tham số trong train_args)
    2. Load YOLOv8n-seg pretrained weights (khóa `model` trong config)
    3. Train trên dataset data/yolo_dataset/, lưu kết quả vào models/trained/<name>/
"""

import argparse
import json
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
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


def _fit_ridge(x, y, alpha):
    """Ridge trên đặc trưng đã chuẩn hóa. Trả về (hệ số, hằng số)."""
    coef = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - y.mean()))
    return coef, float(y.mean())


def fit_length_model(data_cfg: str = None, device: str = None):
    """
    Học mô hình chiều dài (hồi quy Ridge) từ số đo thật trong data/yolo_dataset/subjects_clean.csv.

    Dữ liệu: ảnh train+valid có tên P###_{L|R}_##, chạy pipeline --method yolo để lấy đặc trưng (doan.length_features), kể cả ảnh che gót
    (chỉ dùng ảnh thấy gót thì còn 123 ảnh và mô hình kém hơn: MAE 1,27 so với 1,01 cm). Tập test không tham gia. Hệ số alpha chọn bằng cách bỏ ra từng người (không để ảnh cùng người nằm ở cả hai phía).
    Lưu hệ số vào đường dẫn measure.length_model.path trong config (mặc định models/trained/length_model.json).
    """
    from .. import doan
    from .evaluate import load_ground_truth
    from .predict import load_detector

    cfg = load_config(data_cfg)
    out_path = get_project_root() / cfg["measure"]["length_model"]["path"]
    gt = load_ground_truth()
    detector = load_detector(device=device)

    feats, ys, persons = [], [], []
    for split in ("train", "valid"):
        for p in doan.load_dataset(split=split):
            m = re.match(r"P(\d+)_([LR])_", Path(p).name)
            if not m or (int(m.group(1)), m.group(2)) not in gt:
                continue
            with redirect_stdout(StringIO()):
                rec = doan.process_image_pipeline(p, split_name=split, save_images=False, method="yolo", detector=detector,
                                                  require_heel=False)  # lấy cả ảnh che gót: nhiều ảnh hơn, mô hình tốt hơn
            if rec["Trang_thai"] == "Thành công":
                feats.append(rec["length_features"])
                ys.append(gt[(int(m.group(1)), m.group(2))][0])
                persons.append(int(m.group(1)))
    x, y, persons = np.array(feats, dtype=np.float64), np.array(ys), np.array(persons)
    mean, scale = x.mean(0), np.where(x.std(0) > 1e-9, x.std(0), 1.0)
    xs = (x - mean) / scale

    def lopo_mae(alpha):
        err = []
        for person in np.unique(persons):
            tr, te = persons != person, persons == person
            coef, intercept = _fit_ridge(xs[tr], y[tr], alpha)
            err += list(np.abs(intercept + xs[te] @ coef - y[te]))
        return float(np.mean(err))

    alphas = np.logspace(-2, 3, 20)
    best_alpha = float(min(alphas, key=lopo_mae))
    baseline = float(np.mean([np.abs(y[persons != q].mean() - y[persons == q]).mean() for q in np.unique(persons)]))
    coef, intercept = _fit_ridge(xs, y, best_alpha)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "features": doan.LENGTH_FEATURES, "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(),
        "intercept": intercept, "alpha": best_alpha, "n_images": int(len(y)), "n_persons": int(len(np.unique(persons))),
        "trained_on": ["train", "valid"], "lopo_mae_cm": round(lopo_mae(best_alpha), 3),
    }, indent=2), encoding="utf-8")
    print(f"[MÔ HÌNH CHIỀU DÀI] {len(y)} ảnh, {len(np.unique(persons))} người, alpha={best_alpha:.3g}")
    print(f"  MAE bỏ-từng-người: {lopo_mae(best_alpha):.2f} cm (luôn đoán trung bình: {baseline:.2f} cm)")
    print(f"  Đã lưu: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n-seg để phân đoạn bàn chân và thẻ ID")
    parser.add_argument("--data", type=str, default=None, help="Đường dẫn config (mặc định: configs/yolo_config.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch (mặc định: theo config)")
    parser.add_argument("--imgsz", type=int, default=None, help="Kích thước ảnh (mặc định: theo config)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (mặc định: theo config)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị: '0' (GPU), 'mps' (Mac), 'cpu'; mặc định tự chọn")
    parser.add_argument("--name", type=str, default=None, help="Tên run (mặc định: theo config)")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint cuối của run --name")
    parser.add_argument("--fit-length", action="store_true", help="Học mô hình chiều dài từ số đo thật (không train YOLO)")
    args = parser.parse_args()

    if args.fit_length:
        fit_length_model(data_cfg=args.data, device=args.device)
        return

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
