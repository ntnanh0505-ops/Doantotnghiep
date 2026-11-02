"""
predict.py — Inference với YOLOv8n-seg model đã train (Giai đoạn 2).

Sử dụng:
    python -m src.yolo_y8.predict --weights models/trained/yolov8n_seg_foot/weights/best.pt
                                  --source data/yolo_dataset/images/test
                                  --output outputs/yolo_y8/predictions/
"""

import argparse
from pathlib import Path

import numpy as np

from .train import get_project_root, load_config, resolve_device

FOOT_CLASS, CARD_CLASS = 0, 1


def load_detector(weights: str = None, device: str = None) -> dict:
    """
    Nạp model cho pipeline đo. Weights và ngưỡng lấy từ khóa `infer` trong configs/yolo_config.yaml.

    Trả về dict gồm model và tham số inference để truyền cho detect_foot_card().
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics chưa được cài đặt. Chạy: pip install -r requirements.txt")

    infer = load_config().get("infer", {})
    path = Path(weights or infer.get("weights", "models/trained/yolov8n_seg_foot/weights/best.pt"))
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy weights: {path} (train bằng python -m src.yolo_y8.train)")

    return {
        "model": YOLO(str(path)),
        "conf": infer.get("conf", 0.25),
        "iou": infer.get("iou", 0.45),
        "imgsz": infer.get("imgsz", 640),
        "retina_masks": infer.get("retina_masks", True),
        "device": resolve_device(device),
    }


def detect_foot_card(detector: dict, image: np.ndarray) -> dict:
    """
    Phân đoạn bàn chân và thẻ trên 1 ảnh BGR. Mỗi lớp chỉ giữ đối tượng có conf cao nhất
    (đánh giá test: ~12% ảnh có thêm 1 đối tượng thừa nhỏ hơn nhiều, luôn có conf thấp hơn).

    Trả về {"foot_polygon", "card_polygon"} (ndarray Nx2 theo pixel ảnh gốc, hoặc None nếu không thấy),
    cùng "foot_conf", "card_conf", "n_foot", "n_card".
    """
    r = detector["model"].predict(
        image,
        conf=detector["conf"],
        iou=detector["iou"],
        imgsz=detector["imgsz"],
        retina_masks=detector["retina_masks"],
        device=detector["device"],
        verbose=False,
    )[0]

    out = {}
    cls = r.boxes.cls.cpu().numpy().astype(int) if r.masks is not None else np.array([], dtype=int)
    conf = r.boxes.conf.cpu().numpy() if r.masks is not None else np.array([])
    for name, c in (("foot", FOOT_CLASS), ("card", CARD_CLASS)):
        idx = np.flatnonzero(cls == c)
        out[f"n_{name}"] = int(len(idx))
        if len(idx) == 0:
            out[f"{name}_polygon"], out[f"{name}_conf"] = None, 0.0
            continue
        best = idx[np.argmax(conf[idx])]
        out[f"{name}_polygon"] = r.masks.xy[best].astype(np.float32)
        out[f"{name}_conf"] = float(conf[best])
    return out


def predict(
    weights: str,
    source: str,
    output_dir: str = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = None,
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
        device    : Thiết bị (mặc định tự chọn CUDA > MPS > CPU)
        save_txt  : Lưu labels dạng txt
        save_conf : Lưu confidence score trong txt
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics chưa được cài đặt. Chạy: pip install ultralytics")

    root = get_project_root()
    device = resolve_device(device)
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
        exist_ok=True,
        stream=False,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference YOLOv8n-seg phân đoạn bàn chân và thẻ ID")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Ảnh, thư mục hoặc glob pattern")
    parser.add_argument("--output", type=str, default=None, help="Thư mục lưu kết quả")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence (mặc định: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng IoU NMS (mặc định: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh (mặc định: 640)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị: '0', 'mps', 'cpu'; mặc định tự chọn")
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
