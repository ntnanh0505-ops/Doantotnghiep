"""
main.py — Entry point chính của dự án.

Đồ án tốt nghiệp:
    "Ứng dụng thị giác máy tính và mô hình RAG xây dựng hệ thống
     đo kích thước bàn chân qua ảnh điện thoại và tư vấn chọn size giày"

Cấu trúc dự án 2 giai đoạn:
    src/opencv/   — Giai đoạn 1: OpenCV thuần (detect thẻ → nắn phối cảnh → tách chân → đo PCA)
    src/yolo_y8/  — Giai đoạn 2: YOLOv8n-seg (training, inference, evaluation)

Sử dụng:
    # Chạy OpenCV (giai đoạn 1) — xử lý ảnh đơn:
    python main.py --stage opencv --image path/to/foot_photo.jpg

    # Chạy OpenCV batch toàn bộ dataset:
    python main.py --stage opencv --split all

    # Training YOLO (giai đoạn 2):
    python main.py --stage yolo --action train

    # Inference YOLO:
    python main.py --stage yolo --action predict --weights models/trained/.../best.pt --source data/raw/

    # Evaluate YOLO:
    python main.py --stage yolo --action evaluate --weights models/trained/.../best.pt
"""

import sys
import argparse
from pathlib import Path

# Đảm bảo in tiếng Việt có dấu ổn định trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path để import src.* hoạt động
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_opencv(args):
    """Chạy Giai đoạn 1 — OpenCV thuần."""
    from src.opencv.pipeline import process_image_pipeline, run_batch_pipeline

    if args.image:
        img_p = Path(args.image).resolve()
        if not img_p.exists():
            print(f"[LỖI] Không tìm thấy file ảnh: {args.image}")
            return
        out_dir = Path(args.output).resolve() if args.output else (project_root / "outputs" / "opencv")
        print(f"\n{'='*70}")
        print("XỬ LÝ ẢNH ĐƠN LẺ — GIAI ĐOẠN OPENCV")
        print(f"- File ảnh : {img_p}")
        print(f"- Thư mục  : {out_dir}")
        print(f"- Debug    : {'BẬT' if args.debug else 'TẮT'}")
        print(f"{'='*70}")

        rec = process_image_pipeline(img_p, output_base_dir=out_dir, split_name="single",
                                     save_images=True, debug=args.debug)
        print("\n[KẾT QUẢ XỬ LÝ]")
        print(f"- Trạng thái      : {rec['status']}")
        print(f"- Tìm thấy thẻ ID : {'CÓ' if rec['card_detected'] else 'KHÔNG'}")
        print(f"- Tách bàn chân   : {'CÓ' if rec['segmentation_success'] else 'KHÔNG'}")
        if rec["status"] == "SUCCESS":
            print(f"- Chiều dài chân  : {rec['foot_length_mm']:.1f} mm ({rec['foot_length_mm']/10.0:.2f} cm)")
            print(f"- Chiều rộng chân : {rec['foot_width_mm']:.1f} mm ({rec['foot_width_mm']/10.0:.2f} cm)")
        else:
            print(f"- Lý do thất bại  : {rec['error_message']}")
        print(f"{'='*70}\n")
    else:
        data_dir = Path(args.data).resolve() if args.data else (project_root / "data" / "processed")
        out_dir = Path(args.output).resolve() if args.output else (project_root / "outputs" / "opencv")
        splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
        run_batch_pipeline(dataset_dir=data_dir, output_dir=out_dir, splits=splits, debug=args.debug)


def run_yolo(args):
    """Chạy Giai đoạn 2 — YOLOv8n-seg."""
    action = getattr(args, "action", "train")

    if action == "train":
        from src.yolo_y8.train import train
        train(data_cfg=args.data, epochs=args.epochs, device=args.device, name=args.name)

    elif action == "predict":
        if not args.weights:
            print("[LỖI] --weights là bắt buộc khi action=predict")
            return
        from src.yolo_y8.predict import predict
        predict(weights=args.weights, source=args.source or str(project_root / "data" / "raw"),
                output_dir=args.output, conf=args.conf)

    elif action == "evaluate":
        if not args.weights:
            print("[LỖI] --weights là bắt buộc khi action=evaluate")
            return
        from src.yolo_y8.evaluate import evaluate
        evaluate(weights=args.weights, data_cfg=args.data, device=args.device)

    else:
        print(f"[LỖI] Action không hợp lệ: {action}. Chọn: train | predict | evaluate")


def main():
    parser = argparse.ArgumentParser(
        description="Hệ thống đo kích thước bàn chân — 2 giai đoạn: OpenCV & YOLOv8n-seg",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--stage", "-S", choices=["opencv", "yolo"], default="opencv",
                        help="Chọn giai đoạn: 'opencv' (giai đoạn 1) hoặc 'yolo' (giai đoạn 2)")

    # --- OpenCV args ---
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="[opencv] Ảnh đơn lẻ cần xử lý")
    parser.add_argument("--split", "-s", type=str, choices=["train", "valid", "test", "all"], default="all",
                        help="[opencv] Tập dữ liệu cần chạy batch (mặc định: all)")
    parser.add_argument("--data", "-d", type=str, default=None,
                        help="[opencv/yolo] Thư mục dữ liệu hoặc data.yaml")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Thư mục xuất kết quả")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="[opencv] Bật lưu ảnh debug trung gian")

    # --- YOLO args ---
    parser.add_argument("--action", "-a", choices=["train", "predict", "evaluate"], default="train",
                        help="[yolo] Hành động: train | predict | evaluate")
    parser.add_argument("--weights", "-w", type=str, default=None,
                        help="[yolo] Đường dẫn file weights (.pt)")
    parser.add_argument("--source", type=str, default=None,
                        help="[yolo predict] Ảnh/thư mục nguồn")
    parser.add_argument("--epochs", type=int, default=100,
                        help="[yolo train] Số epoch (mặc định: 100)")
    parser.add_argument("--device", type=str, default="0",
                        help="[yolo] Thiết bị: '0' (GPU) hoặc 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="[yolo predict] Ngưỡng confidence (mặc định: 0.25)")
    parser.add_argument("--name", type=str, default="yolov8n_seg_foot",
                        help="[yolo train] Tên run")

    args = parser.parse_args()

    if args.stage == "opencv":
        run_opencv(args)
    elif args.stage == "yolo":
        run_yolo(args)


if __name__ == "__main__":
    main()
