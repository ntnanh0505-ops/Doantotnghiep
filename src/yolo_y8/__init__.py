"""
src/yolo_y8 — Giai đoạn 2: Training và inference YOLOv8n-seg
Đồ án: "Ứng dụng thị giác máy tính đo kích thước bàn chân qua ảnh điện thoại"

Các module:
    train    — Huấn luyện model YOLOv8n-seg (Ultralytics)
    predict  — Inference và visualize kết quả
    evaluate — Đánh giá model (mAP, IoU, so sánh với OpenCV)
    pipeline — Pipeline end-to-end: detect + measure với YOLO model
"""
