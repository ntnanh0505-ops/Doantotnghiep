# AGENTS.md — Quy tắc cho AI agent (và người) làm việc trong repo này

Đồ án: đo kích thước bàn chân bằng thẻ ID-1 tham chiếu (OpenCV, giai đoạn 1) và YOLOv8n-seg (giai đoạn 2).

Repo này từng bị đẩy lên hơn 7.000 file rác (ảnh debug, pycache, dataset, bản sao code, skill AI), khiến `.git` nặng 640MB và người hướng dẫn phải viết lại lịch sử. **Mọi quy tắc dưới đây là bắt buộc.**

## 1. Cấu trúc — chỉ những thứ này được nằm trong git

```
main.py                 Entry point, gọi src.doan.main()
src/doan.py             Pipeline chính (bản DUY NHẤT, sửa trực tiếp ở đây)
src/opencv/             Module OpenCV: card_detector, perspective, foot_segmenter, foot_measurer, pipeline
src/yolo_y8/            train.py, predict.py, evaluate.py
configs/                opencv_config.yaml, yolo_config.yaml
data/yolo_dataset/      Bộ train sạch (images/, labels/, data.yaml, CSV). Chỉ thư mục này được commit
docs/                   Đề cương, tài liệu báo cáo
scripts/                Script tiện ích lâu dài (không phải script thử nghiệm)
requirements.txt
```

Muốn thêm thư mục cấp gốc mới thì phải hỏi người dùng trước.

## 2. Không tạo file rác

- **Không tạo bản sao hay phiên bản mới của code**: không `doan_v2.py`, `main_new.py`, `test_v3.py`, `code_moi/`, `chuongtrinhchay/`, `lamlai.py`. Sửa thẳng vào file đang có. Lịch sử thay đổi đã có git lưu.
- **Không tạo file ở thư mục gốc**, ngoại trừ khi người dùng yêu cầu rõ ràng.
- **Script thử nghiệm hay debug tạm thời** chỉ đặt trong `scratch/` (đã gitignore). Xong việc thì xóa đi. Không bao giờ commit chúng.
- **Mọi ảnh và CSV kết quả** phải ghi vào `outputs/` (đã gitignore), ví dụ `outputs/opencv/...`, `outputs/debug/...`. Không `cv2.imwrite` ra thư mục gốc, `src/` hay `data/`.
- **Không tạo project hoặc app khác trong repo này** (web bán hàng, demo Flask…). Nếu cần, phải hỏi người dùng và đặt trong `apps/`.
- Không tạo notebook `.ipynb`, file `.log` hay file markdown ghi chú/tóm tắt nếu người dùng không yêu cầu.
- Trước khi tạo file mới, tìm xem đã có file làm việc đó chưa và mở rộng file đó.

## 3. Dữ liệu, model, đường dẫn

- `data/yolo_dataset/` (bộ train sạch, ~70MB) **được commit** để chuyển sang máy train. Mọi thứ khác trong `data/` (`raw_images/`, `roboflow_export/`, bản backup…) **không đưa lên git**, chia sẻ qua Drive/Roboflow. Không commit `labels/*.cache`.
- Không tự thay đổi `data/yolo_dataset/` (thêm/xóa/đổi tên ảnh) nếu người dùng không yêu cầu, vì mỗi lần commit lại ảnh sẽ làm `.git` phình thêm.
- Weights (`*.pt`, `*.pth`, `*.onnx`), `runs/`, `models/trained/` **không commit**. Những thứ này tải lại hoặc train lại được.
- **Chỉ dùng đường dẫn tương đối** tính từ gốc dự án (`get_project_root()` / `Path(__file__)`). Không ghi cứng `C:/Users/...` vào code hay file yaml rồi commit.
- Tham số (ngưỡng, kích thước thẻ…) đặt trong `configs/*.yaml`, không ghi cứng rải rác trong code.

## 4. Công cụ AI (BMAD, Claude, Antigravity, Codex…)

- `.agent/`, `.agents/`, `.claude/`, `_bmad/`, `_bmad-output/` là cài đặt riêng của từng máy: **được dùng, không được commit**. Chúng đã có trong `.gitignore`.
- Tài liệu sinh ra từ các tool đó (PRD, story, ghi chú) không đưa vào repo trừ khi người dùng yêu cầu. Nếu được yêu cầu thì đặt trong `docs/`.

## 5. Git — quy trình đúng

Trước khi bắt đầu làm:
```
git pull
```

Trước khi commit, **luôn kiểm tra danh sách file**:
```
git status
git diff --stat
```

- Chỉ `git add <file cụ thể>`. **Không `git add .` hoặc `git add -A`** khi chưa đọc `git status`.
- **Không bao giờ dùng `git add -f` / `--force`**. File bị ignore là có lý do.
- Nếu `git status` hiện ảnh, `.pyc`, `__pycache__`, `outputs/`, `data/` (ngoài `data/yolo_dataset/`), `.pt` hay thư mục AI thì **dừng lại, không commit**, và báo người dùng.
- Mỗi commit chỉ làm một việc, kèm message mô tả rõ, ví dụ `Sửa phát hiện thẻ khi ảnh bị lóa`. Không dùng message kiểu `k`, `update`, `fix`.
- **Không `git push --force`, không viết lại lịch sử, không xóa nhánh** nếu người dùng không yêu cầu rõ ràng.
- Khi pull bị conflict thì giải quyết conflict. Không xóa file của người khác để cho qua.
- Mỗi commit phải nhỏ (vài trăm KB). Nếu diff có file > 1MB thì kiểm tra lại. Ngoại lệ: commit đưa `data/yolo_dataset/` lên, và commit đó chỉ chứa thư mục này.

## 6. Chạy chương trình

```
pip install -r requirements.txt

python main.py --image <đường_dẫn_ảnh>          # 1 ảnh
python main.py --split test --no-gui            # chạy cả tập test (data/yolo_dataset/images/test)
python -m src.yolo_y8.train --data configs/yolo_config.yaml
python -m src.yolo_y8.predict --weights models/trained/.../best.pt --source <ảnh|thư mục>
```

Kết quả nằm trong `outputs/`. Muốn chia sẻ kết quả thì gửi bảng CSV hoặc ảnh tiêu biểu qua kênh khác, hoặc đưa vào báo cáo trong `docs/`. Không commit `outputs/`.

## 7. Checklist trước khi báo "xong"

- [ ] Không có file mới nào ngoài những file thật sự cần, và không có bản sao code.
- [ ] `scratch/` đã dọn, file tạm đã xóa.
- [ ] `git status` sạch, chỉ còn đúng các file đã định sửa.
- [ ] Code vẫn chạy được: `python main.py --split test --no-gui`.
