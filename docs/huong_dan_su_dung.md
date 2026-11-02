# Hướng dẫn sử dụng: đo kích thước bàn chân bằng thẻ ID-1

Chương trình nhận **một ảnh** bàn chân chụp từ trên xuống, có đặt kèm một thẻ ID-1 (CCCD, thẻ sinh viên, thẻ ngân hàng) làm thước,
và trả về **chiều dài, chiều rộng bàn chân (cm)**. Thẻ có kích thước chuẩn 85,60 × 53,98 mm.

- **Không quy đổi sang cỡ giày.** Chương trình chỉ trả cm; bên máy chủ tự đổi sang cỡ theo bảng cỡ của mình.
- **Không thấy gót thì từ chối đo.** Không trả số cm, chỉ trả lý do để người dùng chụp lại (mục 6).

## 1. Chương trình làm gì

```
ảnh ─► thu nhỏ về tối đa Full HD (~2 MP) nếu ảnh lớn hơn
    ─► kiểm tra chất lượng (độ nét, độ sáng)
    ─► YOLOv8n-seg tách mặt nạ bàn chân và thẻ
    ─► lấy 4 góc thẻ, nắn phối cảnh (thẻ làm thước)
    ─► tìm trục bàn chân, kiểm tra có thấy gót không ── không thấy ─► TỪ CHỐI, trả lý do
    ─► đo chiều dài / chiều rộng, hiệu chỉnh độ cao (mu bàn chân cao hơn mặt thẻ)
    ─► chiều dài: mô hình hồi quy học từ số đo thật; chiều rộng: số đo mặt nạ đã hiệu chỉnh
    ─► trả cm
```

## 2. Quy định ảnh đầu vào

- **Kích thước:** nhận ảnh JPG, PNG, BMP, WEBP bất kỳ cỡ nào. Ảnh lớn hơn khung **Full HD (1920 × 1080, tối đa ~2 MP)** được tự động
  thu nhỏ trước khi xử lý (giữ tỉ lệ, không phóng to). Ví dụ ảnh điện thoại 12 MP (3024 × 4032) thu nhỏ về 1080 × 1440. Điện thoại nào
  cũng chụp được độ phân giải này; chỉnh khung ở `configs/yolo_config.yaml`, khóa `input` (mục 9).
- **Độ phân giải nên dùng:** từ khoảng 1 MP trở lên (bộ ảnh thử là 960 × 1280). Ảnh nhỏ hơn chưa được kiểm tra.
- **Cách chụp** (theo [yeu_cau_thu_thap_du_lieu.md](yeu_cau_thu_thap_du_lieu.md)):
  - Chụp **từ trên xuống**, điện thoại song song mặt sàn, cao khoảng 40–60 cm.
  - Phải thấy **đủ gót và 5 ngón chân**. Cẳng chân nên ra ngoài khung hình; ống quần hoặc cổ chân che gót thì ảnh bị từ chối.
  - Thẻ đặt **cạnh chân, cùng mặt sàn**, thẳng, không cong, không bị che, cách chân khoảng 2 cm, thấy đủ 4 cạnh.
  - Chân chiếm khoảng 30–50% khung hình, ảnh rõ nét, đủ sáng, dùng ảnh gốc (không làm đẹp, không gửi qua ứng dụng nén ảnh).

## 3. Yêu cầu hệ thống và RAM

Đo trong môi trường Python 3.12 mới cài, chỉ có torch bản CPU, trên Linux. **Không cần GPU.**

| Trường hợp | RAM tối đa (RSS) | Thời gian |
|---|---|---|
| Nạp mô hình, chưa xử lý ảnh | ~290 MB | ~0,7 giây |
| 1 ảnh 960 × 1280 | ~590 MB | ~1,9 giây (gồm nạp mô hình) |
| 1 ảnh 12 MP (3024 × 4032, thu nhỏ về 1080 × 1440) | ~580 MB | ~1,8 giây |
| 105 ảnh trong một lần chạy | ~630 MB (không tăng theo số ảnh) | ~30 giây (~0,29 giây/ảnh) |

- **Dành 1 GB RAM trống cho mỗi tiến trình.** Máy 2 GB chạy được một tiến trình; chạy song song thì thêm khoảng 0,6 GB cho mỗi
  tiến trình. Vì ảnh luôn được thu nhỏ về tối đa ~2 MP nên RAM không tăng theo cỡ ảnh gốc.
- Chạy bằng GPU thì RAM hệ thống cao hơn (~2,3 GB vì nạp thư viện CUDA), GPU chỉ dùng khoảng 70–90 MB VRAM, và không nhanh hơn
  đáng kể với mô hình nhỏ này.
- Ổ đĩa: môi trường Python (torch CPU, ultralytics, OpenCV) khoảng 1,5 GB; mã nguồn kèm mô hình khoảng 8 MB.

## 4. Cài đặt

Lấy mã (đã kèm sẵn mô hình, không cần huấn luyện) rồi cài thư viện:

```bash
git clone https://github.com/ntnanh0505-ops/Doantotnghiep.git
cd Doantotnghiep
python3.12 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Cài torch bản CPU trước để môi trường nhẹ (bản mặc định kèm CUDA nặng hơn nhiều, không cần). Sau đó chạy ngay, không phải tải thêm gì.

Các file cần để chạy (đều nằm trong git):

```
main.py                                            điểm chạy
src/                                               mã nguồn (doan.py là pipeline chính)
configs/yolo_config.yaml                           đường dẫn mô hình và tham số đo
models/trained/yolov8n_seg_foot/weights/best.pt    mô hình YOLOv8n-seg (7 MB)
models/trained/length_model.json                   mô hình chiều dài (hồi quy)
data/yolo_dataset/                                 bộ dữ liệu (chỉ cần khi huấn luyện / đánh giá)
```

Gói zip `foot-measure-v1.zip` (nếu được gửi riêng) chỉ gồm phần chạy, kèm 4 ảnh mẫu trong `samples/`.

## 5. Chạy chương trình

**Luôn thêm `--method yolo --no-gui`.** Mặc định của chương trình là phương pháp OpenCV thuần (sai số lớn, hơn 10 cm); `--no-gui`
tắt cửa sổ hiển thị để chạy được trên máy chủ không có màn hình. Ảnh mẫu `P038_R_01.jpg` nằm ở
`data/yolo_dataset/images/test/` (trong gói zip: `samples/`).

### 5.1. Tích hợp vào máy chủ: `--json`

```bash
python main.py --image data/yolo_dataset/images/test/P038_R_01.jpg --method yolo --no-gui --json
```

In **đúng 1 dòng JSON** ra stdout (báo cáo chi tiết chuyển sang stderr). Đơn vị là **cm**:

```json
{"ok": true, "length_cm": 22.94, "width_cm": 9.63, "error": ""}
```

Khi từ chối hoặc lỗi, `ok` là `false`, hai số là `null` và `error` nêu lý do:

```json
{"ok": false, "length_cm": null, "width_cm": null, "error": "Không thấy gót (gót bị cổ chân/ống quần che): từ chối đo. Chụp lại từ trên xuống sao cho thấy rõ gót và 5 ngón chân, cẳng chân nằm ngoài khung hình"}
```

Bên máy chủ nên: nếu `ok` là `true` thì đổi `length_cm` (và `width_cm` nếu cần) sang cỡ theo bảng cỡ của mình; nếu `false` thì hiển thị
`error` cho người dùng và yêu cầu chụp lại. Các `error` có thể gặp ở mục 8. Chương trình thoát với mã 0 khi chạy được; hãy dựa vào
trường `ok`.

### 5.2. Báo cáo chi tiết và ảnh minh họa

```bash
python main.py --image data/yolo_dataset/images/test/P038_R_01.jpg --method yolo --no-gui
```

In báo cáo từng bước, ví dụ:

```
5. KẾT QUẢ ĐO ĐẠC BÀN CHÂN (PCA):
   - Chiều dài lớn nhất (Gót -> Ngón): 229.4 mm (22.94 cm | 1575.3 px) [mô hình hồi quy học từ số đo thật]
   - Chiều rộng lớn nhất (1/3 trước) : 96.3 mm (9.63 cm | 677.5 px)
   - Đã hiệu chỉnh độ cao (mu bàn chân cao hơn mặt thẻ): dài x0.884, rộng x0.847 (chưa hiệu chỉnh: 26.45 x 11.38 cm)
   - Thấy gót                        : CÓ
```

và lưu ảnh vào `outputs/single_yolo/`: `*_ket_qua_do.jpg` (ảnh nắn phối cảnh, đường đỏ là chiều dài, đường xanh dương là chiều rộng),
`*_xoay_dung.jpg`, `*_the_id.jpg` (thẻ phát hiện được). Ảnh bị từ chối không lưu ảnh kết quả.

### 5.3. Cả thư mục ảnh

```bash
python main.py --data ./anh_cua_toi --method yolo --no-gui
```

Quét cả thư mục con (đuôi .jpg, .jpeg, .png, .bmp, .webp) và ghi `outputs/bang_ket_qua_do_test_yolo.csv` (khi không có `--split` thì
tên tập là `test`). Các cột chính:

| Cột | Ý nghĩa |
|---|---|
| `Trang_thai` | `Thành công` hoặc `Thất bại` (lý do ở `error_message`; ảnh bị từ chối vì che gót cũng là `Thất bại`) |
| `Chieu_dai_cm`, `Chieu_rong_cm` | số đo cuối cùng (cm); để trống khi thất bại |
| `Chieu_dai_cm_chua_hieu_chinh`, `Chieu_rong_cm_chua_hieu_chinh` | số đo trực tiếp từ mặt nạ, chưa hiệu chỉnh độ cao |
| `Thay_got` | `CÓ` hoặc `KHÔNG` |
| `Ty_le_W_L` | chiều rộng / chiều dài |
| `Dang_ban_chan` | dáng chân Ai Cập / La Mã / Hy Lạp (chỉ tham khảo, chưa được kiểm chứng) |
| `Do_sac_net`, `Do_sang` | điểm kiểm tra chất lượng ảnh |
| `Sai_so_goc_chup_pct` | mức biến dạng phối cảnh ước tính từ thẻ |

### 5.4. Tùy chọn

| Tùy chọn | Ý nghĩa |
|---|---|
| `--image`, `-i` | ảnh đơn (hoặc ghi đường dẫn ảnh ngay sau `main.py`) |
| `--data`, `-d` | thư mục ảnh để chạy hàng loạt |
| `--method`, `-m` | `yolo` (khuyên dùng) hoặc `opencv` |
| `--json` | ảnh đơn: in 1 dòng JSON ra stdout (mục 5.1) |
| `--weights` | dùng file `.pt` khác thay cho mô hình mặc định |
| `--output`, `-o` | thư mục lưu kết quả (mặc định `outputs/`) |
| `--no-gui` | không mở cửa sổ |

## 6. Từ chối đo khi che gót

Mặt nạ bàn chân chỉ cho biết hình bóng. Khi cổ chân, cẳng chân hoặc ống quần che gót, mặt nạ dừng ở chỗ bị che và không có cách nào
biết gót thật nằm đâu, nên số đo chiều dài có thể sai hơn 1 cm. Vì vậy chương trình **từ chối đo** các ảnh đó (khóa
`measure.require_heel` trong cấu hình, mặc định bật).

Chương trình xem là "thấy gót" khi đầu sau mặt nạ tròn lồi (độ lồi của 25% đầu sau ≥ 0,91) và không có chỗ lõm sâu hơn 25 mm (mép
ống quần cắt vào làm lõm). Cần biết:

- Trong bộ ảnh thử, phần lớn ảnh không thấy rõ gót, nên **tỉ lệ bị từ chối cao**: 37/53 ảnh test, 24/49 ảnh valid, 109/232 ảnh train.
  Chụp đúng quy trình (mục 2) thì tỉ lệ này thấp hơn nhiều.
- Điều kiện này chưa hoàn hảo: nhận đúng 7/7 ảnh thấy gót (xem bằng mắt) nhưng vẫn nhận nhầm khoảng 8–9/46 ảnh bị che. Số đo của các
  ảnh nhận nhầm đó có thể sai trên 1 cm. Kết quả ổn định khi ảnh được lưu lại JPEG, phóng hoặc thu nhỏ, hay đổi độ sáng 10%.

## 7. Độ chính xác đã đo

Trên các ảnh **được nhận** (đã qua bước từ chối), so với số đo thật:

| Tập | Ảnh được nhận | Dài: sai số TB | Dài: trong ±1 cm | Rộng: sai số TB |
|---|---|---|---|---|
| Test (7 người, không dùng để huấn luyện) | 16/53 | 0,98 cm | 56% | 0,82 cm |
| Valid | 25/49 | 0,84 cm | 60% | 0,92 cm |
| 7 ảnh test thấy gót (xem bằng mắt) | 7/7 | 0,80 cm | 86% | 0,73 cm |

- Trong ±0,5 cm: chiều dài 12% (test), chiều rộng 44% (test).
- Mô hình phân đoạn: mAP50-95 mặt nạ 0,96 trên tập test.
- Kiểm tra bằng chính thẻ: thẻ đo ra lệch khoảng 1% so với 85,6 × 53,98 mm (khoảng 0,2 cm với bàn chân 23 cm), nên phép đo và tỷ lệ
  mm/px không phải nguồn sai số chính.

Giới hạn khi dùng thực tế:

- Chỉ huấn luyện và kiểm tra với **39 người, 1 loại điện thoại (iPhone 13), nền gạch, đèn trần**. Điện thoại khác hoặc ánh sáng khác
  chưa kiểm chứng.
- Chiều dài lấy từ mô hình hồi quy nên bị "kéo về mức trung bình" của dữ liệu huấn luyện (21–25 cm): bàn chân rất ngắn hoặc rất dài có
  thể bị đo lệch về phía giữa. Chiều rộng kém ổn định hơn chiều dài.
- Số đo thật làm tròn 0,5 cm và mỗi người chỉ đo một lần, nên sai số dưới ~0,3 cm không còn ý nghĩa.
- Không phải thiết bị y tế; chỉ dùng để tham khảo khi chọn giày.

## 8. Lỗi thường gặp

| `error` | Nguyên nhân và cách xử lý |
|---|---|
| `Không thấy gót (gót bị cổ chân/ống quần che): từ chối đo...` | gót bị che; chụp lại thấy rõ gót và 5 ngón, cẳng chân ra ngoài khung |
| `Ảnh mờ/nhòe (Laplacian var = ... < 80)` | ảnh không nét; chụp lại, giữ máy chắc, đủ sáng |
| `Ảnh quá tối` / `Ảnh quá sáng/cháy sáng` | chỉnh ánh sáng (độ sáng trung bình phải trong 50–215) |
| `YOLO không phát hiện được thẻ ID-1` | thẻ bị che, quá nhỏ hoặc lẫn màu nền; đặt thẻ rõ ràng cạnh chân |
| `Tỷ lệ cạnh thẻ YOLO bất thường (...)` | thấy không đủ 4 cạnh thẻ, thẻ cong hoặc chụp quá nghiêng; chụp thẳng từ trên xuống |
| `Không thể tách mặt nạ bàn chân khỏi nền sàn` | không tìm thấy bàn chân; kiểm tra chân nằm trong khung |
| `Không thể đọc file ảnh` / `Không tìm thấy file ảnh` | sai đường dẫn hoặc file không phải ảnh |
| `FileNotFoundError: Không tìm thấy weights` | thiếu `models/trained/yolov8n_seg_foot/weights/best.pt`; kiểm tra thư mục hoặc dùng `--weights` |

Ảnh được thu nhỏ trước bước kiểm tra độ nét; ngưỡng độ nét 80 được đặt cho ảnh cỡ ~1–2 MP.

## 9. Cấu hình

`configs/yolo_config.yaml`:

| Khóa | Ý nghĩa |
|---|---|
| `input.max_long_side`, `input.max_short_side` | khung tối đa của ảnh (mặc định 1920 × 1080); ảnh lớn hơn bị thu nhỏ |
| `infer.weights` | đường dẫn mô hình YOLO |
| `infer.conf`, `infer.iou`, `infer.imgsz` | ngưỡng tin cậy, ngưỡng NMS, kích thước ảnh khi suy luận (640) |
| `measure.require_heel` | `true`: từ chối ảnh che gót (mặc định); `false`: vẫn đo (chỉ dùng khi nghiên cứu) |
| `measure.height_correction` | hiệu chỉnh độ cao: `enabled`, `focal_ratio`, `h_length_cm`, `h_width_cm` |
| `measure.length_model` | mô hình chiều dài: `enabled`, `path`; `enabled: false` thì chiều dài dùng số đo mặt nạ đã hiệu chỉnh độ cao |

Các tham số hiệu chỉnh độ cao được tìm cho iPhone 13 (`focal_ratio` 0,72 là tiêu cự ước tính / cạnh dài ảnh). Dùng điện thoại khác thì
cần tìm lại (mục 10).

## 10. Huấn luyện lại và đánh giá (cần bộ dữ liệu `data/yolo_dataset/`, huấn luyện nên dùng máy có GPU)

```bash
python -m src.yolo_y8.train                       # huấn luyện YOLOv8n-seg
python -m src.yolo_y8.train --fit-length          # học lại mô hình chiều dài (cần chạy lại mỗi khi đổi mô hình YOLO)
python -m src.yolo_y8.evaluate --measure --split test                    # sai số cm; ảnh che gót bị từ chối như khi triển khai
python -m src.yolo_y8.evaluate --measure --split test --include-hidden   # vẫn đo cả ảnh che gót (để so sánh)
python -m src.yolo_y8.evaluate --measure --split test --heel-only        # chỉ 7 ảnh thấy gót (cột thay_got trong images_meta.csv)
```

`--fit-length` chỉ dùng ảnh train và valid có số đo thật, kể cả ảnh che gót (nhiều ảnh hơn nên mô hình tốt hơn); tập test không tham
gia. Mô hình chiều dài lưu ở `models/trained/length_model.json`. Máy không có GPU không cần huấn luyện: mô hình đã có sẵn trong git.
