# Yêu cầu thu thập dữ liệu bàn chân

## 1. Số lượng

- Ít nhất **30 người**, mỗi người chụp **cả 2 chân**.
- Mỗi chân chụp **3–4 ảnh** (tổng khoảng 180–240 ảnh).
- Nên có cả nam và nữ, chiều dài chân khoảng 220–290 mm.

## 2. Cách chụp

- Chụp **từ trên xuống**, điện thoại song song mặt sàn, cao khoảng 40–60 cm.
- Thấy **đủ gót và 5 ngón chân**. Cẳng chân có thể ra ngoài khung hình.
- Đặt **thẻ ID-1** (CCCD, thẻ SV, thẻ ngân hàng…) cạnh chân, **cùng mặt phẳng sàn**, không cong, không bị che, cách chân khoảng 2 cm.
- Chân chiếm khoảng 30–50% khung hình, thẻ rõ nét.
- Mỗi chân chụp **ít nhất 2 loại nền** và **2 điều kiện ánh sáng** khác nhau; 1–2 ảnh có thể chụp hơi nghiêng (~15–20°).
- Dùng **ảnh gốc**: không crop, không bộ lọc, không chế độ làm đẹp.
- Nộp qua **Google Drive / file zip**. Không gửi qua Zalo, Messenger (bị nén ảnh, mất thông tin xoay ảnh).

## 3. Cách đo số đo thật

Đứng thẳng, dồn trọng lượng lên chân cần đo, đo trên tờ giấy phẳng bằng thước có chia mm. Đo **2 lần**, ghi **giá trị trung bình**.

- **Chiều dài (`length_mm`)**: gót chạm sát tường, đánh dấu đầu ngón dài nhất, đo từ tường đến vạch (song song trục chân, không đo chéo).
- **Chiều rộng (`width_mm`)**: chỗ rộng nhất của bàn chân (phần lòng ngón), đo vuông góc với chiều dài.

## 4. Cấu trúc thư mục nộp

```
dataset_v2/
├── images/          # ảnh gốc
├── subjects.csv     # số đo thật (1 dòng = 1 bàn chân)
└── images.csv       # thông tin từng ảnh (1 dòng = 1 ảnh)
```

**Tên ảnh:** `P{3 số}_{R|L}_{2 số}.jpg`. Ví dụ `P012_L_03.jpg` (người 012, chân trái, ảnh số 3).

## 5. `subjects.csv`

```csv
person_id,foot,length_mm,width_mm,gender,age
P001,R,251.5,98,M,21
P001,L,250,97.5,M,21
```

- `foot`: `R` (phải) hoặc `L` (trái).
- `gender`: `M` hoặc `F`.
- Số đo tính bằng **mm**, dùng dấu chấm cho số thập phân.

## 6. `images.csv`

```csv
file,person_id,foot,background,lighting,tilt,phone,card_type
P001_R_01.jpg,P001,R,san_go,den_tran,thang,iPhone13,cccd
P001_R_02.jpg,P001,R,gach,ban_ngay,nghieng_nhe,iPhone13,the_sv
```

| Cột | Giá trị cho phép |
|---|---|
| `background` | `giay_a4`, `san_go`, `gach`, `tham`, `khac` |
| `lighting` | `ban_ngay`, `den_tran`, `nguoc_sang`, `toi` |
| `tilt` | `thang` (lệch ≤ 10°), `nghieng_nhe` (10–20°), `nghieng_manh` (> 20°) |
| `card_type` | `cccd`, `the_sv`, `the_ngan_hang`, `bang_lai` |
| `phone` | tên máy, ví dụ `iPhone13`, `SamsungA54` |

## 7. Lưu file CSV

- Lưu dạng **CSV UTF-8** (Excel: *Save As → CSV UTF-8*), phân cách bằng **dấu phẩy**.
- Dòng đầu là tên cột đúng như mẫu, không gộp ô, không để dòng trống.
