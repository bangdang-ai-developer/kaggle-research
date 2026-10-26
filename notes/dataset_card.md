# Dataset Card — NWPU-RESISC45

**Ngày audit:** 2026-10-26 (local) — sẽ bổ sung audit đầy đủ bằng notebook 010 chạy trên Kaggle.

## Thông tin chung

| Thuộc tính | Giá trị | Xác minh |
|---|---|---|
| Tên | NWPU-RESISC45 | paper gốc arXiv:1703.00121 |
| Nguồn | Northwestern Polytechnical University (Cheng et al., 2017) | ✓ paper |
| Bản Kaggle | `punfake/resisc45` (usability 0.31; các bản khác cùng size 427.052.154 bytes → cùng nguồn) | ✓ `kaggle datasets files` |
| Số ảnh | 31.500 | ✓ `find ... -name '*.jpg' | wc -l` local |
| Số lớp | 45 | ✓ `ls` local |
| Ảnh/lớp | 700 | ✓ đếm `airplane` = 700 (audit notebook sẽ đếm đủ 45) |
| Kích thước ảnh | mô tả 256×256 RGB | ⏳ notebook 010 sẽ đo phân bố thật |
| GSD | 0.2–30 m/px (3 mức) | paper (không có metadata per-image trong bản Kaggle) |

## Cấu trúc

```
NWPU-RESISC45/
  <class_name>/          # 45 thư mục
    <class_name>_001.jpg ... <class_name>_700.jpg
```

Tên lớp = tên thư mục, sorted alphabetically → class index (lưu `class_order.json` khi build split).

## Rủi ro chất lượng dữ liệu — checklist audit (notebook 010)

- [ ] Decode lỗi / ảnh 0 byte
- [ ] Phân bố kích thước thật (có ảnh ≠ 256×256 không)
- [ ] Exact duplicate (hash nội dung MD5/SHA1 toàn ảnh)
- [ ] Near-duplicate (pHash/aHash Hamming ≤ ngưỡng; sample ứng viên kiểm tra thủ công)
- [ ] Trùng ảnh khác nhãn (duplicate group đa lớp)
- [ ] Số ảnh/lớp ≠ 700
- [ ] RGB thật (không grayscale/RGBA lệch)

## Protocol split (áp cho MỌI model)

- **75/10/15 stratified:** train 525 / val 70 / test 105 mỗi lớp (chênh lệch do ràng buộc duplicate-group được ghi thật).
- **Duplicate group:** mọi ảnh trong cùng group (exact hoặc near-dup xác nhận) nằm cùng 1 split. Xử lý: gom group → chia theo group, giữ tỉ lệ gần 75/10/15.
- **Seed split:** `SPLIT_SEED = 42`, tạo 1 lần, lưu `split_manifest.csv` (image_id, relpath, class, class_index, split, dup_group) + checksum SHA256 của file manifest. Mọi notebook đọc manifest, không tạo split riêng.
- **Test locked:** không dùng cho bất kỳ lựa chọn nào; chỉ chạy lần cuối sau `final_plan.json`.

## Không làm

- Không tạo ảnh tổng hợp để lấp đầy số lượng (mọi lớp đã đủ 700).
- Không tuyên bố generalization theo địa lý.
- Không dùng val làm dữ liệu huấn luyện confusion graph của CMP.
