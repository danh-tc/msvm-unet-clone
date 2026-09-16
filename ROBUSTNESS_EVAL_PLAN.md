# Kế hoạch đánh giá Robustness cho MSVM-UNet (Zero-Shot)

Trạng thái: **đang lên kế hoạch** — file này là nguồn thông tin duy nhất về
scope/quyết định. Cập nhật trực tiếp vào file này khi có thay đổi, không bàn
lại những điểm đã chốt ở chat.

## Mục tiêu

Đo mức độ suy giảm chất lượng segmentation (DSC, HD95) của MSVM-UNet khi ảnh
CT đầu vào bị nhiễu bởi các tác nhân thực tế khi chụp (nhiễu cảm biến, mờ,
lệch tương phản), ở nhiều mức độ tăng dần. Checkpoint đóng băng (frozen),
chỉ inference — **không train lại** ở phase này.

## Bối cảnh (vì sao pivot sang hướng này)

- Hướng trước đó (SAM-Med2D / MedSAM refine hậu kỳ mask của MSVM-UNet) không
  vượt được baseline — xem lịch sử `infer_single_sam.py` / log hội thoại.
  Box, box+mask, box+points đều cho DSC bằng hoặc thấp hơn baseline trên
  case0001 (81.32% sạch → 79-80.55% sau refine, mọi biến thể).
- Bỏ hẳn hướng SAM refine. Kế hoạch này không đụng tới SAM/MedSAM.
- So sánh chéo kiến trúc khác (UNet/SwinUNet/...) so với MSVM-UNet: **ngoài
  scope phase 1** — `model/` có sẵn code nhiều kiến trúc nhưng `log/` chỉ có
  checkpoint của MSVM-UNet; train các model khác từ đầu là một việc lớn và
  tách biệt. Chỉ xem lại sau khi phase 1 xong.

## Phạm vi (Scope)

**Trong scope**: làm nhiễu 12 volume test của Synapse lúc inference, chạy qua
checkpoint MSVM-UNet đã đóng băng, đo mức suy giảm DSC/HD95 so với baseline
sạch, tách theo loại nhiễu / mức độ / từng organ.

**Ngoài scope (phase 1)**: train lại, data augmentation để khắc phục, kiến
trúc khác, SAM/MedSAM.

## Các thông tin đã xác nhận thực tế (không phải giả định)

- Ảnh trong `data/Synapse/test_vol_h5/*.npy.h5` là `float32` trong khoảng
  **[0.0, 1.0]** (đã được windowed/normalize sẵn, không phải HU thô).
- Inference của MSVM-UNet (`infer_single.py: predict_volume`) chạy theo từng
  slice 2D: resize slice về 224×224 (`scipy.ndimage.zoom`, order=3) →
  `Normalize([0.5],[0.5])` → forward qua model → argmax → resize prediction
  về lại kích thước gốc (order=0).
- Không có TTA / multi-scale / sliding-window / post-processing ở bất kỳ đâu
  trong pipeline gốc (`test.py`) — chỉ chạy 1 lần duy nhất, khớp với
  `infer_single.py`.
- Checkpoint: `log/msvm_unet-synapse-r0/checkpoints/epoch.259-val_mean_dice.0.8500.ckpt`
  (chỉ là 1 trong 3 seed train của paper — chưa có r1/r2 ở local).
- Đã có đủ 12/12 volume test, khớp với `lists/lists_Synapse/test.txt`.

## Bộ corruption

| Corruption | Mô phỏng cho | Áp dụng | Tham số mức độ |
|---|---|---|---|
| Gaussian noise | nhiễu điện tử của cảm biến | theo từng slice | độ lệch chuẩn noise (đơn vị cường độ [0,1]) |
| Poisson noise | nhiễu lượng tử/photon, CT liều thấp | theo từng slice | hệ số scale (proxy cho số lượng photon) |
| Gaussian blur | mờ do tái tạo ảnh/defocus | theo từng slice | kích thước kernel |
| Contrast shift | trôi hiệu chỉnh máy quét | **theo từng volume** (1 giá trị dùng chung cho toàn bộ slice của 1 case — 1 ca chụp thật không tự đổi tương phản ngẫu nhiên theo từng slice) | hệ số shift/scale |

Đã loại khỏi đề xuất gốc: Salt & Pepper (không phải artifact cảm biến CT thực
tế — thay bằng Poisson). Bỏ cách gọi "motion blur" (chỉ dùng "Gaussian blur" =
defocus) trừ khi sau này cài thêm kernel có hướng thật sự.

Mức độ (severity): 0 (sạch, dùng để sanity-check phải khớp chính xác baseline
sạch hiện có) đến 3-4 mức tăng dần cho mỗi loại nhiễu. Con số cụ thể sẽ chốt
lúc code — chọn giá trị trải từ "gần như không thấy" đến "suy giảm rõ nhưng
chưa phá hỏng hoàn toàn" (kiểm tra bằng histogram/nhìn trực quan, không chọn
mù).

## Lưu ý khi implement (đã thống nhất, bắt buộc giữ)

1. Corruption áp lên **slice ở độ phân giải gốc**, *trước* bước resize
   512→224 và *trước* `Normalize([0.5],[0.5])` — không áp sau, để severity
   (VD kích thước kernel blur) có ý nghĩa nhất quán ở độ phân giải chụp thật,
   không phụ thuộc resize nội bộ của model.
2. Contrast shift chỉ **random 1 lần cho mỗi volume**, dùng lại cho mọi slice
   của volume đó — không random lại theo từng slice.
3. Phải xử lý tường minh range [0,1] float32 — không dựa vào default của
   albumentations/MONAI vốn giả định ảnh uint8/RGB.
4. Severity=0 phải cho ra đúng DSC baseline sạch hiện có (sanity-check lỗi
   wiring) trước khi tin vào bất kỳ con số suy giảm nào.

## Yêu cầu về baseline

`infer_single.py` hiện chưa có chế độ `--all` để tổng hợp (chỉ
`infer_single_sam.py` có). Cần thêm để đo baseline sạch trên đủ 12 case test
theo đúng protocol như các lần chạy có nhiễu — không lẫn số case0001 đơn lẻ
trước đó (81.32%) với số tổng hợp toàn tập.

## Metric & output

- DSC, HD95 theo từng class, từng mức severity, từng loại corruption (tái
  dùng `calc_dsc_hd95` trong `infer_single.py`).
- Degradation rate (% giảm tương đối so với baseline sạch), so sánh giữa organ
  nhỏ (GB, PC) và organ lớn (Liver, SP).
- Line chart: trục X = mức severity, trục Y = mean DSC, mỗi đường ứng với 1
  loại corruption.
- Một số hình minh họa mask dự đoán vỡ dần khi corruption tăng.

## Rủi ro

- Thư viện tạo nhiễu âm thầm không có tác dụng hoặc clip sai range ảnh float
  [0,1] → kiểm tra histogram trước/sau khi áp corruption, cho từng loại.
- Lỗi wiring (sai thứ tự resize, sai thứ tự normalize) bị hiểu nhầm thành
  "model kém robust" → severity=0 sanity-check (ở trên) sẽ bắt được lỗi này.

## Checklist công việc

- [x] Thêm chế độ `--all` tổng hợp vào `infer_single.py` để đo baseline sạch
- [x] Chạy baseline sạch trên đủ 12 case test, ghi lại mean DSC/HD95 (số tham chiếu) — **mean DSC = 83.96%, HD95 = 14.74mm** (khớp số plan gốc trích dẫn). GB 66.57%, PC 71.53% yếu nhất & biến động cao nhất (std lớn)
- [x] Viết các hàm corruption (Gaussian noise, Poisson noise, Gaussian blur, contrast shift), xử lý tường minh range [0,1] float — `corruptions.py`
- [x] Chốt mức severity cụ thể cho từng loại corruption (kiểm tra trực quan/histogram) — blur kernel tăng từ [3,5,7,9] lên [5,9,15,21] sau khi thấy quá yếu ở lần đầu
- [x] Gắn corruption vào đúng vị trí trong vòng lặp inference (độ phân giải gốc, trước resize/normalize) — `infer_robustness.py`, corrupt volume trước khi gọi `predict_volume`
- [x] **Smoke test tầng 0** (không cần model): áp từng hàm corruption lên 1 slice thật (case0001), kiểm tra min/max/histogram trước-sau không NaN/inf/tràn range [0,1], xuất ảnh before/after nhìn trực quan
- [x] **Smoke test tầng 1** (wiring check): chạy full pipeline severity=0 trên 1 volume, xác nhận DSC khớp chính xác baseline sạch của volume đó — case0001: 81.32% khớp 100%
- [x] **Smoke test tầng 2** (integration nhỏ): 1 volume × 4 corruption × 2 severity (nhẹ nhất + nặng nhất) — severity=1 gần như phẳng ở cả 4 loại (đúng hướng); severity=4: gaussian_noise -2.74pp, poisson_noise -4.38pp, gaussian_blur -2.32pp, **contrast_shift -14.99pp (bất thường mạnh, HD95 20.14mm→78.05mm)**
- [x] Check thêm contrast_shift severity 2,3 trên case0001 (chạy song song) — full curve DSC: 81.32/82.23/83.27/78.63/66.33% (sev0-4), HD95: 20.14/19.90/29.99/63.56/78.05mm. **Không suy giảm dần đều** — DSC còn nhích nhẹ ở sev1-2 rồi "gãy" ở sev3-4; nhưng **HD95 đã tăng rõ từ sev2** trong khi DSC còn trông ổn → HD95 là chỉ báo sớm nhạy hơn DSC cho corruption này
- [ ] **Rủi ro mới phát hiện**: `medpy.metric.binary.hd95` chậm bất thường (vài phút thay vì vài giây) ở severity cao — do distance-transform trên mask bị suy biến (degenerate) tốn hơn nhiều so với mask "đẹp". Ước tính thời gian full sweep cần điều chỉnh tăng (có thể 1-2 tiếng ngay cả khi chạy song song 4 job theo corruption), không chỉ dựa trên thời gian inference thuần như tính ban đầu
- [ ] Chạy full sweep: 4 corruption × các mức severity × 12 volume
- [ ] Tổng hợp kết quả, tính degradation rate, so sánh organ nhỏ vs organ lớn
- [ ] Xuất line chart + vài hình minh họa breakdown
