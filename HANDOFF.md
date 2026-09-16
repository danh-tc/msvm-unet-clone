# Handoff — Robustness Eval (MSVM-UNet)

Ngày: 2026-09-16. Đọc file này đầu tiên khi setup máy mới để tiếp tục đúng
mạch, không mất thời gian đọc lại toàn bộ hội thoại.

## TL;DR — việc cần làm tiếp theo ngay

Full sweep chưa chạy. Lệnh cần chạy đầu tiên (4 job song song theo corruption,
máy có 128 core nên không lo tranh chấp CPU):

```bash
cd msvm-unet-clone && source venv/bin/activate
python infer_robustness.py --all --corruptions gaussian_noise --output-json results/robustness_gaussian_noise.json > /tmp/sweep_gaussian_noise.log 2>&1 &
python infer_robustness.py --all --corruptions poisson_noise   --output-json results/robustness_poisson_noise.json   > /tmp/sweep_poisson_noise.log 2>&1 &
python infer_robustness.py --all --corruptions gaussian_blur    --output-json results/robustness_gaussian_blur.json  > /tmp/sweep_gaussian_blur.log 2>&1 &
python infer_robustness.py --all --corruptions contrast_shift   --output-json results/robustness_contrast_shift.json > /tmp/sweep_contrast_shift.log 2>&1 &
wait
```

Ước tính: 35-45 phút (best case) đến 1.5-2.5 tiếng (nếu gặp nhiều case HD95
chậm — xem "Rủi ro đã biết" bên dưới). Sau khi xong, việc còn lại: tổng hợp 4
file JSON, tính degradation rate, vẽ chart (checklist 2 mục cuối trong
`ROBUSTNESS_EVAL_PLAN.md`).

## Yêu cầu hạ tầng máy mới — bắt buộc phải có GPU

**Không thể chạy CPU-only.** Đã check code: MSVM-UNet dùng Mamba block gọi
thẳng compiled CUDA kernel (`model/msvm_unet/vmamba/csms6s.py:113-132`, import
`selective_scan_cuda_oflex`/`_core`/`_cuda`) — không có fallback thuần
PyTorch chạy CPU. Thuê máy phải có GPU + CUDA driver, và phải build lại kernel
này qua `install.sh` (bước `CC=gcc-11 pip install -e kernels/selective_scan`,
cần GCC 11).

Máy hiện tại dùng để tham khảo cấu hình: A100 80GB, 128 vCPU, torch 2.1.0+cu121,
Python 3.8. Bottleneck thực tế nằm ở CPU (scipy resize + medpy HD95 — 2 việc
này không dùng GPU), không phải GPU — GPU utilization đo được chỉ ~3%.

## Trạng thái git — quan trọng

Toàn bộ file mới/sửa **hiện đang chưa commit**:
- Mới: `corruptions.py`, `infer_robustness.py`, `ROBUSTNESS_EVAL_PLAN.md`, `HANDOFF.md`
- Sửa: `infer_single.py` (thêm chế độ `--all`), `infer_single_sam.py` (thêm `--skip-medsam`, `--use-points`, tách columns MedSAM optional)

Remote đã có sẵn: `origin -> github.com/danh-tc/msvm-unet-clone`. **Cần commit
+ push trước khi rời máy hiện tại**, nếu không mọi thứ trong file này sẽ mất
theo môi trường. Máy mới chỉ cần `git clone`/`git pull` là có đủ code + 2 file
plan/handoff này.

Checkpoint (`log/msvm_unet-synapse-r0/...ckpt`) và dataset
(`data/Synapse/...`) không nằm trong git (quá nặng) — máy mới cần tải lại qua
`install.sh` (bước 11, 12 — dataset + MSVM checkpoint qua gdrive). SAM-Med2D
checkpoint (`install.sh` bước 13) **không cần** cho hướng robustness này.

## Bối cảnh — vì sao đang ở đây

1. Hướng đầu: SAM-Med2D/MedSAM refine hậu kỳ mask MSVM-UNet — **không hiệu
   quả**, mọi biến thể prompt (box/box+mask/box+points) đều bằng hoặc thấp
   hơn baseline. Đã bỏ hướng này (script `infer_single_sam.py` vẫn giữ lại để
   tham khảo, không update thêm).
2. Pivot sang robustness eval (zero-shot, không train lại) — xem đầy đủ lý do
   và quyết định thiết kế trong `ROBUSTNESS_EVAL_PLAN.md`.

## Số liệu đã xác lập (dùng làm tham chiếu, không cần đo lại)

- Baseline sạch, 12/12 test case: **mean DSC = 83.96%, HD95 = 14.74mm**
  (`results/infer_single_results.json`). GB (66.57%) và PC (71.53%) yếu nhất,
  std cao nhất.
- Baseline sạch, riêng case0001: DSC 81.32%, HD95 20.14mm.
- Tier-2 smoke test (case0001, severity 1 & 4): gaussian_noise/poisson_noise/
  gaussian_blur giảm nhẹ-vừa ở severity=4 (-2.3 đến -4.4pp). **contrast_shift
  là ngoại lệ** — không suy giảm đều: DSC 81.32/82.23/83.27/78.63/66.33%
  (severity 0-4), tức còn nhích nhẹ ở sev1-2 rồi "gãy" mạnh ở sev3-4. HD95 lại
  tăng rõ từ sev2 (29.99mm) dù DSC lúc đó vẫn "trông ổn" — HD95 là chỉ báo sớm
  nhạy hơn DSC cho corruption này.

## Rủi ro đã biết — chi phí HD95 không ổn định

`medpy.metric.binary.hd95` có thể chậm bất thường (~4-5 phút thay vì ~40s)
khi prediction bị suy biến/lởm chởm (quan sát ở contrast_shift severity 2,3
trên case0001) — do distance-transform tốn hơn trên mask nhiễu, không phải do
tranh CPU (đã loại trừ, máy có 128 core). Chưa biết pattern này có lặp lại ở
severity 2,3 của 3 corruption còn lại, hay ở 11 volume khác, hay không — đây
là lý do ước tính full sweep có khoảng dao động rộng (35 phút - 2.5 tiếng).
Nếu full sweep chạy xong mà có nhiều lượt chậm bất thường, cân nhắc: bỏ HD95
ở các severity/case rõ ràng suy biến, hoặc chấp nhận thời gian dài hơn để giữ
tín hiệu HD95 (đã chứng minh có giá trị — phát hiện sớm hơn DSC).

## File liên quan

- `ROBUSTNESS_EVAL_PLAN.md` — plan đầy đủ, checklist tiến độ, quyết định thiết kế, rationale. **Đọc file này để hiểu chi tiết, file HANDOFF chỉ tóm tắt để resume nhanh.**
- `corruptions.py` — 4 hàm corruption (gaussian_noise, poisson_noise, gaussian_blur, contrast_shift) + `corrupt_volume()`.
- `infer_robustness.py` — script sweep chính, CLI: xem `--help`.
- `infer_single.py` — baseline gốc, đã thêm `--all`.
- `results/infer_single_results.json` — baseline 12-case đầy đủ.
- `results/infer_robustness_results.json` — kết quả tier-1/tier-2 smoke test (case0001 only, chưa phải full sweep).
