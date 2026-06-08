# Hybrid G1 Arm Inference Commands

Các lệnh dưới đây dùng cho setup tạm thời:

- Dùng video và Dex3 state từ dataset Kaggle/LeRobot.
- Dùng 14 khớp tay thật của G1 làm `observation.state[:14]`.
- Policy ACT vẫn xuất `100 x 28`, nhưng robot chỉ nhận `100 x 14` cho hai tay.
- Chạy qua `rt/arm_sdk`, không dùng debug/development low-level mode.

## Chuẩn Bị Robot

Trước khi chạy real control:

1. Đưa robot về Regular motion-control mode bằng remote R3:
   - `L2 + B`: Damping mode.
   - `L2 + UP`: Locked Standing.
   - `R1 + X`: Regular mode.
2. Không dùng:
   - `R2 + A`: Running mode.
   - `L2 + R2`: Debug/development mode.
3. Giữ sẵn nút dừng khẩn cấp.
4. Đảm bảo interface Ethernet là `enp1s0`. Nếu khác, đổi `--network-interface`.

## 1. Dry-Run Không Gửi Lệnh Robot

Dùng để kiểm tra inference, DDS state, CSV log. Không có `--send-actions`.

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/home/jkl0909/code/Son/unitree_lerobot/unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model \
  --episode=0 \
  --max-policy-steps=120 \
  --actions-per-inference=75 \
  --prefetch-threshold=0.5 \
  --frequency=30 \
  --max-action-delta-rad=0.02 \
  --network-interface=enp1s0
```

## 2. Test An Toàn 120 Step Trên Robot Thật

Đây là lệnh nên chạy đầu tiên sau khi robot đã vào Regular mode.

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/home/jkl0909/code/Son/unitree_lerobot/unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model \
  --episode=0 \
  --max-policy-steps=120 \
  --actions-per-inference=75 \
  --prefetch-threshold=0.5 \
  --frequency=30 \
  --initialization-speed-rad-s=0.05 \
  --initialization-max-tracking-error-rad=0.05 \
  --initialization-timeout-s=120 \
  --max-action-delta-rad=0.02 \
  --network-interface=enp1s0 \
  --motion \
  --initialize-from-dataset \
  --send-actions \
  --control-confirmation=SEND_TO_REAL_G1
```

Sau khi hiện:

```text
Dataset initial arm pose reached. Policy has not started yet.
Enter 's' to start policy control, or anything else to stop while holding pose:
```

Chỉ nhập `s` nếu tư thế tay ổn định và an toàn.

## 3. Chạy 300 Step

Dùng sau khi 120 step ổn.

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/home/jkl0909/code/Son/unitree_lerobot/unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model \
  --episode=0 \
  --max-policy-steps=300 \
  --actions-per-inference=75 \
  --prefetch-threshold=0.5 \
  --frequency=30 \
  --initialization-speed-rad-s=0.05 \
  --initialization-max-tracking-error-rad=0.05 \
  --initialization-timeout-s=120 \
  --max-action-delta-rad=0.02 \
  --network-interface=enp1s0 \
  --motion \
  --initialize-from-dataset \
  --send-actions \
  --control-confirmation=SEND_TO_REAL_G1
```

## 4. Chạy Hết Episode 0

Episode 0 dài 620 frame, khoảng 20.67 giây ở tốc độ dataset 30 Hz.

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/home/jkl0909/code/Son/unitree_lerobot/unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model \
  --episode=0 \
  --max-policy-steps=620 \
  --actions-per-inference=75 \
  --prefetch-threshold=0.5 \
  --frequency=30 \
  --initialization-speed-rad-s=0.05 \
  --initialization-max-tracking-error-rad=0.05 \
  --initialization-timeout-s=120 \
  --max-action-delta-rad=0.02 \
  --network-interface=enp1s0 \
  --motion \
  --initialize-from-dataset \
  --send-actions \
  --control-confirmation=SEND_TO_REAL_G1
```

Cũng có thể bỏ hẳn `--max-policy-steps=620`; chương trình sẽ chạy tới cuối episode.

## 5. Chạy Episode Khác

Dataset có 418 episode, ID hợp lệ `0..417`.

Ví dụ episode 1:

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/home/jkl0909/code/Son/unitree_lerobot/unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model \
  --episode=1 \
  --max-policy-steps=120 \
  --actions-per-inference=75 \
  --prefetch-threshold=0.5 \
  --frequency=30 \
  --initialization-speed-rad-s=0.05 \
  --initialization-max-tracking-error-rad=0.05 \
  --initialization-timeout-s=120 \
  --max-action-delta-rad=0.02 \
  --network-interface=enp1s0 \
  --motion \
  --initialize-from-dataset \
  --send-actions \
  --control-confirmation=SEND_TO_REAL_G1
```

Không nên chạy nối 418 episode liên tục trên robot thật. Mỗi episode nên khởi tạo lại tư thế đầu, chờ xác nhận, rồi mới chạy.

## 6. So Sánh Với Chế Độ Tuần Tự Cũ

Thêm `--synchronous-inference` để tắt prefetch nền. Chế độ này sẽ có delay gần 1 giây ở ranh giới chunk khi chạy CPU.

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/home/jkl0909/code/Son/unitree_lerobot/unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model \
  --episode=0 \
  --max-policy-steps=120 \
  --actions-per-inference=75 \
  --prefetch-threshold=0.5 \
  --frequency=30 \
  --initialization-speed-rad-s=0.05 \
  --initialization-max-tracking-error-rad=0.05 \
  --initialization-timeout-s=120 \
  --max-action-delta-rad=0.02 \
  --network-interface=enp1s0 \
  --motion \
  --initialize-from-dataset \
  --send-actions \
  --control-confirmation=SEND_TO_REAL_G1 \
  --synchronous-inference
```

## 7. Khi Nào Tăng/Đổi Tham Số

Giữ mặc định khuyến nghị:

```text
--actions-per-inference=75
--prefetch-threshold=0.5
--frequency=30
--max-action-delta-rad=0.02
```

Nếu terminal xuất hiện `wait=...ms` đáng kể ở các chunk sau, tăng:

```text
--prefetch-threshold=0.6
```

Không nên tăng `--frequency` quá 30 vì dataset/policy train ở 30 Hz.

`--max-action-delta-rad=0.02` tương đương khoảng 0.6 rad/s ở 30 Hz. Chưa nên tăng cao hơn nếu chưa xem log và quan sát chuyển động an toàn.

## 8. Cách Đọc Terminal

Các dòng tốt:

```text
G1 MotionSwitcher: code=0, mode={'form': '0', 'name': 'ai'}
G1 arm SDK preflight: the robot must be in Regular motion-control mode ...
Preparing initial ACT action chunk...
inference anchor=0 time=...ms usable_actions=75 queue=75 wait=...ms
step=30 dataset_index=30 queue=... prefetch=True dry_run=False
```

`dry_run=False` nghĩa là đang gửi lệnh thật vì có `--send-actions`.

Nếu sau lần đầu các dòng inference vẫn có `wait` lớn, prefetch chưa kịp. Tăng `--prefetch-threshold` hoặc giảm tải CPU.

## 9. Xem Video Dataset Để Đối Chiếu

Góc nhìn phù hợp nhất là `cam_left_high` hoặc `cam_right_high`.

```bash
vlc ~/.cache/huggingface/lerobot/unitreerobotics/G1_Dex3_ToastedBread_Dataset/videos/observation.images.cam_left_high/chunk-000/file-000.mp4
```

Episode 0 là 620 frame đầu, khoảng 20.67 giây đầu video.
