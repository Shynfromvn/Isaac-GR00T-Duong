# Serving GR00T với Server-Client

Ngày cập nhật: 2026-06-05

## Checkpoints

| Checkpoint | Steps | Cameras |
|------------|-------|---------|
| `checkpoints/checkpoint-50000` | 50,000 | head_cam, left_wrist_cam |
| `checkpoints/checkpoint-100000` | 100,000 | head_cam, left_wrist_cam |
| `checkpoints/checkpoint-150000` | 150,000 | head_cam, left_wrist_cam |
| `checkpoints/checkpoint-200000` | 200,000 | head_cam, left_wrist_cam |

## Dataset

```text
demo_data/pick_and_put_v4_converted
```

Dataset và checkpoint hiện dùng cùng camera keys:

```text
video: head_cam, left_wrist_cam
state: left_arm, left_hand
action: left_arm, left_hand
language: annotation.human.task_description
```

---

# Open-loop trực tiếp

Mẫu dưới đây dùng `checkpoint-50000`. Muốn chạy checkpoint khác thì chỉ cần thay `CHECKPOINT_NAME` thành `checkpoint-100000`, `checkpoint-150000`, hoặc `checkpoint-200000`.

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong


export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1


CHECKPOINT_NAME=checkpoint-200000
ACTION_HORIZON=8


.venv/bin/python gr00t/eval/open_loop_eval.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/pick_and_put_v4_converted \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/checkpoints/${CHECKPOINT_NAME} \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 100 \
  --action-horizon ${ACTION_HORIZON} \
  --steps 180 \
  --smooth-actions \
  --one-euro-freq 30 \
  --one-euro-min-cutoff 1 \
  --one-euro-beta 0.7 \
  --one-euro-d-cutoff 1.0 \
  --action-delta-clip 1.2 \
  --save-plot-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/open_loop_eval/${CHECKPOINT_NAME}_${ACTION_HORIZON}.jpeg

```

## Smoke test đã chạy

Đã chạy thử bản ngắn để kiểm tra pipeline:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

.venv/bin/python gr00t/eval/open_loop_eval.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/pick_and_put_v4_converted \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/checkpoints/checkpoint-50000 \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 \
  --action-horizon 16 \
  --steps 16 \
  --save-plot-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/open_loop_eval/checkpoint-50000/traj_0_smoke.jpeg
```

Kết quả:

```text
Dataset length: 261
Running trajectory: 0
Using 16 steps (requested: 16, trajectory length: 910)
Unnormalized Action MSE: 0.0008547097095288336
Unnormalized Action MAE: 0.02072237804532051
Output: my-outputs/open_loop_eval/checkpoint-50000/traj_0_smoke.jpeg
```

---

# Server-client nếu cần

Dùng cách này khi muốn load model một lần rồi chạy nhiều client/eval.

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

CHECKPOINT_NAME=checkpoint-50000
PORT=5555

.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/checkpoints/${CHECKPOINT_NAME} \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port ${PORT}
```

Client open-loop:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

CHECKPOINT_NAME=checkpoint-50000
PORT=5555

.venv/bin/python gr00t/eval/open_loop_eval.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/pick_and_put_v4_converted \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port ${PORT} \
  --traj-ids 0 \
  --action-horizon 16 \
  --steps 180 \
  --smooth-actions \
  --one-euro-freq 30 \
  --one-euro-min-cutoff 0.5 \
  --one-euro-beta 0.2 \
  --one-euro-d-cutoff 1.0 \
  --action-delta-clip 0.2 \
  --save-plot-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/open_loop_eval/${CHECKPOINT_NAME}/traj_0_server.jpeg
```

---

# Closed-loop MuJoCo / video 4 panels

Mẫu dưới đây dùng `checkpoint-50000`. Muốn chạy checkpoint khác thì chỉ cần thay `CHECKPOINT_NAME`.

Video output có layout 2x2:

```text
Dataset Head Cam  | Dataset Wrist Cam
MuJoCo GroundTruth| MuJoCo Predicted
```

## Chạy trực tiếp

Lệnh dưới đây bật One Euro smoothing cho predicted action và lưu sang folder riêng
`mujoco_four_panel_eval_smoth` để không lẫn với output cũ.

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

CHECKPOINT_NAME=checkpoint-200000
ACTION_HORIZON=8

.venv/bin/python gr00t/eval/mujoco_four_panel_replay.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/pick_and_put_v4_converted \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/checkpoints/${CHECKPOINT_NAME} \
  --embodiment-tag NEW_EMBODIMENT \
  --mujoco-model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml \
  --output-dir /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/mujoco_four_panel_eval_smoth/${CHECKPOINT_NAME}_${ACTION_HORIZON} \
  --traj-ids 5 25 125 \
  --max-steps 360 \
  --action-horizon ${ACTION_HORIZON} \
  --smooth-actions \
  --one-euro-freq 30 \
  --one-euro-min-cutoff 1 \
  --one-euro-beta 0.7 \
  --one-euro-d-cutoff 1.0 \
  --action-delta-clip 1.2 \
  --camera free \
  --width 640 \
  --height 480 \
  --fps 30
```

## Qua server-client

Start server giống mục **Server-client nếu cần** ở trên, rồi chạy client video:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

CHECKPOINT_NAME=checkpoint-50000
ACTION_HORIZON=8
PORT=5555

.venv/bin/python gr00t/eval/mujoco_four_panel_replay.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/pick_and_put_v4_converted \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port ${PORT} \
  --mujoco-model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml \
  --output-dir /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/mujoco_four_panel_eval_smoth/${CHECKPOINT_NAME}_${ACTION_HORIZON}ah_server \
  --start-traj-id 0 \
  --num-trajs 1 \
  --max-steps 180 \
  --action-horizon ${ACTION_HORIZON} \
  --smooth-actions \
  --one-euro-freq 30 \
  --one-euro-min-cutoff 1 \
  --one-euro-beta 0.7 \
  --one-euro-d-cutoff 1.0 \
  --action-delta-clip 1.2 \
  --camera free \
  --width 640 \
  --height 480 \
  --fps 30
```

Output mẫu:

```text
my-outputs/mujoco_four_panel_eval_smoth/${CHECKPOINT_NAME}_${ACTION_HORIZON}ah/
├── traj_0_four_panel.mp4
├── traj_0_four_panel.csv
└── summary.json
```

---

# Troubleshooting

Kiểm tra server:

```bash
ps -ef | grep -E "run_gr00t_server|checkpoint|PolicyServer" | grep -v grep
ss -ltnp 2>/dev/null | grep -E ":5555|:5556|:5557|:5558|python"
```

Nếu CUDA OOM:

```bash
nvidia-smi
# Tắt process GPU khác hoặc đổi device trong lệnh server:
--device cuda:1
```
