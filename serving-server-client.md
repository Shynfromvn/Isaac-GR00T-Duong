# Serving GR00T checkpoint-100000 with server-client

Ngay cap nhat: 2026-06-02

## Trang thai hien tai

Da kiem tra trong WSL bang:

```bash
ps -ef | grep -E "run_gr00t_server|checkpoint-100000|PolicyServer|5555" | grep -v grep
ss -ltnp 2>/dev/null | grep -E ":5555|run_gr00t_server|python"
```

Ket qua: **checkpoint `my-outputs/checkpoint-100000` chua duoc serving**.

- Khong thay process `gr00t/eval/run_gr00t_server.py`.
- Khong thay port ZMQ mac dinh `5555` dang listen.

## Terminal 1: start policy server

Chay trong WSL:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/checkpoint-100000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Server load dung khi thay log gan nhu:

```text
Starting GR00T inference server...
  Embodiment tag: EmbodimentTag.NEW_EMBODIMENT
  Model path: /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/checkpoint-100000
  Device: cuda:0
  Host: 0.0.0.0
  Port: 5555

Server ready - listening on 0.0.0.0:5555
```

Giu terminal nay mo. Dung server bang `Ctrl+C`.

## Neu muon dung `uv run`

Co the dung:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/checkpoint-100000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Nhung trong repo nay `uv run` thinh thoang re-check/reinstall wheel, nen `.venv/bin/python` nhanh va on dinh hon neu environment da sync xong.

## Terminal 2: check server da listen

```bash
ss -ltnp | grep 5555
```

Neu server dang chay, se thay mot dong co `:5555` va process `python`.

Kiem tra process:

```bash
ps -ef | grep run_gr00t_server | grep -v grep
```

## Terminal 2: test client ket noi server

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

.venv/bin/python -c "from gr00t.policy.server_client import PolicyClient; p=PolicyClient(host='127.0.0.1', port=5555); print(p.get_modality_config())"
```

Neu thanh cong, client se in modality config cua checkpoint, gom cac key:

```text
video: head_cam
state: left_arm, right_arm, left_hand, right_hand
action: left_arm, right_arm, left_hand, right_hand
language: annotation.human.task_description
```

Neu loi:

```text
zmq.error.Again: Resource temporarily unavailable
```

thi client khong nhan duoc response. Thuong la do server chua chay, server dang load model chua xong, sai port, hoac server crash.

## Chay open-loop qua server-client

Khi server da chay o Terminal 1, Terminal 2 chay:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

.venv/bin/python gr00t/eval/open_loop_eval.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/G1_Dex3_PickApple_Dataset_HeadcamOnly \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port 5555 \
  --traj-ids 0 \
  --action-horizon 16 \
  --save-plot-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/open_loop_eval/traj_0_server_client.jpeg
```

Quan trong: khi chay qua server-client thi **khong truyen `--model-path` vao client**. Model da duoc load trong server.

## Chay lai de tao video 3 panel MuJoCo

Script da dung de tao video dung yeu cau la:

```text
gr00t/eval/mujoco_three_panel_replay.py
```

Moi video output gom 3 panel:

```text
left  = DATASET HEAD_CAM
mid   = MUJOCO G1 GROUND TRUTH, replay action that tu dataset
right = MUJOCO G1 PREDICTED, replay action du doan tu checkpoint/server
```

Output mac dinh duoc ghi vao:

```text
my-outputs/mujoco_three_panel_eval/
```

Voi `--start-traj-id 0 --num-trajs 2`, script se tao 2 video:

```text
my-outputs/mujoco_three_panel_eval/traj_0_three_panel.mp4
my-outputs/mujoco_three_panel_eval/traj_1_three_panel.mp4
my-outputs/mujoco_three_panel_eval/summary.json
my-outputs/mujoco_three_panel_eval/traj_0_three_panel.csv
my-outputs/mujoco_three_panel_eval/traj_1_three_panel.csv
```

### Cach A: chay truc tiep bang checkpoint local

Dung cach nay neu chi chay nhanh 1 lan, khong muon bat server rieng.

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

.venv/bin/python gr00t/eval/mujoco_three_panel_replay.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/G1_Dex3_PickApple_Dataset_HeadcamOnly \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/checkpoint-100000 \
  --embodiment-tag NEW_EMBODIMENT \
  --mujoco-model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml \
  --output-dir /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/mujoco_three_panel_eval \
  --start-traj-id 0 \
  --num-trajs 2 \
  --max-steps 180 \
  --action-horizon 8 \
  --camera free \
  --width 640 \
  --height 480 \
  --fps 30
```

Luu y: cach nay se load checkpoint moi lan chay script.

### Cach B: chay qua server-client

Dung cach nay neu can tao video nhieu lan hoac so sanh nhieu trajectory. Server load checkpoint 1 lan, client chi gui observation sang server va nhan action predict.

Terminal 1, start server:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/checkpoint-100000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Giu terminal server mo. Khi thay server da ready/listening, mo Terminal 2 de tao video:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

.venv/bin/python gr00t/eval/mujoco_three_panel_replay.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/G1_Dex3_PickApple_Dataset_HeadcamOnly \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port 5555 \
  --mujoco-model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml \
  --output-dir /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/mujoco_three_panel_eval \
  --start-traj-id 0 \
  --num-trajs 2 \
  --max-steps 180 \
  --action-horizon 8 \
  --camera free \
  --width 640 \
  --height 480 \
  --fps 30
```

Quan trong: khi dung server-client thi client **khong can `--model-path`**. Checkpoint da nam trong server process.

### Dieu khien so luong video

Moi trajectory tao ra 1 file video. Cach nen dung khi tao nhieu video la `--start-traj-id` va `--num-trajs`.

Tao 3 video lien tiep tu episode 0:

```bash
--start-traj-id 0 --num-trajs 3
```

Output:

```text
traj_0_three_panel.mp4
traj_1_three_panel.mp4
traj_2_three_panel.mp4
```

Tao 20 video lien tiep tu episode 0:

```bash
--start-traj-id 0 --num-trajs 20
```

Tao 10 video lien tiep bat dau tu episode 50:

```bash
--start-traj-id 50 --num-trajs 10
```

Tao tat ca video trong dataset tu episode 0 den het:

```bash
--start-traj-id 0 --all-trajs
```

Neu chi muon chon episode roi rac, van co the dung `--traj-ids`:

```bash
--traj-ids 0 5 12
```

Output khi do se co:

```text
traj_0_three_panel.mp4
traj_5_three_panel.mp4
traj_12_three_panel.mp4
```

### Cac tham so co the sua

`--dataset-path`

Duong dan dataset LeRobot dung lam input. Trong bai hien tai la:

```text
demo_data/G1_Dex3_PickApple_Dataset_HeadcamOnly
```

Dataset nay cung cap `video.head_cam`, state, action ground truth va text instruction.

`--model-path`

Duong dan checkpoint local, vi du:

```text
my-outputs/checkpoint-100000
```

Chi dung tham so nay khi chay truc tiep bang checkpoint local. Neu dung server-client thi bo `--model-path`.

`--embodiment-tag`

Tag embodiment checkpoint da train voi dataset. Hien tai dung:

```text
NEW_EMBODIMENT
```

Neu tag sai, model co the load sai modality/action mapping.

`--host` va `--port`

Dung khi client ket noi policy server. Vi du:

```text
--host 127.0.0.1 --port 5555
```

Neu doi port server sang `5556` thi client cung phai doi sang `--port 5556`.

`--device`

Device cho local inference khi khong dung server-client:

```text
cuda:0
cuda:1
cpu
```

`cpu` thuong rat cham voi model nay. Khi dung server-client, device nam o lenh server, khong nam o client.

`--mujoco-model-path`

File MuJoCo XML dung de replay G1:

```text
assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml
```

File nay co G1 body 29DoF va hand joints phu hop hon cho dataset Dex3 so voi USD truc tiep.

`--output-dir`

Thu muc ghi video, CSV va `summary.json`. Neu khong muon ghi de output cu, doi sang folder moi:

```text
my-outputs/mujoco_three_panel_eval_run2
```

`--start-traj-id`

Episode dau tien can render khi khong dung `--traj-ids`. Vi du:

```text
--start-traj-id 0
--start-traj-id 50
```

`--num-trajs`

So luong trajectory lien tiep can render khi khong dung `--traj-ids` va khong bat `--all-trajs`. Vi du:

```text
--num-trajs 2
--num-trajs 20
--num-trajs 100
```

Moi trajectory tao ra 1 video rieng.

`--all-trajs`

Render tat ca trajectory trong dataset, bat dau tu `--start-traj-id`. Vi du:

```text
--start-traj-id 0 --all-trajs
```

Dung can than voi dataset lon vi thoi gian chay va dung luong output se tang nhanh.

`--traj-ids`

Danh sach episode trong dataset can render thu cong. Tham so nay uu tien hon `--start-traj-id`, `--num-trajs`, va `--all-trajs`.

```text
--traj-ids 0 1
--traj-ids 0 1 2
--traj-ids 0 5 12
```

Dung `--traj-ids` khi can chon episode roi rac. Neu can tao nhieu video lien tiep, nen dung `--start-traj-id` va `--num-trajs`.

`--max-steps`

So frame toi da render moi trajectory. Vi du:

```text
--max-steps 180
```

Tang len thi video dai hon va chay lau hon. Giam xuong de smoke test nhanh.

`--action-horizon`

So action predict duoc replay truoc khi query policy lan tiep theo. Vi du:

```text
--action-horizon 8
```

Gia tri nho hon query model nhieu hon, cham hon nhung cap nhat prediction thuong xuyen hon. Gia tri lon hon query model it hon, nhanh hon nhung prediction co the drift lau hon truoc khi duoc cap nhat.

`--camera`

Camera render MuJoCo cho panel ground truth va predict:

```text
free
global_view
head_camera
```

`free` la camera mac dinh da can cho thay robot ro. `global_view` nhin toan canh hon. `head_camera` dung camera gan tren model neu asset co camera do.

`--camera-distance`, `--camera-azimuth`, `--camera-elevation`

Chi co tac dung ro khi `--camera free`.

Vi du:

```text
--camera-distance 2.0
--camera-azimuth 145
--camera-elevation -15
```

Neu robot qua nho, giam `camera-distance`. Neu goc nhin chua dung, sua `camera-azimuth` hoac `camera-elevation`.

`--width` va `--height`

Kich thuoc moi panel truoc khi ghep ngang. Mac dinh:

```text
--width 640 --height 480
```

Video cuoi cung se co chieu rong bang `3 * width`, vi co 3 panel. Voi mac dinh, video output la `1920x480`.

`--fps`

Frame rate video output:

```text
--fps 30
```

`--video-backend`

Backend doc video dataset. Mac dinh:

```text
torchcodec
```

Chi doi tham so nay neu backend hien tai bi loi doc video trong dataset.

## Doi port neu bi trung

Neu Terminal 1 bao port 5555 da bi dung, doi sang 5556:

Server:

```bash
.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/checkpoint-100000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5556
```

Client:

```bash
.venv/bin/python gr00t/eval/open_loop_eval.py \
  --dataset-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/demo_data/G1_Dex3_PickApple_Dataset_HeadcamOnly \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port 5556 \
  --traj-ids 0 \
  --action-horizon 16 \
  --save-plot-path /mnt/e/Vin/Groot/Isaac-GR00T-Duong/my-outputs/open_loop_eval/traj_0_server_client.jpeg
```

## Troubleshooting nhanh

### 1. Hugging Face gated repo

Neu server load model roi gap loi `nvidia/Cosmos-Reason2-2B` 401/403, check:

```bash
.venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('nvidia/Cosmos-Reason2-2B', 'config.json'))"
```

Neu loi 403 thi account/token chua co access model.

### 2. CUDA het VRAM

Neu server crash do CUDA OOM:

```bash
nvidia-smi
```

Tat process GPU khac hoac chay device khac:

```bash
--device cuda:1
```

### 3. Client timeout

Neu client timeout:

```bash
ss -ltnp | grep 5555
ps -ef | grep run_gr00t_server | grep -v grep
```

Neu khong co process/port thi start lai server.

### 4. Dung server

Trong Terminal 1 bam:

```text
Ctrl+C
```

Hoac tim PID:

```bash
ps -ef | grep run_gr00t_server | grep -v grep
```
