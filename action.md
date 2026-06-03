# Closed-loop MuJoCo evaluation changes

Ngay chay: 2026-06-02

## Cap nhat sau phan hoi cua user

Phan implementation `closed_loop_mujoco_eval.py` ben duoi la **khong dung voi yeu cau that** va da bi go bo. Ly do: no dung frame render tu MuJoCo lam input cho policy, trong khi yeu cau dung la:

- input policy/video ben trai phai lay tu dataset `head_cam`;
- MuJoCo chi dung de replay/visualize G1;
- video output can co 3 cot: `head_cam dataset` | `ground-truth G1 replay` | `predicted G1 replay`.

Khong su dung noi dung implementation cu ben duoi lam workflow chinh.

## Asset G1 da tai them

Da tai asset G1 tu hai nguon chinh thuc/cong khai:

1. Unitree USD 29DoF tu Hugging Face dataset `unitreerobotics/unitree_model`

```text
Nguon: https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main/G1/29dof/usd/g1_29dof_rev_1_0
Local:
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_base.usd
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_physics.usd
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_sensor.usd
```

2. Unitree MuJoCo G1 29DoF tu GitHub repo `unitreerobotics/unitree_mujoco`

```text
Nguon: https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/g1
Local:
assets/unitree_mujoco/unitree_robots/g1/g1_29dof.xml
assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
assets/unitree_mujoco/unitree_robots/g1/meshes/
```

Da test load MuJoCo:

```text
assets/unitree_mujoco/unitree_robots/g1/g1_29dof.xml OK nq 36 nv 35 nu 29 njnt 30
assets/unitree_mujoco/unitree_robots/g1/g1_23dof.xml OK nq 36 nv 35 nu 29 njnt 30
```

Ghi chu: USD dung cho Isaac Sim/Isaac Lab; MuJoCo workflow nen dung XML/MJCF `g1_29dof.xml` kem `meshes/`.

3. LeRobot Unitree G1 MuJoCo body29 + hand14 tu Hugging Face repo `lerobot/unitree-g1-mujoco`

```text
Nguon: https://huggingface.co/lerobot/unitree-g1-mujoco/tree/main/assets
Local:
assets/lerobot_unitree_g1_mujoco/assets/g1_29dof_with_hand.xml
assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml
assets/lerobot_unitree_g1_mujoco/assets/g1_body29_hand14.urdf
assets/lerobot_unitree_g1_mujoco/assets/meshes/
assets/lerobot_unitree_g1_mujoco/assets/meshes_exo_left/
assets/lerobot_unitree_g1_mujoco/assets/meshes_exo_right/
```

Da test load MuJoCo:

```text
assets/lerobot_unitree_g1_mujoco/assets/g1_29dof_with_hand.xml OK nq 50 nv 49 nu 43 njnt 44
```

Day la asset dung cho replay video 3 cot vi co du body29 + hand14, map duoc 28 joint action cua dataset `G1_Dex3_PickApple_Dataset_HeadcamOnly`.

## Script dung cho video 3 cot

### File: `gr00t/eval/mujoco_three_panel_replay.py`

- Trang thai cu: khong ton tai trong repo goc.
- Trang thai moi: them script tao video 3 cot dung yeu cau.
- Ly do: can dung `head_cam` tu dataset lam input policy, replay ground-truth action va predicted action tren hai scene G1 MuJoCo doc lap, roi ghep ngang thanh mot video danh gia.

Code moi chinh:

```python
DEFAULT_DATASET_JOINT_NAMES = [
    "kLeftShoulderPitch",
    ...
    "kRightHandMiddle1",
]

DATASET_TO_MJ_JOINT = {
    "kLeftShoulderPitch": "left_shoulder_pitch_joint",
    ...
    "kRightHandMiddle1": "right_hand_middle_1_joint",
}
```

```python
def _make_policy_observation(...):
    input_configs = deepcopy(modality_configs)
    input_configs.pop("action", None)
    step_data = extract_step_data(
        traj,
        step,
        input_configs,
        embodiment_tag,
        allow_padding=True,
    )
    ...
    return obs
```

```python
for step in range(steps):
    if not pred_cache:
        obs = _make_policy_observation(...)
        action_chunk, _ = policy.get_action(obs)
        pred_cache = [
            _concat_action(action_chunk, list(action_keys), action_step)
            for action_step in range(horizon)
        ]

    gt_action = _extract_vector(traj, step, list(action_keys), "action")
    pred_action = pred_cache.pop(0)
    _set_pose(model, gt_data, gt_action, qpos_map)
    _set_pose(model, pred_data, pred_action, qpos_map)

    head_frame = _resize_rgb(traj["video.head_cam"].iloc[step], (args.width, args.height))
    gt_frame = _render(renderer, gt_data, args)
    pred_frame = _render(renderer, pred_data, args)

    panels = [
        _caption(head_frame, "DATASET HEAD_CAM", subtitle),
        _caption(gt_frame, "MUJOCO G1 GROUND TRUTH", "dataset action replay"),
        _caption(pred_frame, "MUJOCO G1 PREDICTED", f"L2={l2_error:.3f} MAE={mae_error:.3f}"),
    ]
    recorder.write_frame(np.concatenate(panels, axis=1))
```

Ket qua layout:

```text
left   = DATASET HEAD_CAM
middle = MUJOCO G1 GROUND TRUTH
right  = MUJOCO G1 PREDICTED
```

Lenh da chay:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

.venv/bin/python gr00t/eval/mujoco_three_panel_replay.py \
  --traj-ids 0 1 \
  --max-steps 180 \
  --action-horizon 8 \
  --output-dir my-outputs/mujoco_three_panel_eval \
  --device cuda:0
```

Output da tao:

```text
my-outputs/mujoco_three_panel_eval/traj_0_three_panel.mp4
my-outputs/mujoco_three_panel_eval/traj_1_three_panel.mp4
my-outputs/mujoco_three_panel_eval/traj_0_three_panel.csv
my-outputs/mujoco_three_panel_eval/traj_1_three_panel.csv
my-outputs/mujoco_three_panel_eval/summary.json
```

Validation video:

```text
traj_0_three_panel.mp4 frame shape: (480, 1920, 3)
traj_1_three_panel.mp4 frame shape: (480, 1920, 3)
```

Metrics:

```json
{
  "traj_0": {
    "steps": 180,
    "mean_action_l2_error": 0.2408754693137275,
    "mean_action_mae_error": 0.02904729302972555,
    "mean_inference_ms": 1639.198930391222
  },
  "traj_1": {
    "steps": 180,
    "mean_action_l2_error": 0.23114419144888718,
    "mean_action_mae_error": 0.026526921056210996,
    "mean_inference_ms": 1555.7815922171437
  }
}
```

## Muc tieu

Them workflow closed-loop de dung checkpoint GR00T hien tai voi mot MuJoCo visual simulator, luu khoang 2 video va cac metrics can thiet de xem/danh gia.

## File da sua/them

### 1. `gr00t/eval/closed_loop_mujoco_eval.py`

- Trang thai cu: file khong ton tai trong repo goc.
- Trang thai moi: them script closed-loop MuJoCo evaluator.
- Ly do: repo co `open_loop_eval.py` va cac benchmark sim co san, nhung khong co MuJoCo env rieng cho dataset/checkpoint `NEW_EMBODIMENT` `G1_Dex3_PickApple_Dataset_HeadcamOnly`. Script moi tao MuJoCo kinematic robot 28D theo `meta/modality.json`, render camera `head_cam`, goi `Gr00tPolicy` trong vong lap closed-loop, ghi mp4/csv/json.

Code cu:

```text
Khong co file `gr00t/eval/closed_loop_mujoco_eval.py`.
```

Code moi chinh:

```python
@dataclass
class ArgsConfig:
    dataset_path: str = "demo_data/G1_Dex3_PickApple_Dataset_HeadcamOnly"
    model_path: str = "my-outputs/checkpoint-100000"
    embodiment_tag: str = "NEW_EMBODIMENT"
    output_dir: str = "my-outputs/closed_loop_mujoco_eval"
    num_videos: int = 2
    max_steps: int = 160
    n_action_steps: int = 8
    fps: int = 20
    width: int = 640
    height: int = 480
    device: str = "cuda:0"
    task: str | None = None
    action_mode: str = "absolute"
    simulation_mode: str = "kinematic"
    initial_state: str = "mean"
    reset_noise_std: float = 0.01
    success_distance: float = 0.12
    render_camera: str = "head_cam"
    seed: int = 0
```

```python
def _load_modality_slices(dataset_path: Path, section: str) -> list[ModalitySlice]:
    modality = _load_json(dataset_path / "meta" / "modality.json")
    slices = []
    for name, spec in modality[section].items():
        slices.append(ModalitySlice(name=name, start=int(spec["start"]), end=int(spec["end"])))
    slices.sort(key=lambda x: x.start)
    return slices
```

```python
def _load_action_bounds(dataset_path: Path, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        low = np.full(dim, -3.14, dtype=np.float32)
        high = np.full(dim, 3.14, dtype=np.float32)
        initial = np.zeros(dim, dtype=np.float32)
        return low, high, initial

    stats = _load_json(stats_path)
    low = _stats_vector(stats, "action", "q01", dim)
    high = _stats_vector(stats, "action", "q99", dim)
    if low is None or high is None:
        low = _stats_vector(stats, "action", "min", dim)
        high = _stats_vector(stats, "action", "max", dim)
    mean = _stats_vector(stats, "observation.state", "mean", dim)
    if mean is None:
        mean = _stats_vector(stats, "action", "mean", dim)

    if low is None or high is None:
        low = np.full(dim, -3.14, dtype=np.float32)
        high = np.full(dim, 3.14, dtype=np.float32)
    if mean is None:
        mean = np.zeros(dim, dtype=np.float32)

    pad = np.maximum((high - low) * 0.05, 1e-3)
    return (low - pad).astype(np.float32), (high + pad).astype(np.float32), mean.astype(np.float32)
```

```python
def _build_mjcf(slices: list[ModalitySlice], low: np.ndarray, high: np.ndarray) -> str:
    ...
    return f"""
<mujoco model="gr00t_closed_loop_kinematic">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  ...
  <worldbody>
    <camera name="head_cam" pos="1.25 -1.75 1.15" xyaxes="0.82 0.58 0 -0.23 0.33 0.92"/>
    <geom name="plate" type="cylinder" pos="0.72 0 0.025" size="0.18 0.025" rgba="0.9 0.9 1 1"/>
    <body name="apple" pos="0.45 0 0.09">
      <geom name="apple_geom" type="sphere" size="0.065" rgba="0.95 0.05 0.03 1"/>
    </body>
    {left_chain}
    {right_chain}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""
```

```python
def _make_observation(
    frame_history: list[np.ndarray],
    state_history: list[np.ndarray],
    task: str,
    modality_config: dict[str, Any],
    state_slices: list[ModalitySlice],
) -> dict[str, Any]:
    video_delta = list(modality_config["video"].delta_indices)
    state_delta = list(modality_config["state"].delta_indices)
    language_delta = list(modality_config["language"].delta_indices)

    video_frames = np.stack(_history_sample(frame_history, video_delta), axis=0)
    state_frames = _history_sample(state_history, state_delta)
    ...
    return {"video": video, "state": state, "language": language}
```

```python
while step < args.max_steps:
    obs = _make_observation(
        frame_history,
        state_history,
        task,
        modality_config,
        state_slices,
    )
    start = time.perf_counter()
    action_chunk, _ = policy.get_action(obs)
    infer_ms = (time.perf_counter() - start) * 1000.0
    horizon = min(args.n_action_steps, next(iter(action_chunk.values())).shape[1])

    for action_step in range(horizon):
        decoded = _concat_action(action_chunk, action_slices, action_step)
        target = data.qpos.astype(np.float32) + decoded if args.action_mode == "delta" else decoded
        clipped = np.clip(target, low, high)
        if args.simulation_mode == "kinematic":
            dt = float(model.opt.timestep)
            data.qvel[:] = (clipped - data.qpos.astype(np.float32)) / dt
            data.qpos[:] = clipped
            data.time += dt
            mujoco.mj_forward(model, data)
        else:
            data.ctrl[:] = clipped
            mujoco.mj_step(model, data)
```

```python
summary = {
    "model_path": args.model_path,
    "dataset_path": args.dataset_path,
    "embodiment_tag": embodiment_tag.name,
    "action_mode": args.action_mode,
    "simulation_mode": args.simulation_mode,
    "n_action_steps": args.n_action_steps,
    "num_videos": args.num_videos,
    "success_rate": float(np.mean([item["success"] for item in summaries])),
    "mean_inference_ms": float(np.mean([item["mean_inference_ms"] for item in summaries])),
    "episodes": summaries,
}
```

Dia chi code moi:

- `gr00t/eval/closed_loop_mujoco_eval.py:43` - CLI config.
- `gr00t/eval/closed_loop_mujoco_eval.py:118` - doc modality/action-state slice tu dataset.
- `gr00t/eval/closed_loop_mujoco_eval.py:141` - doc q01/q99/min/max/mean tu stats de clip action.
- `gr00t/eval/closed_loop_mujoco_eval.py:169` - tao MJCF robot 28 joint.
- `gr00t/eval/closed_loop_mujoco_eval.py:269` - tao observation nested dung format `Gr00tPolicy`.
- `gr00t/eval/closed_loop_mujoco_eval.py:345` - closed-loop rollout, ghi video/csv metrics.
- `gr00t/eval/closed_loop_mujoco_eval.py:433` - kinematic MuJoCo stepping mac dinh; giu `actuator` mode neu can debug physics.
- `gr00t/eval/closed_loop_mujoco_eval.py:516` - main/load policy/run episodes/write `summary.json`.

## Dependency moi

Khong sua `pyproject.toml` vi khi thu them `mujoco` vao dependencies, `uv run` phai sync/resolve lai va bi treo rat lau trong moi truong hien tai. Thay vao do da cai truc tiep vao `.venv`:

```bash
uv pip install mujoco
```

Ket qua cai:

```text
Installed 5 packages
+ etils==1.13.0
+ glfw==2.10.0
+ importlib-resources==7.1.0
+ mujoco==3.9.0
+ pyopengl==3.1.10
```

## Lenh da chay

Smoke test MuJoCo render:

```bash
wsl -e bash -lc 'cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong && .venv/bin/python -c "... build model/render one frame ..."'
```

Ket qua:

```text
28 28 (120, 160, 3) uint8 63
```

Smoke test policy + MuJoCo ngan:

```bash
.venv/bin/python gr00t/eval/closed_loop_mujoco_eval.py \
  --num-videos 1 \
  --max-steps 2 \
  --n-action-steps 1 \
  --output-dir my-outputs/closed_loop_mujoco_eval_smoke \
  --device cuda:0
```

Lenh chinh de tao 2 video:

```bash
cd /mnt/e/Vin/Groot/Isaac-GR00T-Duong

.venv/bin/python gr00t/eval/closed_loop_mujoco_eval.py \
  --num-videos 2 \
  --max-steps 160 \
  --n-action-steps 8 \
  --output-dir my-outputs/closed_loop_mujoco_eval \
  --device cuda:0 \
  --simulation-mode kinematic
```

## Output da tao

Thu muc:

```text
my-outputs/closed_loop_mujoco_eval/
```

Files:

```text
closed_loop_kinematic_robot.xml
closed_loop_mujoco_ep000.mp4
closed_loop_mujoco_ep000.csv
closed_loop_mujoco_ep001.mp4
closed_loop_mujoco_ep001.csv
summary.json
```

## Metrics lan chay 2 video

Tong quan:

```json
{
  "success_rate": 0.0,
  "mean_inference_ms": 1242.1644480499253,
  "n_action_steps": 8,
  "simulation_mode": "kinematic"
}
```

Episode 0:

```json
{
  "video_path": "my-outputs/closed_loop_mujoco_eval/closed_loop_mujoco_ep000.mp4",
  "success": false,
  "min_left_tip_apple_distance": 0.428988404558305,
  "min_right_tip_apple_distance": 0.6367699111483798,
  "final_left_tip_apple_distance": 0.5287278986606808,
  "final_right_tip_apple_distance": 0.7151404418846455,
  "mean_abs_action": 0.5497114082798362,
  "mean_abs_action_delta": 0.011873686677372118,
  "clip_fraction": 0.03571428571428571
}
```

Episode 1:

```json
{
  "video_path": "my-outputs/closed_loop_mujoco_eval/closed_loop_mujoco_ep001.mp4",
  "success": false,
  "min_left_tip_apple_distance": 0.6873564717032107,
  "min_right_tip_apple_distance": 0.6470399537971324,
  "final_left_tip_apple_distance": 0.9462272467955021,
  "final_right_tip_apple_distance": 0.6518081595349947,
  "mean_abs_action": 0.5573184235021472,
  "mean_abs_action_delta": 0.013474477671742814,
  "clip_fraction": 0.0
}
```

## Ghi chu danh gia

- Day la MuJoCo visual/kinematic closed-loop evaluator cho checkpoint custom, khong phai benchmark task vat ly chuan cua Unitree G1/Dex3. Repo hien tai khong co MuJoCo asset/env that cho `G1_Dex3_PickApple_Dataset_HeadcamOnly`.
- `success=false` ca 2 episode vi khoang cach nho nhat tu dau tay toi apple marker van lon hon nguong `0.12m`.
- `clip_fraction` episode 0 khoang `3.57%`, episode 1 `0%`; action khong bi clip qua nhieu trong kinematic run.
- Da thu actuator physics stepping truoc do, MuJoCo bao `QACC` instability. Vi robot MJCF nay la robot kinematic sinh tu action dimension, mac dinh da doi sang `--simulation-mode kinematic` de video/metrics on dinh. Neu co asset robot that va controller that, co the chuyen sang env physics rieng sau.

## Cleanup output cu

Da xoa cac output cu/smoke khong con dung:

- `my-outputs/closed_loop_mujoco_eval`
- `my-outputs/closed_loop_mujoco_eval_smoke`
- `my-outputs/mujoco_three_panel_eval_smoke`

Da giu lai output moi dung yeu cau 3 panel:

- `my-outputs/mujoco_three_panel_eval/traj_0_three_panel.mp4`
- `my-outputs/mujoco_three_panel_eval/traj_1_three_panel.mp4`
- `my-outputs/mujoco_three_panel_eval/summary.json`

## Them cach chon nhieu trajectory de tao video

### File sua

- `gr00t/eval/mujoco_three_panel_replay.py`
- `serving-server-client.md`

### Code cu

Truoc do script chi co danh sach trajectory cu the:

```python
traj_ids: list[int] = field(default_factory=lambda: [0, 1])
"""Dataset episode ids to render."""
```

Va vong lap render dung truc tiep danh sach nay:

```python
for traj_id in args.traj_ids:
    if traj_id >= len(loader):
        logging.warning("Skipping traj_id=%s because dataset length is %s", traj_id, len(loader))
        continue
```

Neu can tao nhieu video, phai go dai:

```bash
--traj-ids 0 1 2 3 4 5 6 7 8 9
```

### Code moi

Da them cac tham so:

```python
traj_ids: list[int] = field(default_factory=list)
"""Explicit dataset episode ids to render. If empty, uses start_traj_id/num_trajs."""

start_traj_id: int = 0
"""First dataset episode id to render when traj_ids is empty."""

num_trajs: int = 2
"""Number of consecutive trajectories to render when traj_ids is empty."""

all_trajs: bool = False
"""Render every dataset trajectory from start_traj_id onward."""
```

Da them ham resolve trajectory:

```python
def _resolve_traj_ids(args: ArgsConfig, dataset_len: int) -> list[int]:
    if args.traj_ids:
        requested = list(args.traj_ids)
    elif args.all_trajs:
        requested = list(range(args.start_traj_id, dataset_len))
    else:
        if args.num_trajs < 1:
            raise ValueError("--num-trajs must be at least 1.")
        requested = list(range(args.start_traj_id, args.start_traj_id + args.num_trajs))
```

Vong lap render bay gio dung danh sach da resolve va tu skip id sai/duplicate:

```python
traj_ids = _resolve_traj_ids(args, len(loader))

summaries = []
for traj_id in traj_ids:
    logging.info("Rendering three-panel MuJoCo replay for traj_id=%s", traj_id)
```

### Ly do sua

- `--traj-ids` khong tien khi can tao so luong video lon vi phai liet ke tung id.
- Them `--start-traj-id` va `--num-trajs` de tao N video lien tiep, vi du `--start-traj-id 0 --num-trajs 20`.
- Them `--all-trajs` de render toan bo dataset khi can.
- Giu `--traj-ids` cho truong hop can chon episode roi rac, vi du `--traj-ids 0 5 12`.
- Cap nhat `serving-server-client.md` de ghi ro cach chay va y nghia cac tham so moi.

### Kiem tra

Da chay compile check:

```bash
python -m py_compile gr00t/eval/mujoco_three_panel_replay.py
```
