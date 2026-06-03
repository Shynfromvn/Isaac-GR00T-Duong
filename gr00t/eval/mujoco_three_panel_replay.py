# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import dataclass, field
import csv
import json
import logging
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import tyro

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecorder
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient


try:
    import mujoco
except ImportError as exc:
    raise ImportError(
        "MuJoCo is required. Install it with `uv pip install mujoco`, then run this script "
        "with `.venv/bin/python` or `uv run --no-sync python`."
    ) from exc


DEFAULT_DATASET_JOINT_NAMES = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
]

DATASET_TO_MJ_JOINT = {
    "kLeftShoulderPitch": "left_shoulder_pitch_joint",
    "kLeftShoulderRoll": "left_shoulder_roll_joint",
    "kLeftShoulderYaw": "left_shoulder_yaw_joint",
    "kLeftElbow": "left_elbow_joint",
    "kLeftWristRoll": "left_wrist_roll_joint",
    "kLeftWristPitch": "left_wrist_pitch_joint",
    "kLeftWristYaw": "left_wrist_yaw_joint",
    "kRightShoulderPitch": "right_shoulder_pitch_joint",
    "kRightShoulderRoll": "right_shoulder_roll_joint",
    "kRightShoulderYaw": "right_shoulder_yaw_joint",
    "kRightElbow": "right_elbow_joint",
    "kRightWristRoll": "right_wrist_roll_joint",
    "kRightWristPitch": "right_wrist_pitch_joint",
    "kRightWristYaw": "right_wrist_yaw_joint",
    "kLeftHandThumb0": "left_hand_thumb_0_joint",
    "kLeftHandThumb1": "left_hand_thumb_1_joint",
    "kLeftHandThumb2": "left_hand_thumb_2_joint",
    "kLeftHandMiddle0": "left_hand_middle_0_joint",
    "kLeftHandMiddle1": "left_hand_middle_1_joint",
    "kLeftHandIndex0": "left_hand_index_0_joint",
    "kLeftHandIndex1": "left_hand_index_1_joint",
    "kRightHandThumb0": "right_hand_thumb_0_joint",
    "kRightHandThumb1": "right_hand_thumb_1_joint",
    "kRightHandThumb2": "right_hand_thumb_2_joint",
    "kRightHandIndex0": "right_hand_index_0_joint",
    "kRightHandIndex1": "right_hand_index_1_joint",
    "kRightHandMiddle0": "right_hand_middle_0_joint",
    "kRightHandMiddle1": "right_hand_middle_1_joint",
}


@dataclass
class ArgsConfig:
    dataset_path: str = "demo_data/G1_Dex3_PickApple_Dataset_HeadcamOnly"
    """Dataset that provides head_cam video, state, action, and task text."""

    model_path: str | None = "my-outputs/checkpoint-100000"
    """Local checkpoint. Leave empty only when using --host/--port policy server."""

    embodiment_tag: str = "NEW_EMBODIMENT"
    """Embodiment tag used by the checkpoint/server."""

    mujoco_model_path: str = "assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml"
    """MuJoCo XML with G1 body29+hand14 model."""

    output_dir: str = "my-outputs/mujoco_three_panel_eval"
    """Directory for mp4 videos, per-step CSV files, and summary.json."""

    traj_ids: list[int] = field(default_factory=list)
    """Explicit dataset episode ids to render. If empty, uses start_traj_id/num_trajs."""

    start_traj_id: int = 0
    """First dataset episode id to render when traj_ids is empty."""

    num_trajs: int = 2
    """Number of consecutive trajectories to render when traj_ids is empty."""

    all_trajs: bool = False
    """Render every dataset trajectory from start_traj_id onward."""

    max_steps: int = 180
    """Max frames/steps per trajectory."""

    action_horizon: int = 8
    """Number of predicted action steps to execute before querying policy again."""

    host: str = ""
    """Policy server host. If set with --port, uses server-client instead of local model."""

    port: int | None = None
    """Policy server port."""

    device: str = "cuda:0"
    """Torch device for local model inference."""

    camera: str = "free"
    """MuJoCo camera for GT/PRED replay render. Use free, global_view, or head_camera."""

    camera_distance: float = 2.0
    """Distance for the default free camera."""

    camera_azimuth: float = 145.0
    """Azimuth angle for the default free camera."""

    camera_elevation: float = -15.0
    """Elevation angle for the default free camera."""

    width: int = 640
    """Width of each panel before final concat."""

    height: int = 480
    """Height of each panel before final concat."""

    fps: int = 30
    """Output video FPS."""

    video_backend: str = "torchcodec"
    """Dataset video backend."""


def _load_joint_names(dataset_path: Path) -> list[str]:
    info_path = dataset_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    names = info.get("features", {}).get("action", {}).get("names")
    if names and names[0]:
        return list(names[0])
    return DEFAULT_DATASET_JOINT_NAMES


def _create_policy(args: ArgsConfig) -> BasePolicy:
    if args.host and args.port is not None:
        logging.info("Using PolicyClient at %s:%s", args.host, args.port)
        return PolicyClient(host=args.host, port=args.port)
    if not args.model_path:
        raise ValueError("Provide --model-path or both --host and --port.")
    logging.info("Loading local Gr00tPolicy from %s", args.model_path)
    return Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=args.model_path,
        device=args.device,
    )


def _resolve_traj_ids(args: ArgsConfig, dataset_len: int) -> list[int]:
    if args.traj_ids:
        requested = list(args.traj_ids)
    elif args.all_trajs:
        requested = list(range(args.start_traj_id, dataset_len))
    else:
        if args.num_trajs < 1:
            raise ValueError("--num-trajs must be at least 1.")
        requested = list(range(args.start_traj_id, args.start_traj_id + args.num_trajs))

    resolved = []
    seen = set()
    for traj_id in requested:
        if traj_id < 0 or traj_id >= dataset_len:
            logging.warning("Skipping traj_id=%s because dataset length is %s", traj_id, dataset_len)
            continue
        if traj_id in seen:
            logging.warning("Skipping duplicate traj_id=%s", traj_id)
            continue
        resolved.append(traj_id)
        seen.add(traj_id)

    if not resolved:
        raise ValueError(
            "No valid trajectories selected. Check --traj-ids, --start-traj-id, "
            "--num-trajs, or --all-trajs."
        )
    return resolved


def _make_policy_observation(
    traj: Any,
    step: int,
    loader: LeRobotEpisodeLoader,
    modality_configs: dict[str, Any],
    embodiment_tag: EmbodimentTag,
) -> dict[str, Any]:
    input_configs = deepcopy(modality_configs)
    input_configs.pop("action", None)
    step_data = extract_step_data(
        traj,
        step,
        input_configs,
        embodiment_tag,
        allow_padding=True,
    )

    obs: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
    for key, value in step_data.images.items():
        obs["video"][key] = np.asarray(value, dtype=np.uint8)[None, ...]
    for key, value in step_data.states.items():
        obs["state"][key] = np.asarray(value, dtype=np.float32)[None, ...]
    for language_key in loader.modality_configs["language"].modality_keys:
        obs["language"][language_key] = [[step_data.text]]
    return obs


def _concat_action(action: dict[str, np.ndarray], action_keys: list[str], step: int) -> np.ndarray:
    parts = []
    for key in action_keys:
        arr = np.asarray(action[key][0], dtype=np.float32)
        parts.append(np.atleast_1d(arr[min(step, arr.shape[0] - 1)]))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _extract_vector(traj: Any, step: int, keys: list[str], prefix: str) -> np.ndarray:
    parts = []
    safe_step = min(max(step, 0), len(traj) - 1)
    for key in keys:
        parts.append(np.atleast_1d(np.asarray(traj[f"{prefix}.{key}"].iloc[safe_step])))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _qpos_addresses(model: Any, dataset_joint_names: list[str]) -> dict[int, int]:
    mapping = {}
    missing = []
    for dataset_idx, dataset_name in enumerate(dataset_joint_names):
        mj_name = DATASET_TO_MJ_JOINT.get(dataset_name)
        if mj_name is None:
            missing.append(dataset_name)
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
        if joint_id < 0:
            missing.append(f"{dataset_name}->{mj_name}")
            continue
        mapping[dataset_idx] = int(model.jnt_qposadr[joint_id])
    if missing:
        raise ValueError(f"Missing MuJoCo joint mapping for: {missing}")
    return mapping


def _set_pose(model: Any, data: Any, action_vec: np.ndarray, qpos_map: dict[int, int]) -> None:
    data.qpos[:] = 0.0
    if model.nq >= 7:
        data.qpos[0:7] = np.array([0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for dataset_idx, qpos_addr in qpos_map.items():
        data.qpos[qpos_addr] = float(action_vec[dataset_idx])
    mujoco.mj_forward(model, data)


def _render(renderer: Any, data: Any, args: ArgsConfig) -> np.ndarray:
    if args.camera == "free":
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.0, 0.85]
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
        renderer.update_scene(data, camera=camera)
        return renderer.render().astype(np.uint8)
    try:
        renderer.update_scene(data, camera=args.camera)
    except Exception:
        renderer.update_scene(data)
    return renderer.render().astype(np.uint8)


def _resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[-1]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _caption(frame: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    out = np.ascontiguousarray(frame.copy())
    cv2.rectangle(out, (0, 0), (out.shape[1], 58), (0, 0, 0), thickness=-1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if subtitle:
        cv2.putText(out, subtitle, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    return out


def _write_episode(
    args: ArgsConfig,
    traj_id: int,
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    model: Any,
    qpos_map: dict[int, int],
    dataset_joint_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    modality_configs = policy.get_modality_config()
    action_keys = loader.modality_configs["action"].modality_keys
    traj = loader[traj_id]
    steps = min(args.max_steps, len(traj))

    gt_data = mujoco.MjData(model)
    pred_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    recorder = VideoRecorder.create_h264(fps=args.fps, crf=22, input_pix_fmt="rgb24")

    video_path = output_dir / f"traj_{traj_id}_three_panel.mp4"
    csv_path = output_dir / f"traj_{traj_id}_three_panel.csv"
    recorder.start(video_path)

    pred_cache: list[np.ndarray] = []
    inference_ms: list[float] = []
    action_l2_errors: list[float] = []
    action_mae_errors: list[float] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "inference_ms", "action_l2_error", "action_mae_error"],
        )
        writer.writeheader()

        for step in range(steps):
            if not pred_cache:
                obs = _make_policy_observation(
                    traj,
                    step,
                    loader,
                    modality_configs,
                    embodiment_tag,
                )
                start = time.perf_counter()
                action_chunk, _ = policy.get_action(obs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                inference_ms.append(elapsed_ms)
                horizon = min(args.action_horizon, next(iter(action_chunk.values())).shape[1])
                pred_cache = [
                    _concat_action(action_chunk, list(action_keys), action_step)
                    for action_step in range(horizon)
                ]
            else:
                elapsed_ms = 0.0

            gt_action = _extract_vector(traj, step, list(action_keys), "action")
            pred_action = pred_cache.pop(0)
            _set_pose(model, gt_data, gt_action, qpos_map)
            _set_pose(model, pred_data, pred_action, qpos_map)

            head_frame = _resize_rgb(traj["video.head_cam"].iloc[step], (args.width, args.height))
            gt_frame = _render(renderer, gt_data, args)
            pred_frame = _render(renderer, pred_data, args)

            l2_error = float(np.linalg.norm(gt_action - pred_action))
            mae_error = float(np.mean(np.abs(gt_action - pred_action)))
            action_l2_errors.append(l2_error)
            action_mae_errors.append(mae_error)
            writer.writerow(
                {
                    "step": step,
                    "inference_ms": elapsed_ms if elapsed_ms else "",
                    "action_l2_error": l2_error,
                    "action_mae_error": mae_error,
                }
            )

            subtitle = f"traj={traj_id} step={step}/{steps - 1}"
            panels = [
                _caption(head_frame, "DATASET HEAD_CAM", subtitle),
                _caption(gt_frame, "MUJOCO G1 GROUND TRUTH", "dataset action replay"),
                _caption(pred_frame, "MUJOCO G1 PREDICTED", f"L2={l2_error:.3f} MAE={mae_error:.3f}"),
            ]
            recorder.write_frame(np.concatenate(panels, axis=1))

    recorder.stop()
    renderer.close()

    return {
        "traj_id": traj_id,
        "steps": steps,
        "video_path": str(video_path),
        "per_step_csv": str(csv_path),
        "mean_action_l2_error": float(np.mean(action_l2_errors)) if action_l2_errors else 0.0,
        "mean_action_mae_error": float(np.mean(action_mae_errors)) if action_mae_errors else 0.0,
        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else 0.0,
        "dataset_joint_names": dataset_joint_names,
    }


def main(args: ArgsConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = _create_policy(args)
    modality_configs = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    model = mujoco.MjModel.from_xml_path(args.mujoco_model_path)
    dataset_joint_names = _load_joint_names(Path(args.dataset_path))
    qpos_map = _qpos_addresses(model, dataset_joint_names)
    traj_ids = _resolve_traj_ids(args, len(loader))

    summaries = []
    for traj_id in traj_ids:
        logging.info("Rendering three-panel MuJoCo replay for traj_id=%s", traj_id)
        summaries.append(
            _write_episode(
                args=args,
                traj_id=traj_id,
                policy=policy,
                loader=loader,
                model=model,
                qpos_map=qpos_map,
                dataset_joint_names=dataset_joint_names,
                output_dir=output_dir,
            )
        )

    summary = {
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "mujoco_model_path": args.mujoco_model_path,
        "traj_ids": traj_ids,
        "requested_traj_ids": args.traj_ids,
        "start_traj_id": args.start_traj_id,
        "num_trajs": args.num_trajs,
        "all_trajs": args.all_trajs,
        "action_horizon": args.action_horizon,
        "panel_order": ["dataset_head_cam", "mujoco_ground_truth_replay", "mujoco_predicted_replay"],
        "episodes": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved three-panel videos to: {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
