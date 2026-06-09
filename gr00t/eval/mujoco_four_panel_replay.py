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
    dataset_path: str = "demo_data/pick_and_put_v4_converted"
    """Dataset that provides head/wrist video, state, action, and task text."""

    model_path: str | None = "checkpoints/checkpoint-200000"
    """Local checkpoint. Leave empty only when using --host/--port policy server."""

    embodiment_tag: str = "NEW_EMBODIMENT"
    """Embodiment tag used by the checkpoint/server."""

    mujoco_model_path: str = "assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml"
    """MuJoCo XML with G1 body29+hand14 model."""

    output_dir: str = "my-outputs/mujoco_four_panel_eval"
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

    smooth_actions: bool = False
    """Apply One Euro filtering to predicted actions before MuJoCo replay."""

    one_euro_freq: float = 30.0
    """Sampling frequency for One Euro filtering. Use dataset FPS for per-step actions."""

    one_euro_min_cutoff: float = 1.0
    """Minimum cutoff frequency for One Euro filtering; lower values smooth more at low speed."""

    one_euro_beta: float = 0.05
    """Speed coefficient for One Euro filtering; higher values reduce lag on fast changes."""

    one_euro_d_cutoff: float = 1.0
    """Cutoff frequency for the derivative used by One Euro filtering."""

    action_delta_clip: float = 0.0
    """If positive, clip per-step predicted action deltas before One Euro filtering."""


def _load_joint_names(dataset_path: Path, action_keys: list[str]) -> list[str]:
    info_path = dataset_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    names = info.get("features", {}).get("action", {}).get("names")
    all_names = list(names[0]) if names and names[0] else DEFAULT_DATASET_JOINT_NAMES

    modality_path = dataset_path / "meta" / "modality.json"
    with modality_path.open("r", encoding="utf-8") as f:
        modality_meta = json.load(f)
    action_meta = modality_meta.get("action", {})

    selected_names = []
    for key in action_keys:
        if key not in action_meta:
            raise ValueError(
                f"Action key {key!r} is missing from {modality_path}. "
                f"Available action keys: {list(action_meta)}."
            )
        start = int(action_meta[key]["start"])
        end = int(action_meta[key]["end"])
        selected_names.extend(all_names[start:end])

    return selected_names


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


def _smoothing_alpha(cutoff: np.ndarray | float, freq: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


class OneEuroVectorFilter:
    """One Euro filter for a vector action stream."""

    def __init__(
        self,
        freq: float,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
    ) -> None:
        if freq <= 0.0:
            raise ValueError(f"freq must be positive, got {freq}")
        if min_cutoff <= 0.0:
            raise ValueError(f"min_cutoff must be positive, got {min_cutoff}")
        if d_cutoff <= 0.0:
            raise ValueError(f"d_cutoff must be positive, got {d_cutoff}")

        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.prev_x: np.ndarray | None = None
        self.prev_x_hat: np.ndarray | None = None
        self.prev_dx_hat: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.prev_x is None:
            self.prev_x = x.copy()
            self.prev_x_hat = x.copy()
            self.prev_dx_hat = np.zeros_like(x)
            return x.copy()

        dx = (x - self.prev_x) * self.freq
        d_alpha = _smoothing_alpha(self.d_cutoff, self.freq)
        dx_hat = d_alpha * dx + (1.0 - d_alpha) * self.prev_dx_hat

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = _smoothing_alpha(cutoff, self.freq)
        x_hat = alpha * x + (1.0 - alpha) * self.prev_x_hat

        self.prev_x = x.copy()
        self.prev_x_hat = x_hat.copy()
        self.prev_dx_hat = dx_hat.copy()
        return x_hat.astype(np.float32)


class StreamingActionSmoother:
    """Apply optional delta clipping and One Euro filtering step by step."""

    def __init__(self, args: ArgsConfig) -> None:
        self.enabled = args.smooth_actions
        self.action_delta_clip = float(args.action_delta_clip)
        if self.action_delta_clip < 0.0:
            raise ValueError(f"action_delta_clip must be non-negative, got {self.action_delta_clip}")
        self.prev_clipped_action: np.ndarray | None = None
        self.filter = OneEuroVectorFilter(
            freq=args.one_euro_freq,
            min_cutoff=args.one_euro_min_cutoff,
            beta=args.one_euro_beta,
            d_cutoff=args.one_euro_d_cutoff,
        )

    def __call__(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if not self.enabled:
            return action

        filter_input = action
        if self.action_delta_clip > 0.0:
            if self.prev_clipped_action is None:
                filter_input = action.copy()
            else:
                filter_input = np.clip(
                    action,
                    self.prev_clipped_action - self.action_delta_clip,
                    self.prev_clipped_action + self.action_delta_clip,
                )
            self.prev_clipped_action = filter_input.copy()

        return self.filter(filter_input)


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
    if len(action_vec) < len(qpos_map):
        raise ValueError(
            f"Action vector has {len(action_vec)} values, but MuJoCo qpos map expects "
            f"{len(qpos_map)} joints. Check meta/modality.json action slices."
        )
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
    video_keys = loader.modality_configs["video"].modality_keys
    traj = loader[traj_id]
    steps = min(args.max_steps, len(traj))

    gt_data = mujoco.MjData(model)
    pred_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    recorder = VideoRecorder.create_h264(fps=args.fps, crf=22, input_pix_fmt="rgb24")

    video_path = output_dir / f"traj_{traj_id}_four_panel.mp4"
    csv_path = output_dir / f"traj_{traj_id}_four_panel.csv"
    recorder.start(video_path)

    pred_cache: list[np.ndarray] = []
    action_smoother = StreamingActionSmoother(args)
    inference_ms: list[float] = []
    action_l2_errors: list[float] = []
    action_mae_errors: list[float] = []
    raw_action_l2_errors: list[float] = []
    raw_action_mae_errors: list[float] = []

    required_video_keys = ["head_cam", "left_wrist_cam"]
    missing_video_keys = [key for key in required_video_keys if key not in video_keys]
    if missing_video_keys:
        raise ValueError(
            f"Missing required video keys {missing_video_keys}. "
            f"Expected canonical keys {required_video_keys}, got {video_keys}."
        )
    head_cam_key = "head_cam"
    wrist_cam_key = "left_wrist_cam"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "inference_ms",
                "raw_action_l2_error",
                "raw_action_mae_error",
                "action_l2_error",
                "action_mae_error",
                "action_is_smoothed",
            ],
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
            raw_pred_action = pred_cache.pop(0)
            pred_action = action_smoother(raw_pred_action)
            _set_pose(model, gt_data, gt_action, qpos_map)
            _set_pose(model, pred_data, pred_action, qpos_map)

            # Get both camera frames from dataset
            head_frame = _resize_rgb(traj[f"video.{head_cam_key}"].iloc[step], (args.width, args.height))
            wrist_frame = _resize_rgb(traj[f"video.{wrist_cam_key}"].iloc[step], (args.width, args.height))
            gt_frame = _render(renderer, gt_data, args)
            pred_frame = _render(renderer, pred_data, args)

            raw_l2_error = float(np.linalg.norm(gt_action - raw_pred_action))
            raw_mae_error = float(np.mean(np.abs(gt_action - raw_pred_action)))
            l2_error = float(np.linalg.norm(gt_action - pred_action))
            mae_error = float(np.mean(np.abs(gt_action - pred_action)))
            raw_action_l2_errors.append(raw_l2_error)
            raw_action_mae_errors.append(raw_mae_error)
            action_l2_errors.append(l2_error)
            action_mae_errors.append(mae_error)
            writer.writerow(
                {
                    "step": step,
                    "inference_ms": elapsed_ms if elapsed_ms else "",
                    "raw_action_l2_error": raw_l2_error,
                    "raw_action_mae_error": raw_mae_error,
                    "action_l2_error": l2_error,
                    "action_mae_error": mae_error,
                    "action_is_smoothed": args.smooth_actions,
                }
            )

            subtitle = f"traj={traj_id} step={step}/{steps - 1}"
            # Create 2x2 layout: head cam | wrist cam on top row, GT | Pred on bottom row
            top_row = np.concatenate([
                _caption(head_frame, "DATASET HEAD CAM", subtitle),
                _caption(wrist_frame, "DATASET WRIST CAM", subtitle),
            ], axis=1)
            bottom_row = np.concatenate([
                _caption(gt_frame, "MUJOCO G1 GROUND TRUTH", "dataset action replay"),
                _caption(
                    pred_frame,
                    "MUJOCO G1 PREDICTED SMOOTHED"
                    if args.smooth_actions
                    else "MUJOCO G1 PREDICTED",
                    f"L2={l2_error:.3f} MAE={mae_error:.3f}",
                ),
            ], axis=1)
            recorder.write_frame(np.concatenate([top_row, bottom_row], axis=0))

    recorder.stop()
    renderer.close()

    return {
        "traj_id": traj_id,
        "steps": steps,
        "video_path": str(video_path),
        "per_step_csv": str(csv_path),
        "mean_raw_action_l2_error": float(np.mean(raw_action_l2_errors))
        if raw_action_l2_errors
        else 0.0,
        "mean_raw_action_mae_error": float(np.mean(raw_action_mae_errors))
        if raw_action_mae_errors
        else 0.0,
        "mean_action_l2_error": float(np.mean(action_l2_errors)) if action_l2_errors else 0.0,
        "mean_action_mae_error": float(np.mean(action_mae_errors)) if action_mae_errors else 0.0,
        "action_is_smoothed": args.smooth_actions,
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
    action_keys = list(loader.modality_configs["action"].modality_keys)
    dataset_joint_names = _load_joint_names(Path(args.dataset_path), action_keys)
    qpos_map = _qpos_addresses(model, dataset_joint_names)
    traj_ids = _resolve_traj_ids(args, len(loader))

    summaries = []
    for traj_id in traj_ids:
        logging.info("Rendering four-panel MuJoCo replay for traj_id=%s", traj_id)
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
        "smoothing": {
            "enabled": args.smooth_actions,
            "one_euro_freq": args.one_euro_freq,
            "one_euro_min_cutoff": args.one_euro_min_cutoff,
            "one_euro_beta": args.one_euro_beta,
            "one_euro_d_cutoff": args.one_euro_d_cutoff,
            "action_delta_clip": args.action_delta_clip,
        },
        "panel_order": [
            "dataset_head_cam",
            "dataset_wrist_cam",
            "mujoco_ground_truth_replay",
            "mujoco_predicted_replay",
        ],
        "panel_layout": "2x2",
        "episodes": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved four-panel videos to: {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
