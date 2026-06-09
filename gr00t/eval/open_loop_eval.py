# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    filtered_pred_action_across_time: np.ndarray | None,
    filtered_pred_action_label: str,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        if filtered_pred_action_across_time is None:
            ax.plot(pred_action_across_time[:, action_idx], label="pred action")
        else:
            ax.plot(pred_action_across_time[:, action_idx], label="pred action raw", alpha=0.35)
            ax.plot(
                filtered_pred_action_across_time[:, action_idx],
                label=filtered_pred_action_label,
            )

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


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


def apply_one_euro_filter(
    action_across_time: np.ndarray,
    freq: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    action_filter = OneEuroVectorFilter(
        freq=freq,
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
    )
    return np.stack([action_filter(action) for action in action_across_time], axis=0)


def apply_action_delta_clip(action_across_time: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0.0:
        raise ValueError(f"max_delta must be positive, got {max_delta}")

    clipped_actions = []
    prev_action = None
    for action in action_across_time:
        action = np.asarray(action, dtype=np.float32)
        if prev_action is None:
            clipped_action = action.copy()
        else:
            clipped_action = np.clip(
                action,
                prev_action - max_delta,
                prev_action + max_delta,
            )
        clipped_actions.append(clipped_action)
        prev_action = clipped_action

    return np.stack(clipped_actions, axis=0).astype(np.float32)


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
    smooth_actions=False,
    one_euro_freq=30.0,
    one_euro_min_cutoff=1.0,
    one_euro_beta=0.05,
    one_euro_d_cutoff=1.0,
    action_delta_clip=0.0,
):
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    for step_count in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        _action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(action_horizon):
            # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    filtered_pred_action_across_time = None
    filtered_pred_action_label = "pred action one-euro"
    if smooth_actions:
        filter_input_action_across_time = pred_action_across_time
        if action_delta_clip > 0.0:
            filter_input_action_across_time = apply_action_delta_clip(
                pred_action_across_time,
                max_delta=action_delta_clip,
            )
            filtered_pred_action_label = "pred action clipped+one-euro"

        filtered_pred_action_across_time = apply_one_euro_filter(
            filter_input_action_across_time,
            freq=one_euro_freq,
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            d_cutoff=one_euro_d_cutoff,
        )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")
    if filtered_pred_action_across_time is not None:
        if action_delta_clip > 0.0:
            clipped_mse = np.mean((gt_action_across_time - filter_input_action_across_time) ** 2)
            clipped_mae = np.mean(np.abs(gt_action_across_time - filter_input_action_across_time))
            logging.info(f"Delta-clipped Action MSE across single traj: {clipped_mse}")
            logging.info(f"Delta-clipped Action MAE across single traj: {clipped_mae}")
        filtered_mse = np.mean((gt_action_across_time - filtered_pred_action_across_time) ** 2)
        filtered_mae = np.mean(np.abs(gt_action_across_time - filtered_pred_action_across_time))
        logging.info(f"One Euro filtered Action MSE across single traj: {filtered_mse}")
        logging.info(f"One Euro filtered Action MAE across single traj: {filtered_mae}")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        filtered_pred_action_across_time=filtered_pred_action_across_time,
        filtered_pred_action_label=filtered_pred_action_label,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    smooth_actions: bool = False
    """Apply One Euro filtering to predicted actions in the open-loop plot."""

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


def main(args: ArgsConfig):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy = PolicyClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=args.save_plot_path,
            smooth_actions=args.smooth_actions,
            one_euro_freq=args.one_euro_freq,
            one_euro_min_cutoff=args.one_euro_min_cutoff,
            one_euro_beta=args.one_euro_beta,
            one_euro_d_cutoff=args.one_euro_d_cutoff,
            action_delta_clip=args.action_delta_clip,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
