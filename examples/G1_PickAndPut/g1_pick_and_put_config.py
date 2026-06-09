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

"""
Modality configuration for Unitree G1 Dex3 with pick_and_put_v4_converted dataset.

Dataset info:
- Robot: Unitree G1 Dex3
- Cameras: head_cam (head), left_wrist_cam (wrist)
- Arms: left_arm only (7 DOF)
- Hands: left_hand only (7 DOF)
- Total action dimension: 14 DOF (7 arm + 7 hand)
- Task: "pick apple and put in the box"
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


g1_pick_and_put_config = {
    # Video: current frame only; keys must match "video" entries in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head_cam",        # head camera (top view)
            "left_wrist_cam",  # wrist camera (egocentric view)
        ],
    ),
    # State: current proprioceptive reading; keys must match "state" entries in meta/modality.json
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",   # 7 joint positions (shoulder, elbow, wrist)
            "left_hand",  # 7 finger joint positions
        ],
    ),
    # Action: 16-step prediction horizon; one ActionConfig per modality key
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "left_arm",
            "left_hand",
        ],
        action_configs=[
            # left_arm: RELATIVE = delta from current state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand: ABSOLUTE = target position (gripper/finger control works better absolute)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation field in the dataset
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(g1_pick_and_put_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
