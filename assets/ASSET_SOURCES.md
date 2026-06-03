# Downloaded Robot Assets

Ngay tai: 2026-06-02

## Unitree G1 USD, 29DoF

Nguon: Hugging Face dataset `unitreerobotics/unitree_model`

URL:

```text
https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main/G1/29dof/usd/g1_29dof_rev_1_0
```

Local path:

```text
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_base.usd
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_physics.usd
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_sensor.usd
```

Ghi chu: day la USD asset phu hop voi Isaac Sim/Isaac Lab. MuJoCo Python khong load truc tiep USD trong workflow hien tai.

## Unitree G1 MuJoCo, 29DoF

Nguon: GitHub repo `unitreerobotics/unitree_mujoco`

URL:

```text
https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/g1
```

Local path:

```text
assets/unitree_mujoco/unitree_robots/g1/g1_29dof.xml
assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
assets/unitree_mujoco/unitree_robots/g1/meshes/
```

Validation:

```text
mujoco.MjModel.from_xml_path("assets/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
OK: nq=36, nv=35, nu=29, njnt=30
```

Ghi chu: dung `g1_29dof.xml` cho MuJoCo replay/render. File nay va thu muc `meshes/` da duoc test load thanh cong bang `mujoco==3.9.0`.

## LeRobot Unitree G1 MuJoCo, body29 + hand14

Nguon: Hugging Face model repo `lerobot/unitree-g1-mujoco`

URL:

```text
https://huggingface.co/lerobot/unitree-g1-mujoco/tree/main/assets
```

Local path:

```text
assets/lerobot_unitree_g1_mujoco/assets/g1_29dof_with_hand.xml
assets/lerobot_unitree_g1_mujoco/assets/scene_43dof.xml
assets/lerobot_unitree_g1_mujoco/assets/g1_body29_hand14.urdf
assets/lerobot_unitree_g1_mujoco/assets/meshes/
assets/lerobot_unitree_g1_mujoco/assets/meshes_exo_left/
assets/lerobot_unitree_g1_mujoco/assets/meshes_exo_right/
```

Validation:

```text
mujoco.MjModel.from_xml_path("assets/lerobot_unitree_g1_mujoco/assets/g1_29dof_with_hand.xml")
OK: nq=50, nv=49, nu=43, njnt=44
```

Ghi chu: day la model duoc dung cho video three-panel vi dataset co 28D arm+Dex3 hand action. `scene_43dof.xml` co `global_view` va `head_camera`, nhung script replay dung free camera gan hon de thay ro tay.
