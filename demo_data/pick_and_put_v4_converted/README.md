# pick_and_put_v4_converted

Place the custom Unitree G1 Dex3 LeRobot dataset here.

Expected layout:

```text
demo_data/pick_and_put_v4_converted/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── modality.json
├── data/
│   └── chunk-000/
└── videos/
    └── chunk-000/
```

The dataset contents are intentionally ignored by git. The expected canonical keys are:

```text
video: head_cam, left_wrist_cam
state: left_arm, left_hand
action: left_arm, left_hand
language: annotation.human.task_description
```
