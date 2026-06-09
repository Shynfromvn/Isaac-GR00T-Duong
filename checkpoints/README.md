# Checkpoints

Put local fine-tuned checkpoints here. The real checkpoint files are intentionally ignored by git.

Expected layout:

```text
checkpoints/
├── checkpoint-50000/
├── checkpoint-100000/
├── checkpoint-150000/
└── checkpoint-200000/
```

Each checkpoint folder should contain the GR00T checkpoint files, for example `config.json`, `processor_config.json`, `statistics.json`, `model-*.safetensors`, and related index/config files.
