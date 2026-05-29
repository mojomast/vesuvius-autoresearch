# Config And Data Path Audit

`autoresearch.py` uses `configs/baseline.yaml` and pivot configs `robust_multisegment_dice035_expanded.yaml`, `robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml`, `robust_tta_seed_ensemble.yaml`, and `residual_25d_torch_unet_cpu.yaml`.

Baseline must define `dataset.train_npz`, `dataset.val_npz`, `dataset.patch_size`, `model.name`, `training.epochs`, `training.learning_rate`, `training.weight_decay`, `training.pos_weight`, `evaluation.main_metric`, `evaluation.threshold`, and `outputs.runs_dir`. Recommended research metadata includes `dataset.research_scope`, `dataset.validation_mode`, and `autoresearch.scope_policy`.

The four pivot configs unlock robust search when they exist and contain both `dataset.train_npz` and `dataset.val_npz`; torch pivots should use a model name containing `torch`, repo-relative prepared NPZ paths, `evaluation.main_metric: val_f1`, and a robust multi-segment `autoresearch.scope_policy`.

Hardcoded paths are repo-relative. Baseline uses `data/real_cross/segment_20230827161847/train.npz` and `data/real_cross/segment_20230520175435/val.npz`. Promotion LOO uses `data/real_cross_folds_expanded_combined/fold_map.json`.
