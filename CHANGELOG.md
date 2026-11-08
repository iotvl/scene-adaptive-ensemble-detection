# Changelog

## 1.1.0 — 2026-11-08

### Added

- `NATIVE_CACHE_IMG_TOKENS` registry in `src/config.py`: a detector declared in `finetuned_models` with a framework other than `ultralytics` gets its own image-size token in the prediction cache (`native<eval_size>`); `cache_path` resolves it and `load_or_cache_predictions` refuses to run YOLO inference for such a key instead of silently producing a wrong cache. Tests in `tests/unit/test_scene_eval.py`.
- `finetuned_models.rtdetrv4l_vd` (RT-DETRv4-L fine-tuned on VisDrone, `dronefreak/visdrone-rtdetrv4-l`, Apache-2.0) in `configs/default.yaml`; exp36 pools `full8` and `hetero4`.
- `results/uavdt/scac_timed_pass.json`, `results/uavdt/scac_timed_pass_deployconf.json`: timed end-to-end passes of the UAVDT cascade at the evaluation and deployment thresholds.
- `results/uavdt/scac_crossfit_bootstrap_signoignore.json`: recalibration of the cascade on the complexity signal computed without the ignore-region mask.
- `results/rtdetrv4/`: RT-DETRv4-L as single model, pool member, routing candidate and cascade tier on VisDrone and UAVDT.

### Changed

- README: cross-paradigm member, timed passes, the role of the ignore-region mask in the UAVDT signal.

## 1.0.0 — 2026-11-05

Initial public release: core library (`src`), `scac` package (`libs/scac`), robustness bench (`exp36`), tests, reference UAVDT cascade result and library parity check.
