# Scene-adaptive ensemble object detection for UAV imagery (SCAC)

Code accompanying the PhD thesis **«Оптимізація ансамблевих методів розпізнавання об'єктів на основі нейронних мереж для систем кібербезпеки»** (Vladyslav Kromkach, Lviv Polytechnic National University) and the two articles listed under [Citation](#citation).

The thesis treats a detector ensemble inside a security video-monitoring pipeline as an optimisation problem with three decision variables — **pool composition**, **prediction fusion** (NMS / Soft-NMS / NMW / WBF) and **compute allocation** between a cheap and an expensive configuration — under constraints on compute budget, robustness to deliberate interference and trustworthiness of evaluation. This repository contains the core library that implements the studied methods:

- a corrected evaluation protocol for dense aerial imagery (full-curve mAP@50 at the evaluation threshold, class-layout remapping, `predict`-path inference) with a paired / cluster bootstrap for comparing configurations;
- the ensemble detector with four fusion methods and per-model weight search;
- the robustness test bench for a **compromised pool member** (suppression and phantom-injection primitives, per-class fusion cache, bootstrap, capped confirmation multiplier);
- **SCAC** — a two-tier cascade with an annotation-free scene-complexity signal (C-count), a cross-fitted escalation threshold, an oracle feasibility gate and an escalation guard, packaged as a NumPy-only library (`libs/scac`).

## Repository layout

| Component | Modules | Purpose | Thesis |
|---|---|---|---|
| `src/data` | `converter`, `scene_splitter`, `weather_augmentation` | VisDrone → YOLO conversion, heuristic scene split, synthetic weather | §2.6 |
| `src/scene_classifier` | `classifier` (heuristic, MobileNetV3), `train` | scene classifier used for routing | §3.5 |
| `src/ensemble` | `methods`, `ensemble_detector`, `optimizer`, `cached_search` | NMS / Soft-NMS / NMW / WBF, multi-model inference, weight search, fusion over cached predictions | §3.1–3.5 |
| `src/ensemble/scac.py` | `signal_values`, `per_image_ap50`, `eval_split` | complexity signals, per-frame gain, cascade evaluation bench | §4.1–4.3 |
| `src/evaluation` | `metrics`, `bootstrap`, `profiler`, `ground_truth` | mAP (COCO-101), paired / cluster bootstrap, GFLOPs and latency | §2.3–2.6 |
| `src/pipeline.py` | `DetectionPipeline`, `AdaptiveEnsembleDetector` | end-to-end inference: classifier → ensemble → fusion | §2.6 |
| `libs/scac` | `signals`, `policy`, `calibrate`, `gate`, `guard`, `cli` | installable deployment core of the cascade (NumPy only) | §4.6 |
| `exp36` | `attacks`, `fusion`, `eval`, `runner`, `report` | attack primitives A1–A3, per-class fusion cache, grid, bootstrap, reports | §3.6 |
| `configs/default.yaml` | Pydantic schema in `src/config.py` | single source of hyperparameters and protocol settings | §2.6 |
| `results/uavdt` | two JSON files | reference cascade result on UAVDT and the library parity check | §4.3, §4.6 |

### What is deliberately not included

- The numbered experiment scripts (`scripts/00–15`) and the prediction caches they operate on. Re-running the result tables requires the datasets and several gigabytes of cached detector outputs; an archive of the experiment artefacts can be provided on request.
- Datasets and model weights (see [Data and models](#data-and-models)).
- The dissertation and article sources.

## Install and test

Python 3.10 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --group dev          # runtime + dev dependencies, scac installed editable
uv run pytest -m "not slow"  # core, exp36 and scac unit tests
uv run ruff check .
```

Only `libs/scac` is needed for deployment:

```bash
pip install ./libs/scac      # single dependency: numpy
```

## Quick start

End-to-end inference with the heuristic scene classifier (models are downloaded by Ultralytics on first use):

```bash
uv run python -m src.pipeline --input path/to/image_or_dir --output out/ --heuristic
```

Is a cascade worth building? Gate first, on per-frame AP of tier 1 and tier 2 over cached predictions:

```python
from scac import gate_stats

delta = ap_tier2 - ap_tier1        # per-frame gain of the ensemble over tier 1 (NaN = no GT)
print(gate_stats(delta))           # sparse gain (UAVDT: 41 % positive) -> cascade pays off
                                   # dense gain (VisDrone: 75-90 %)   -> it does not
```

The same over a JSON frame table from the command line, with tier costs in GFLOPs per frame:

```bash
uv run scac gate frames.json --b1 684.4 --b2 1605.5
```

See `libs/scac/README.md` for the full interface (threshold calibration by cross-fitting, the cascade runner with tier-1 prediction reuse, the escalation guard).

## Data and models

- **VisDrone-DET 2019** — official release from the VisDrone challenge. The project uses an 8-class layout (tricycle and awning-tricycle dropped); see `src/constants.py` and `configs/default.yaml`.
- **UAVDT** — official release (zero-shot transfer target, official day / night / fog attributes).
- **Detectors** — Ultralytics YOLOv8 weights are downloaded automatically; VisDrone-fine-tuned YOLOv8 weights (`mshamrai/yolov8*-visdrone`, 10-class layout, remapped before evaluation) and the heterogeneous pool (YOLOv9e, YOLOv10x, YOLO11x, YOLO26x from the dronefreak VisDrone Model Zoo, Apache-2.0) are referenced in `configs/default.yaml` under `finetuned_models`.

Expected locations are `data/VisDrone`, `data/UAVDT` and `weights/finetuned`; all are git-ignored.

## Reference result

`results/uavdt/scac_crossfit_bootstrap.json` is the cross-fitted SCAC result on UAVDT (1372 frames, 50 sequences): the cascade with the C-count signal matches the full three-model ensemble in scene-balanced mAP@50 (0.618 vs 0.611) at 45 % of its compute (1024 vs 2290 GFLOPs per frame). `results/uavdt/scac_lib_parity.json` records that `libs/scac` reproduces that run exactly (fold-wise p*, τ and calibration mAP@50, realised escalation fraction overall and per scene).

## Citation

Thesis: Кромкач В. О. *Оптимізація ансамблевих методів розпізнавання об'єктів на основі нейронних мереж для систем кібербезпеки*: дис. … д-ра філософії: 123 Комп'ютерна інженерія. Львів: Національний університет «Львівська політехніка».

Articles:

1. Руда Х. С., Кромкач В. О. Сценозалежне ансамблеве виявлення об'єктів на аерознімках безпілотників: межі корисності маршрутизації кадрів за сценами. *Науковий вісник НЛТУ України*. 2026. Т. 36, № 4. С. 166–177. https://doi.org/10.36930/40360419
2. Руда Х. С., Кромкач В. О. Адаптивний каскад за складністю сцени для виявлення об'єктів на аерознімках безпілотників: коли вибіркова ескалація обчислень виправдана. *Кібербезпека: освіта, наука, техніка*. 2026. № 2(34). С. 463–502. https://doi.org/10.28925/2663-4023.2026.34.1340

Machine-readable metadata is in `CITATION.cff`.

## License

MIT — see `LICENSE`.

---

## Українською

Програмна реалізація дисертації «Оптимізація ансамблевих методів розпізнавання об'єктів на основі нейронних мереж для систем кібербезпеки» (Кромкач В. О., Національний університет «Львівська політехніка», спеціальність 123 Комп'ютерна інженерія).

У репозиторії — ядро програмного комплексу: виправлений протокол оцінювання детекторів на щільних аерознімках зі статистичним апаратом парного і кластерного бутстрепу (`src/evaluation`), ансамблевий детектор із чотирма методами злиття (`src/ensemble`), стенд стійкості злиття до компрометації члена пулу (`exp36`), а також бібліотека `scac` — експлуатаційне ядро каскаду за складністю сцени із сигналом C-count, крос-фітованим калібруванням порога, оракульним ґейтом доцільності та обмежувачем ескалації (`libs/scac`, єдина залежність NumPy).

Не включено: нумеровані сценарії етапів експериментів і кеші передбачень (потрібні набори даних і кілька гігабайт кешу; архів надається на запит), набори даних, ваги моделей, текст дисертації.

Встановлення й тести: `uv sync --group dev`, `uv run pytest -m "not slow"`. Еталонний результат каскаду на UAVDT і перевірка паритету бібліотеки — у `results/uavdt`.
