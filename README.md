# Ethics-of-AI
Spring 2026 - CS621 Project

![project overview](assets/absld-df-stages.png.png)


This repository contains experiments on robust and fair classification for CelebA-based binary prediction (target: `Smiling`) under clean and adversarial settings, including teacher training and distillation pipelines.

## What this project does

- Trains robust image classifiers (teacher/student variants) with PGD-based adversarial training.
- Evaluates clean and robust performance.
- Audits fairness metrics across multiple sensitive attributes.
- Supports teacher-to-student distillation experiments.

## Repository layout

```text
Ethics-of-AI/
├── README.md
├── requirements.txt
├── core/                    # Shared training/attack/metric logic
│   ├── attacks.py
│   ├── trainer.py
│   ├── kd_trainer.py
│   └── metrics.py
├── data/
│   └── celeba.py            # CelebA dataset + dataloaders
├── docs/                    # Optional project docs
├── src-teacher/             # Teacher and baseline training variants
│   ├── config.py
│   ├── mainr.py             # ResNet18 teacher training
│   ├── main.py              # MobileNet-v3-small variant
│   ├── mainl.py             # MobileNet-v3-large variant
│   ├── main3.py             # Additional experiment variant
│   ├── test.py              # Evaluation/testing script
│   ├── core/                # Local duplicate of core modules
│   └── data/                # Local duplicate of data loader
└── src-distil/              # Distillation training variants
   ├── config.py
   ├── main.py              # KD pipeline using loaded teacher checkpoint
   ├── mainl.py             # Alternate distillation variant
   ├── core/                # Local duplicate of core modules
   └── data/                # Local duplicate of data loader
```

## Key modules to modify

If you want to change behavior, start from these files:

1. `src-teacher/config.py` and `src-distil/config.py`
   - Update dataset/checkpoint paths (`DATA_DIR`, `CHECKPOINT_DIR`)
   - Change hyperparameters (`EPOCHS`, `LEARNING_RATE`, `EPSILON`, attack steps)
   - Change fairness target/sensitive attributes (`TARGET_ATTR`, `SENSITIVE_ATTRS`)

2. `core/trainer.py`
   - Controls teacher training and evaluation loops
   - Clean vs robust evaluation behavior
   - PGD schedule usage during training

3. `core/kd_trainer.py`
   - Distillation loss behavior and teacher-student objective

4. `core/attacks.py`
   - PGD attack settings and normalization path

5. `core/metrics.py`
   - Fairness metrics definitions and subgroup gap computation

6. `data/celeba.py`
   - Dataset loading, transforms, split logic, and sensitive-attribute extraction

## Important note on duplicated modules

The repository currently contains duplicated logic for keeping the separation of concerns under:
- `core/` and `data/` (root)
- `src-teacher/core/`, `src-teacher/data/`
- `src-distil/core/`, `src-distil/data/`

When making modifications, keep these copies synchronized or consolidate to a single shared module to avoid behavior drift between experiments.

## Data expectations

The dataloader expects the following files under `Config.DATA_DIR`:

```text
<DATA_DIR>/
├── imgs/
├── list_attr_celeba.csv
└── list_eval_partition.csv
```

Default path values are currently set inside each `config.py`. Update them before running if your dataset location differs.

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

## Running experiments

Run from the corresponding source directory so local imports resolve correctly.

### 1) Train teacher (ResNet18 variant)

```bash
cd src-teacher
python mainr.py
```

### 2) Train teacher (MobileNet variants)

```bash
cd src-teacher
python main.py
python mainl.py
```

### 3) Distillation run

```bash
cd src-distil
python main.py
```

> `src-distil/main.py` uses a teacher checkpoint path defined in `TEACHER_PATH`. Update that path if your checkpoint is elsewhere.

## Reproducibility checklist

- Confirm `DATA_DIR` and `CHECKPOINT_DIR` in active `config.py`.
- Keep model entrypoint and config paired (teacher vs distil).
- Log which script was run (`mainr.py`, `main.py`, `mainl.py`, etc.).
- Save checkpoint names used for distillation runs.
- Keep duplicate `core/` and `data/` logic aligned if editing.

## Minimal development workflow

1. Update config paths/hyperparameters.
2. Run training script from correct subfolder.
3. Inspect printed clean/robust/fairness metrics.
4. Save checkpoints and logs for comparison.

## Team

| Name | Roll No |
|---|---|
| Parv Thacker | 25210095 |
| Sankalp Turankar | 25210115 |
| Smrutee Behera | 25250041 |
| Balbir Prasad | 25210034 |
