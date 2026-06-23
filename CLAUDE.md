# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.10 (dGPU, Orin); Python 3.12 (Thor, DGX Spark — see deployment dir)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (`pyproject.toml`)
- **CI:** internal GitLab CI; public GitHub Actions (`.github/workflows/`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Run a single test
python -m pytest tests/gr00t/model/test_action_head.py -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- `__init__.py` ignores F401; isort uses `lines-after-imports = 2`, `combine-as-imports = true`

## Architecture

### Model forward pass

The model (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`) has two components wired together in `Gr00tN1d7`:

1. **Backbone** (`Gr00tN1d7Backbone`): `Cosmos-Reason2-2B` (Qwen3-VL) VLM. Processes video + language → `backbone_features (B, S, 2048)` + `backbone_attention_mask`.

2. **Action head** (`Gr00tN1d7ActionHead`): Flow-matching diffusion policy (`AlternateVLDiT` by default) that cross-attends to `backbone_features` to denoise action trajectories. Sub-modules:
   - `state_encoder`: per-embodiment `CategorySpecificMLP`; input state shape `(B, H, max_state_dim)` → reshaped to `(B, 1, H*max_state_dim)` before encoding
   - `action_encoder` / `action_decoder`: per-embodiment projections
   - `model`: `AlternateVLDiT` (alternates image/text cross-attention every N blocks) or plain `DiT`

**`AlternateVLDiT` vs `DiT`:** `AlternateVLDiT` requires both `image_mask` (bool marking image vs text tokens) and `backbone_attention_mask`. Plain `DiT` needs only `encoder_attention_mask`. Whenever you extend `encoder_hidden_states` (e.g. appending a new token), you must extend both masks for `AlternateVLDiT`.

### Checkpoint loading (`gr00t/model/gr00t_n1d7/setup.py`)

`Gr00tN1d7Pipeline._create_model()` uses **strict weight matching** — any missing or unexpected keys (beyond `mask_token`) raise `RuntimeError`. This means:

- You **cannot** add new submodules to the action head config before loading a base checkpoint.
- Pattern for extending the model post-load: load with the new params disabled (e.g. `use_recap=False`), then initialise new components after `pipeline.return_model()`, then patch `model.config.*` so those flags are saved in every downstream checkpoint.

### Modality config system

Each embodiment's data format is described by a modality config: a `dict` mapping keys (`video`, `state`, `action`, `language`) to `ModalityConfig` objects (`gr00t/data/types.py`).

- Built-in configs: `gr00t/configs/data/embodiment_configs.py`
- Custom configs: `data_config/` (loaded by `importlib` at runtime via `--modality-config-path`)
- Registration: `register_modality_config(config_dict)` must be called; happens automatically on import
- The dict key must match the dataset's `meta/modality.json` at training time

`ModalityConfig.delta_indices` specifies which frame offsets to load. `ActionConfig` within the action modality specifies representation (relative/absolute), type (EEF/non-EEF), and format (XYZ_ROT6D, etc.).

### Training pipeline

Standard fine-tune: `gr00t/experiment/launch_finetune.py` → `gr00t/experiment/experiment.py:run()` → `Gr00tTrainer`.

- **Config layering:** `get_default_config()` provides base config; `launch_finetune.py` overlays `FinetuneConfig` CLI values
- **`FinetuneConfig`** (`gr00t/configs/finetune_config.py`): CLI schema parsed by `tyro`; extend via `@dataclass` subclass
- **`Gr00tTrainer`** (`gr00t/experiment/trainer.py`): HF `Trainer` subclass with profiling callback
- **Optimizer creation** happens inside HF `Trainer.__init__` — any parameter freeze/unfreeze must happen **before** trainer creation

### RECAP (RL from demonstrations)

RECAP is a two-phase advantage-conditioned training strategy (arXiv:2511.14759 / arXiv:2505.23458).

Key files:
- `gr00t/model/modules/recap.py` — `AdvantageEmbedding`, `DistributionalValueHead`, `compute_normalised_returns`
- `gr00t/configs/model/gr00t_n1d7.py` — RECAP fields on `Gr00tN1d7Config` (`use_recap`, `recap_alpha`, etc.)
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` — RECAP methods on `Gr00tN1d7ActionHead`
- `gr00t/experiment/launch_recap.py` — standalone launcher (extends `FinetuneConfig` → `RECAPFinetuneConfig`)
- `scripts/train_recap_vrc2.sh` — two-phase bash driver (`phase1` / `phase2` / `both`)

**Phase 1** (`--recap-phase value_head`): all policy params frozen, trains `DistributionalValueHead` via cross-entropy on discretised returns. Reward fields (`reward`, `reward.current_frame_idx`, `reward.episode_lengths`) are **optional** — defaults assume all-success (correct for curated demo datasets).

**Phase 2** (`--recap-phase policy`): value head frozen (used for advantage labelling), policy trained with dual-term loss: `L = ||uncond||² + α · ||cond||²`. CFG dropout (`advantage_cfg_dropout_prob`) trains both uncond and cond branches.

**Advantage token** (`AdvantageEmbedding`): 3-class embedding (`NULL=0`, `NEG=1`, `POS=2`) appended to `encoder_hidden_states`. For `AlternateVLDiT`, its `image_mask` entry = `False`, `backbone_attention_mask` entry = `True`.

**Checkpoint loading caveat:** load base checkpoint first with `use_recap=False` (avoids missing-key `RuntimeError`), then call `action_head._init_recap()` post-load, then `action_head.set_phase_value_head()` or `set_phase_policy()`. Phase 1 saves `use_recap=True` in `config.json`, so Phase 2 loads with RECAP weights already present.

## Key entry points

- **Fine-tune:** `torchrun --nproc_per_node=N gr00t/experiment/launch_finetune.py --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **RECAP fine-tune:** `torchrun --nproc_per_node=N gr00t/experiment/launch_recap.py --recap-phase value_head|policy ...`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Testing

- Test markers: `gpu` (requires GPU); default runs are CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- `tests/test_support/` has helpers for syncing, runtime detection, and README code-block extraction

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — `scripts/deployment/dgpu/install_deps.sh`, `docker/Dockerfile`
- **Jetson Orin:** CUDA 12.6 — `scripts/deployment/orin/`
- **Jetson Thor:** CUDA 13.0 — `scripts/deployment/thor/`
- **DGX Spark:** CUDA 13.0 — `scripts/deployment/spark/`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, etc.) that exports platform-specific library paths. For dGPU, `source .venv/bin/activate` is sufficient.

## Custom embodiment data configs (`data_config/`)

Files in `data_config/` are **not** part of the installed package. Each must call `register_modality_config(config_dict)`. Current configs:

- `vrc2_left_arm_eef_rel_2cam.py` — VRC2 left arm, relative EEF, 2 cameras (head + left)
- `vrc2_right_arm_eef_abs.py` — VRC2 right arm, absolute EEF
- `vrc2_right_arm_eef_rel.py` — VRC2 right arm, relative EEF

The N1.5 reference implementation lives in `gr00tn1.5/` (separate package, not imported by N1.7).
