# CALVIN Setup for `eval_calvin.py`

`eval_calvin.py` imports CALVIN via `sys.path` (see [eval_calvin.py:19-22](eval_calvin.py#L19-L22)), so you do **not** need to `pip install` the calvin packages — you just need the source trees on disk.

## 1. Clone CALVIN (with submodules)

```bash
# Adjust to your own root; the eval script defaults to /mnt/data/sftp/data/locht1/calvin
export CALVIN_ROOT=/mnt/data/sftp/data/locht1/calvin

mkdir -p $(dirname $CALVIN_ROOT)
git clone --recurse-submodules https://github.com/mees/calvin.git $CALVIN_ROOT
cd $CALVIN_ROOT
git submodule update --init --recursive
```

After this you should have:

```
$CALVIN_ROOT/
├── calvin_env/         # PyBullet simulation env (submodule)
├── calvin_models/      # task oracle + language annotations
└── ...
```

## 2. Install dependencies into the GR00T conda env

Use the existing GR00T env so CUDA / PyTorch versions stay consistent.

```bash
conda activate gr00t_equi_fa_simpler_fuse

# Core sim + IO deps used by calvin_env / play_table_env
pip install \
  pybullet \
  hydra-core==1.2.0 \
  omegaconf \
  gym==0.21.0 \
  imageio imageio-ffmpeg \
  opencv-python \
  scipy

# Language annotations / task oracle deps
pip install pyhash sentence-transformers
```

> Notes
> - **Do not** run `pip install -e calvin_env` or `pip install -e calvin_models` — `eval_calvin.py` adds these to `sys.path` directly. Editable installs can shadow the `gr00t` env's transformers / torch versions.
> - The `play_table_env.py` warnings about `version_base` and missing `_self_` in hydra are harmless.

## 3. (Optional) Download the dataset

The eval script **does not require the dataset** — it composes the env from `calvin_env`'s built-in hydra configs (Path B in [`_load_calvin_env`](eval_calvin.py#L185-L228)). Skip this section unless you specifically want the canonical config that ships with the dataset.

```bash
# ABCD-D split (~520 GB — needed only for training or canonical-config eval)
cd $CALVIN_ROOT/dataset
sh download_data.sh ABCD
# OR a single-scene debug split (~6 GB)
sh download_data.sh debug
```

If downloaded, pass `--args.dataset_path $CALVIN_ROOT/dataset/task_ABCD_D` to use the dataset's hydra config.

## 4. Smoke test (no policy required)

This verifies env build, EGL rendering, and video save. **Do this before touching the policy.**

```bash
cd /mnt/data/sftp/data/locht1/workspace/gr00t_equi_fa_simpler_fuse  # your GR00T workspace
python evaluation/eval_calvin.py \
    --args.exp_name calvin_smoke \
    --args.debug \
    --args.debug_steps 200 \
    --args.scene calvin_scene_D
```

Expected output:

```
CALVIN debug smoke test (scene=calvin_scene_D, 200 steps)
...
EGL device choice: -1 of 1.
GL_RENDERER=llvmpipe (LLVM ...)   # software rendering — see below
...
Videos saved to: <save_videos_root>/<exp_name>/debug/
```

If you see `Renderer = llvmpipe` you are using **software (CPU) rendering**. That's normal in pods without graphics driver caps and is slow but functional. To get hardware EGL (NVIDIA GPU rendering) you need the pod to expose `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` — ask your cluster admin.

## 5. Full eval (with policy)

```bash
python evaluation/eval_calvin.py \
    --args.exp_name calvin_rel_s10000 \
    --args.pretrained_model_path /path/to/checkpoint-XXXX \
    --args.calvin_models_root $CALVIN_ROOT/calvin_models \
    --args.scene calvin_scene_D \
    --args.num_sequences 1000 \
    --args.n_workers 1
```

Notes:
- `--args.n_workers 1` runs in-process (lowest RAM). Raise it only if your pod has enough RAM for `N × (policy + env)` simultaneously.
- The official LH-MTLC protocol uses `num_sequences=1000`. Use smaller numbers (e.g. 5–100) for quick iteration; the result is noisy below ~100.
- Videos are saved every `--args.save_video_every` sequence (default 25).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Killed` right after `Renderer = llvmpipe` | Cgroup OOM during `from_pretrained` | Request more RAM (`--mem=64G` in Slurm), or use `n_workers=1` |
| `failed to EGL with glad` | Forced NVIDIA EGL but pod lacks graphics caps | `unset PYOPENGL_PLATFORM __EGL_VENDOR_LIBRARY_FILENAMES EGL_DEVICE_ID` |
| Workers `terminated abruptly` | Parent OOM-killed before children init | Use `--args.n_workers 1` |
| `ModuleNotFoundError: calvin_env` | `sys.path` mismatch | Update the four `sys.path.insert` lines at top of `eval_calvin.py` |
