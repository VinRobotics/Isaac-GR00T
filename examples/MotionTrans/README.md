# MotionTrans human + robot co-training

Fine-tune GR00T N1.7 on a mixture of **human egocentric VR demonstrations** and **robot
teleop demonstrations** exported by the MotionTrans pipeline (LeRobot format with
`observation.state` / `action` / `observation.camera0_pose`, see
`gr00t_lerobot_dataset.md` in the motiontrans-pi0 repo).

## How it works

1. **Shared action space.** Human and robot data are retargeted into the same EEF action
   space (`[pos3, rot6d6, gripper]` per arm) and trained under **one embodiment tag**, so
   they share the action head — that is what transfers skills from human to robot data.
2. **Relative EEF actions.** GR00T's relative EEF representation
   (`inv(T_state) @ T_action`) is frame-invariant, so the moving head camera cannot
   corrupt the action labels.
3. **Camera reprojection for the state.** Human data stores absolute EEF poses in the
   *episode-start* camera frame while the head camera moves — so the raw state does not
   match the image at time `t`. The `camera_pose` group (from
   `observation.camera0_pose`) triggers an on-the-fly reprojection of the EEF
   state/action into the camera frame at `t`
   (`gr00t/data/state_action/camera_projection.py`). The projection cancels exactly in
   the relative action, so `relative_stats.json` stays valid; the model's state input
   becomes consistent with the image. Robot datasets (static camera) have no
   `camera_pose` group and are loaded unchanged.

## 1. Prepare the datasets

GR00T needs the action EEF block to match the state layout (`[pos3, rot6d6, gripper]`
per arm) plus a `meta/modality.json`. Current MotionTrans exports already store actions
in rot6d, so preparation is just an in-place annotation:

```bash
python scripts/convert_motiontrans_to_gr00t.py --input-path /data/human_stack_cup
python scripts/convert_motiontrans_to_gr00t.py --input-path /data/robot_stack_cup
```

This writes `meta/modality.json` (including the `camera_pose` group when
`observation.camera0_pose` exists). For older exports whose `action` column is
`[pos3, rotvec3, gripper]`, pass `--output-path <dir>` and the script converts the
action column to rot6d into a new directory (videos symlinked; `--copy-videos` to copy).
`meta/stats.json` and `meta/relative_stats.json` are generated automatically at training
start.

Emitted groups (bimanual, gripper dim 1 — e.g. `human_stack_cup`):

| Group | state slice | action slice |
|---|---|---|
| `right_eef_9d` | `[0:9]` | `[0:9]` |
| `right_gripper` | `[9:10]` | `[9:10]` |
| `left_eef_9d` | `[10:19]` | `[10:19]` |
| `left_gripper` | `[19:20]` | `[19:20]` |
| `camera_pose` (human only) | `observation.camera0_pose [0:6]` | — |

For single-arm datasets the converter emits un-prefixed `eef_9d` / `gripper` groups;
adjust `motiontrans_config.py` accordingly.

## 2. Co-train

```bash
bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/human_stack_cup \
    --dataset-path /data/robot_stack_cup \
    --modality-config-path examples/MotionTrans/motiontrans_config.py \
    --embodiment-tag new_embodiment \
    --output-dir ./outputs/motiontrans_cotrain \
    -- --dataset-mix-ratio 0.5 0.5
```

### Balancing human vs robot data (α)

Two independent knobs, both parallel to the `--dataset-path` list:

- `--dataset-mix-ratio r1 r2` — **sampling** ratio: how often batches draw from each
  dataset, independent of dataset sizes. Omit for equal sampling.
- `--dataset-loss-weight w1 w2` — **loss** ratio: per-sample multiplier applied in the
  action loss (weighted mean, so uniform weights are exactly a no-op). This is
  MotionTrans's α: their pi0 co-training samples uniformly over the concatenated frames
  and re-weights purely in the loss (`loss * alpha` in `pi0.py`), with α renormalized by
  the human/robot frame counts so the *total* gradient contribution is (1−α) : α.

To reproduce MotionTrans α = 0.7 (robot-dominant) here: keep mix ratios equal and set

```bash
--dataset-loss-weight 0.3 0.7   # human first, robot second
```

Total gradient contribution is then `mix_ratio × loss_weight` per dataset →
0.5·0.3 : 0.5·0.7 = 0.3 : 0.7, matching MotionTrans's count-renormalized α exactly.
Alternatively you can skip loss weighting and put the imbalance in `--dataset-mix-ratio
0.3 0.7` instead — same expected gradient ratio, but via sampling frequency (robot
samples seen more often) rather than per-sample scaling.

## Notes & caveats

- **`observation.state` min/max stats** are computed from the *unprojected* values
  (episode-start frame). The reprojected values live in the same numeric range, so
  min/max normalization stays well-behaved, but the stats are not exact for human data.
- **Inference on the robot** needs no camera pose: the robot camera is static, the
  `camera_pose` group is absent, and states pass through unchanged. Serve with the same
  modality config file and embodiment tag.
- **Mixing single-arm and bimanual data** in one parquet schema is not supported —
  zero-pad the missing arm at export time (same constraint as the MotionTrans LeRobot
  pipeline).
