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

## 3. Object-centric keypoint auxiliary head (optional)

Object motion is embodiment-invariant: the same cup slides the same way whether a human
hand or a robot gripper pushes it. The keypoint head exploits that to close the
human↔robot gap — the **same hidden states that decode actions** are additionally asked
to predict the future motion of tracked object keypoints, through a decoder **shared
across embodiments**. A human sample and a robot sample producing the same object motion
are thereby pulled toward the same action-token representation.

### Data format

Produced by the keypoint-tracking pipeline (SAM3 + CoTracker3, see
`convert_keypoint_tracking.py` in the lerobot-convert repo) as two extra per-frame
columns:

| Column | Shape | Meaning |
|---|---|---|
| `observation.keypoint_2d` | `[40, 2]` | pixel `(x, y)` of 20 tracked points × 2 object slots (slot `s` owns rows `[s*20:(s+1)*20)`) |
| `observation.keypoint_active` | `[2]` | 1.0 when the slot's windowed motion exceeds the gate **and** is near a SAM3 mask sighting |

`scripts/convert_motiontrans_to_gr00t.py` emits the `keypoint` section of
`meta/modality.json` automatically when the columns exist. The loader normalizes pixel
coordinates to `[-1, 1]` with the camera resolution from `meta/info.json` (datasets
tracked with `--normalize` are detected by scale and not re-normalized). Datasets
*without* keypoint columns co-train unchanged: their samples carry `has_keypoint = 0`
and contribute nothing to the auxiliary loss.

### Architecture

- **Attach point:** the DiT sequence is `[state_token, action_tokens(40)]`; keypoints
  for step `t` are decoded from the *same* hidden state that decodes action `t` (first
  16 action tokens), via a plain shared MLP (`keypoint_decoder`, not embodiment-indexed).
- **Pure readout:** the model receives **no keypoint input** anywhere — no extra tokens,
  no attention changes. Disabling or removing the head leaves the action path
  bit-identical, and deployment needs no online tracker.
- **Output:** per step `t ∈ [0, 16)`: absolute keypoint positions `[2, 20, 2]` in
  normalized image coordinates plus one active logit per object slot.

### Loss (set matching)

Without keypoint input the model cannot know which points the tracker sampled nor which
object sits in which slot, so the loss is invariant to both:

1. **Point level:** symmetric Chamfer distance (Huber-based) between the 20 predicted
   and 20 ground-truth points, per object per step.
2. **Slot level:** cost is computed for both object-slot permutations and the minimum is
   taken, one consistent assignment per 16-step chunk.
3. **Active mask:** keypoint loss only counts steps with `active == 1` — inactive slots
   can hold zeros (absent object), frozen positions (before the first mask sighting) or
   untrusted extrapolations, indistinguishable from a truly static object
   (`--static-keypoint-weight > 0` re-enables down-weighted supervision on them). The
   active-flag BCE is supervised on every step.

`total = flow_matching + keypoint_loss_weight · chamfer + keypoint_active_loss_weight · bce`,
with the per-sample α co-training `loss_weight` applied the same way as in the action
loss. `keypoint_loss` / `keypoint_active_loss` are logged separately by the trainer.

### Usage

```bash
bash examples/finetune.sh ... \
    -- --enable-keypoint-head \
       --keypoint-loss-weight 1.0 \
       --keypoint-active-loss-weight 0.1
```

Loading a base checkpoint without the head works: the missing `keypoint_decoder`
tensors are freshly initialized (logged, not an error).

For debugging/eval, decode the predicted keypoint trajectories on any rollout (no extra
input needed) and overlay them on the frame to check whether the model understands
object motion:

```python
out = policy.model.action_head.get_action(backbone_output, action_input,
                                          options={"return_keypoints": True})
out["keypoint_pred"]         # [B, 16, 2, 20, 2], [-1, 1] normalized image coords
out["keypoint_active_pred"]  # [B, 16, 2] probabilities
```

### Caveats

- Training random crop (`crop_fraction=0.95`) shifts the visible frame up to ~5% against
  the target coordinate frame; acceptable for representation shaping. If needed later,
  thread the keypoints through the albumentations call (`FractionalRandomCrop` already
  implements `apply_to_keypoints`).
- Predictions are point *sets* per step (Chamfer has no point identity across time);
  fine for checking object-motion understanding, use a one-shot Hungarian match if
  smooth per-point trajectories are needed.
- Config knobs live in `Gr00tN1d7Config`: `enable_keypoint_head`, `keypoint_horizon=16`,
  `max_keypoint_objects=2`, `keypoints_per_object=20`, `keypoint_loss_weight`,
  `keypoint_active_loss_weight`, `static_keypoint_weight`.

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
