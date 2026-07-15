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

`keypoint_head_mode` (`--keypoint-head-mode`) selects how the head attaches. In every
mode, the model receives **no real keypoint data as input, ever** — at train or
inference time. `"default"`/`"tokens"` go further and are a strict **pure readout**:
`action_pred` is architecturally provably unaffected by whether the head runs at all.
`"share_dim"` is *not* a pure readout in that stricter sense (see its bullet below) —
only the "no real data fed in" guarantee holds for it.

- **`"default"`** (default): the DiT sequence is `[state_token, action_tokens(40)]`;
  keypoints for step `t` are decoded from the *same* hidden state that decodes action
  `t` (first 16 action tokens) — no extra tokens, no attention changes, no online
  tracker needed at deployment.
- **`"tokens"`**: appends `keypoint_horizon` dedicated learned query tokens
  (DETR-style) to the DiT sequence —
  `[state_token, action_tokens(40), keypoint_queries(16)]` — decoded from those
  instead, giving the keypoint task its own capacity through the transformer's
  self-/cross-attention rather than squeezing it into the action tokens' hidden
  state. A one-directional self-attention mask lets keypoint query tokens attend to
  the state/action tokens (so keypoint prediction stays conditioned on the specific
  action being generated), while masking state/action tokens from ever attending back
  to the keypoint queries, at every layer — so `action_pred` is bit-identical whether
  or not the head runs, same guarantee as `"default"`. Adds a
  `keypoint_query_embedding` parameter, so this setting must match between saving and
  loading a checkpoint (see Usage).
- **`"share_dim"`**: folds the *active flags only* (not point positions — see Loss
  below for why) as extra channels of the same per-step action vector, jointly
  noised/denoised by flow matching, exactly like this codebase's existing
  `expand_action_dimension` mechanism for other per-embodiment action-space widening
  (`embodiment_conditioned_mlp.py`). Widens `action_encoder`'s input dim and
  `action_decoder`'s output dim by `max_keypoint_objects`. **Not a pure readout**:
  `action_encoder` mixes every input channel into one embedding through a dense
  matrix, so the (noised) active-flag values genuinely influence the shared
  representation that also produces the real action prediction during training — the
  same property this codebase's existing effort/torque-aware channels already have.
  `action_pred`'s *returned tensor* is still sliced back down to the real action width
  everywhere it's produced (so it never leaks the extra channels), and the Euler
  rollout's active channels always start from pure noise like the real action
  channels (no real keypoint data fed in at inference either) — but its *value*, given
  the same action targets, can differ from a `"default"`/`"tokens"`/disabled run,
  because the encoder saw a wider joint input during training. Also must match
  between saving and loading a checkpoint (see Usage).

In `"default"`/`"tokens"` mode, two independent decoder heads with no shared
parameters — `keypoint_position_decoder` (Chamfer position regression) and
`keypoint_active_decoder` (active-flag BCE) — read the same hidden states, rather than
two output columns of one shared trunk. A shared trunk was tried first and regressed:
`static_keypoint_weight=0` hard-masks the position loss to active steps only (~30% of
cells) while the active BCE trains on every step, and `keypoint_loss_weight` (1.0)
dominates `keypoint_active_loss_weight` (0.1) — the shared hidden layer's gradient was
increasingly dominated by the frequent, high-weight position signal, so
`eval_keypoint_active_loss` degraded over training even as position accuracy kept
improving. Separate trunks remove that interference regardless of loss-weight tuning.
`"share_dim"` mode sidesteps the interference differently: active isn't decoded by a
trunk at all, it's a flow-matching regression target folded into `action_decoder`, so
there's no shared trunk to interfere in the first place.

**Output:** per step `t ∈ [0, 16)`: absolute keypoint positions `[2, 20, 2]` in
normalized image coordinates (all modes, via `keypoint_position_decoder`) plus one
active value per object slot — a BCE probability in `"default"`/`"tokens"` mode, or the
raw denoised flow-matching regression value (already ~`[0, 1]`-ranged, no sigmoid) in
`"share_dim"` mode.

### Loss

1. **Point level (permutation-invariant):** symmetric Chamfer distance (Huber-based)
   between the 20 predicted and 20 ground-truth points, per object per step. Point
   *index* within a slot carries no meaning (farthest-point sampling assigns no fixed
   identity to point k), so this stays invariant to point order.
2. **Slot level (fixed, not matched):** predicted slot 0/1 is compared directly to GT
   slot 0/1, no permutation search. The data pipeline's `assign_slots` (see
   `test_keypoint_tracking.py` / `convert_keypoint_tracking.py`) assigns slot identity
   by first-appearance frame then left-to-right centroid x, held fixed for the whole
   episode — a stable, image-visible convention the model can learn via cross-attention
   to the current frame, rather than a truly arbitrary label. Per-sample permutation
   matching was tried first but lets the model dodge learning that convention, and its
   hard argmin flips near-symmetric cases, giving each slot an inconsistent training
   target across steps.
3. **Active mask:** keypoint *position* loss only counts steps with `active == 1` —
   inactive slots can hold zeros (absent object), frozen positions (before the first
   mask sighting) or untrusted extrapolations, indistinguishable from a truly static
   object (`--static-keypoint-weight > 0` re-enables down-weighted supervision on
   them).

Why point positions never fold into `"share_dim"`'s joint flow-matching, unlike active
flags: object-slot index (0/1) *is* fixed by `assign_slots`, so plain index-matched
regression is well posed for it. Point index within a slot has no such convention —
folding it into flow matching the same way would train against an arbitrary
per-episode label the model cannot actually predict from the image, which would push
predictions toward a blurry per-point average rather than a real position. Chamfer
sidesteps that by being invariant to point order in the first place.

`total = flow_matching + keypoint_loss_weight · chamfer + keypoint_active_loss_weight · active_loss`,
with the per-sample α co-training `loss_weight` applied the same way as in the action
loss. `active_loss` is BCE in `"default"`/`"tokens"` mode (supervised on every step) or
masked flow-matching MSE in `"share_dim"` mode (supervised on steps `t < 16`, masked by
`has_keypoint`, same as the action loss). `keypoint_loss` / `keypoint_active_loss` are
logged separately by the trainer in all three modes.

### Usage

```bash
bash examples/finetune.sh ... \
    -- --enable-keypoint-head \
       --keypoint-loss-weight 1.0 \
       --keypoint-active-loss-weight 0.1
```

Loading a base checkpoint without the head works: the missing `keypoint_position_decoder`
/ `keypoint_active_decoder` (and `keypoint_query_embedding`, in `"tokens"` mode)
tensors are freshly initialized (logged, not an error). A checkpoint saved with the old
combined `keypoint_decoder` (single shared trunk) also loads fine — those weights are
discarded and the two new heads start fresh, logged as an expected architecture
migration, not an error. Loading a checkpoint that WAS trained with a *different*
`--keypoint-head-mode` is only safe in one direction: switching *into* `"share_dim"`
from a checkpoint trained without it is handled automatically (the widened
`action_encoder`/`action_decoder` tensors are spliced from the checkpoint's narrower
ones, logged as an expected migration); any other mode change (e.g. `"default"` <->
`"tokens"`, or *out of* `"share_dim"`) is treated as an architecture mismatch
(`unexpected_keys` / `mismatched_keys`) and fails loudly rather than silently dropping
trained weights.

```bash
bash examples/finetune.sh ... \
    -- --enable-keypoint-head --keypoint-head-mode tokens
```

For debugging/eval, decode the predicted keypoint trajectories on any rollout (no extra
input needed) and overlay them on the frame to check whether the model understands
object motion:

```python
out = policy.model.action_head.get_action(backbone_output, action_input,
                                          options={"return_keypoints": True})
out["keypoint_pred"]         # [B, 16, 2, 20, 2], [-1, 1] normalized image coords
out["keypoint_active_pred"]  # [B, 16, 2] probabilities ("default"/"tokens"), or raw
                              # ~[0, 1] flow-matching regression values ("share_dim")
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
  `keypoint_active_loss_weight`, `static_keypoint_weight`, `keypoint_head_mode`.

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
