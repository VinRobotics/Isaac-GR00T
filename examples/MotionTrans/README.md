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

## 3. VLM-backbone motion-keypoint head + human/robot OT alignment (optional)

Object motion is embodiment-invariant: the same cup slides the same way whether a human
hand or a robot gripper pushes it. `num_motion_tokens` learnable query tokens pass
through the **VLM backbone's own transformer layers** (alongside the image/language
tokens — not the DiT/action head) and are trained to predict the future motion of
tracked object keypoints, on **both human and robot samples**. The pooled motion-token
embedding then decides, via an **Optimal Transport (Sinkhorn) alignment loss**, WHICH
human and robot samples in a batch correspond — and that correspondence is used to pull
the general **backbone representation** (the one actually feeding the DiT/action head)
of matched pairs together, since that's the representation that matters for transfer.
See `method_motion_keypoint_ot_v2.md` at the repo root for the full design rationale,
and `gr00t/model/gr00t_n1d7/motion_head.py` / `gr00t/model/modules/optimal_transport.py`
/ `gr00t/model/modules/qwen3_backbone.py` for the implementation.

### Data format

Produced by a keypoint-tracking pipeline (e.g. SAM3 + a point tracker) as two extra
per-frame columns:

| Column | Shape | Meaning |
|---|---|---|
| `observation.keypoint_2d` | `[N*K, 2]` | pixel `(x, y)` of `K` tracked points × `N` object slots |
| `observation.keypoint_valid` | `[N]` | 1.0 when the slot has a real tracked object at the init frame, 0.0 for a padding slot |

`scripts/convert_motiontrans_to_gr00t.py` emits the `keypoint` section of
`meta/modality.json` automatically when the columns exist. The loader normalizes pixel
coordinates to `[-1, 1]` with the camera resolution from `meta/info.json`. Datasets
*without* keypoint columns co-train unchanged: their samples carry `has_keypoint = 0`
and contribute nothing to the motion loss or OT alignment.

### Architecture

The motion tokens are a **pure readout with respect to the action path**: they live
entirely inside the VLM backbone (appended at the end of its input sequence, so causal
attention lets them attend to every real image/language token while nothing attends
back to them — the same guarantee gained for free from ordinary causal masking), and
the DiT/action head never sees them. One feedforward pass through the backbone predicts
each selected point's whole future trajectory directly — no flow-matching rollout is
needed to read out keypoints, unlike the retired Action-Head keypoint design.

Point identity is bound at the decoder via each point's t=0 anchor position (an MLP
over `keypoint_target`'s first step, concatenated with the token's post-backbone
hidden state) rather than the query embedding itself — this is what keeps the
by-index Huber loss well posed without any Hungarian/Chamfer matching (see
`gr00t/model/gr00t_n1d7/motion_head.md` for the point-identity-collapse
analysis this design is built on). `num_motion_tokens` selects the top-K
MOST-MOVING valid points of the window per training sample (deterministic per window,
so eval videos are stable); `None` uses the full flat point set.

OT alignment uses TWO distinct pooled feature spaces, computed by the **Trainer**, not
the model (human vs. robot grouping uses the existing per-sample `is_human` signal,
deliberately never routed into the model's forward pass):

- **Matching space** = pooled motion-token features (`mean` over the token axis, by
  default) — an embodiment-invariant "what motion is this" signal, trained only by
  `motion_loss`. Used to compute the Sinkhorn transport plan `P*`.
- **Alignment target** = pooled backbone features (`backbone_pooled_features` — the
  vlln/vl_self_attention-refined representation that actually feeds the DiT via
  cross-attention) — what actually gets pulled together according to `P*`. Motion
  decides *which* robot/human samples correspond; the pull is applied to the
  representation that matters for transfer.

The transport plan is **detached** before computing the alignment loss
(`stopgrad_plan`, default on in `sinkhorn_ot_loss`), so this never leaks an
alignment-driven gradient back into the motion-token encoder — it stays trained
purely by `motion_loss`. `ot_align_weight` ramps in linearly over `ot_warmup_steps` so
alignment pressure doesn't fight the motion head before its embeddings carry any
signal.

### Loss

`total = flow_matching + motion_loss_weight · motion_loss [+ ot_lambda(step) · ot_loss]`,
with the per-sample α co-training `loss_weight` folded into `motion_loss` the same way
as the action loss. `motion_loss` is a masked Huber position-regression loss against
`keypoint_target`/`keypoint_active_target` (`motion_static_weight` re-enables
down-weighted supervision on padding slots, default 0 = hard-masked). `ot_loss` is the
Sinkhorn entropic-OT cost, matched on pooled motion features and applied to pooled
backbone features (`ot_sinkhorn_eps`/`ot_sinkhorn_iters` control the solver). Both are
logged separately by the trainer.

### Usage

```bash
bash examples/finetune.sh ... \
    -- --enable-motion-head \
       --num-motion-tokens 8 \
       --motion-loss-weight 1.0 \
       --enable-ot-align \
       --ot-align-weight 1.0 \
       --ot-warmup-steps 1000
```

`enable_ot_align` is independent of `enable_motion_head`: train the motion head alone
(no alignment pressure) as an ablation by omitting `--enable-ot-align`. Loading a base
checkpoint without the head works: the missing `motion_query_tokens` (backbone) /
`motion_coord_encoder` / `motion_position_decoder` (`MotionHead`) tensors are freshly
initialized (logged, not an error). A checkpoint saved under the old, retired
Action-Head keypoint mechanism also loads fine — those `action_head.keypoint_*`
weights are discarded (logged as an expected migration, not an error).

For debugging/eval, the trainer automatically logs GT-vs-predicted keypoint overlay
images/videos to W&B (`{prefix}/motion`, `{prefix}/motion_episode_video/...`) plus,
whenever the motion head is enabled and eval is configured (`validation_path` or
`eval_set_split_ratio`) — see `gr00t/experiment/trainer.py`'s
`_log_motion_viz`/`_log_domain_alignment_viz`:

- A t-SNE scatter of pooled robot-vs-human **backbone** features (the actual alignment
  target) plus KNN domain-composition/variance scalars
  (`{prefix}/backbone_domain_tsne`, `{prefix}/backbone_domain_knn_same_domain_frac`,
  `{prefix}/backbone_domain_robot_variance`, `{prefix}/backbone_domain_human_variance`)
  — this is what answers "are robot/human representations actually being pulled
  together", not the loss curve alone.
- A motion-vs-backbone neighbor-structure agreement scalar
  (`{prefix}/motion_backbone_neighbor_agreement`) — the cross-feature design assumes
  correspondence found in motion-space is a reasonable correspondence to impose on
  backbone-space; this isn't automatic, and a low value means that assumption doesn't
  hold for this checkpoint/data even if the OT loss number looks fine.

To decode keypoints and inspect both pooled feature spaces directly from a forward
pass:

```python
out = policy.model(inputs=batch)
out["motion_pred"]  # [B, motion_horizon, P, 1, 2], [-1, 1] normalized image coords.
                     # Weight overlays with the GT valid mask (keypoint_active_target).
out["motion_pooled_features"]    # [B, backbone_embedding_dim], the OT MATCHING space.
out["backbone_pooled_features"]  # [B, backbone_embedding_dim], the OT alignment TARGET.
```

### Caveats

- Training random crop (`crop_fraction=0.95`) shifts the visible frame up to ~5% against
  the target coordinate frame; acceptable for representation shaping. If needed later,
  thread the keypoints through the albumentations call (`FractionalRandomCrop` already
  implements `apply_to_keypoints`).
- Config knobs live in `Gr00tN1d7Config`: `enable_motion_head`, `motion_horizon=16`,
  `max_motion_objects=2`, `motion_points_per_object=8`, `num_motion_tokens=8`,
  `motion_loss_weight`, `motion_static_weight`, `motion_relative`, `motion_pool`,
  `enable_ot_align`, `ot_align_weight`, `ot_warmup_steps`, `ot_sinkhorn_eps`,
  `ot_sinkhorn_iters`.

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
