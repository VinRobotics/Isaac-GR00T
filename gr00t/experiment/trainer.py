# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
import wandb

from gr00t.model.gr00t_n1d7.keypoint_viz import combine_gt_pred, render_keypoint_overlay


def _action_loss_by_group(
    action_loss: torch.Tensor, action_mask: torch.Tensor, is_human: torch.Tensor
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Masked-mean (numerator, denominator) pairs for the action loss, split by
    human/robot origin. Logging only — human+robot co-training shares one
    embodiment tag (so embodiment_id can't distinguish them), and this never
    affects the actual loss used for backprop.

    Returns (num, den) rather than a ready ratio so callers can either take an
    immediate ratio (per-step train logging) or accumulate across many batches
    before dividing once (eval, to avoid averaging per-batch means unevenly).
    Only includes a group if it has any real (masked) contribution in this batch.
    """
    is_human = is_human.to(action_loss.dtype).view(-1, 1, 1)
    weights = {"human": is_human, "robot": 1.0 - is_human}
    result = {}
    for name, w in weights.items():
        den = (action_mask * w).sum()
        if den > 0:
            result[name] = ((action_loss * action_mask * w).sum(), den)
    return result


class _TqdmDataLoader:
    """Wraps a DataLoader to show a tqdm progress bar during iteration."""

    def __init__(self, dataloader, desc="Eval"):
        self._dl = dataloader
        self._desc = desc

    def __len__(self):
        return len(self._dl)

    def __iter__(self):
        return iter(tqdm(self._dl, desc=self._desc, leave=True))

    def __getattr__(self, name):
        return getattr(self._dl, name)


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:  # noqa: D401
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


# Global variables for batched evaluation metrics
_eval_accuracy_accumulated_correct = 0
_eval_accuracy_accumulated_total = 0


def compute_eval_accuracy(
    eval_pred: EvalPrediction, compute_result: bool, action_offset: Optional[int] = None
):
    logits = eval_pred.predictions[0]
    if action_offset is not None:
        logits = logits[..., action_offset:]
    preds = logits.argmax(axis=-1)
    labels = eval_pred.label_ids

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = ((preds == labels) & mask).sum()
    total = mask.sum()

    global _eval_accuracy_accumulated_correct, _eval_accuracy_accumulated_total
    _eval_accuracy_accumulated_correct += correct
    _eval_accuracy_accumulated_total += total

    if compute_result:
        accuracy = _eval_accuracy_accumulated_correct / max(_eval_accuracy_accumulated_total, 1)
        _eval_accuracy_accumulated_correct = 0
        _eval_accuracy_accumulated_total = 0
        return {"eval_accuracy": accuracy}
    else:
        return {}


class Gr00tTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – simple description above
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        self.keypoint_viz_max_images = kwargs.pop("keypoint_viz_max_images", 50)
        self.keypoint_video_episodes = kwargs.pop("keypoint_video_episodes", 1)
        self.keypoint_video_max_frames = kwargs.pop("keypoint_video_max_frames", 100)
        self.keypoint_video_batch_size = kwargs.pop("keypoint_video_batch_size", 8)
        self.keypoint_video_fps = kwargs.pop("keypoint_video_fps", 10)
        super().__init__(
            *args,
            **kwargs,
            # compute_metrics=partial(compute_eval_accuracy, action_offset=self.action_offset),
        )

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    def get_train_dataloader(self):  # noqa: D401
        """Return a iterable dataloader without skipping the data during resume, but reseed the dataset instead."""

        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        # Use persistent workers for sharded dataset if num_workers is greater than 0
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context can only be used with num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        unwrapped_model = model.module if hasattr(model, "module") else model

        total_loss = torch.tensor(0.0, device=self.args.device)
        total_steps = torch.tensor(0, device=self.args.device)
        # Pure flow-matching loss, separate from total_loss (which is action +
        # keypoint combined once the aux head is on).
        total_action_loss = torch.tensor(0.0, device=self.args.device)
        # Per-group (human/robot) action-loss breakdown, accumulated as (num, den)
        # pairs across the whole eval pass and divided once at the end (rather than
        # averaging per-batch means unevenly) — logging only, see _action_loss_by_group.
        group_action_num = {
            "human": torch.tensor(0.0, device=self.args.device),
            "robot": torch.tensor(0.0, device=self.args.device),
        }
        group_action_den = {
            "human": torch.tensor(0.0, device=self.args.device),
            "robot": torch.tensor(0.0, device=self.args.device),
        }

        # Object-centric keypoint aux head: aggregate its losses separately and, on
        # rank 0, reservoir-sample a handful of GT-vs-predicted keypoint overlay
        # images across the WHOLE eval pass to log to W&B (debug/eval only —
        # action_pred is unaffected either way). Reservoir sampling means each
        # evaluate() call independently draws a fresh random subset instead of
        # always showing the same first-N samples encountered.
        enable_keypoint_head = getattr(self.model.config, "enable_keypoint_head", False)
        total_kp_loss = torch.tensor(0.0, device=self.args.device)
        total_kp_active_loss = torch.tensor(0.0, device=self.args.device)
        viz_images_gt: list = []
        viz_images_pred: list = []
        viz_candidates_seen = 0

        for inputs in tqdm(dataloader, desc=description):
            inputs = self._prepare_inputs(inputs)
            with torch.inference_mode():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            total_loss += loss.detach()
            total_steps += 1

            if isinstance(outputs, dict) and "action_loss_scalar" in outputs:
                total_action_loss += outputs["action_loss_scalar"].detach()

            if enable_keypoint_head and isinstance(outputs, dict) and "keypoint_loss" in outputs:
                total_kp_loss += outputs["keypoint_loss"].detach()
                total_kp_active_loss += outputs["keypoint_active_loss"].detach()

            # The collator wraps the real batch as {"inputs": {...actual keys...}} so
            # that model(**inputs) resolves to forward(self, inputs=<batch>) — unwrap
            # here too, otherwise "viz_image"/"keypoint_target"/"is_human" are never
            # found (they live one level down, not on the outer dict compute_loss
            # receives).
            batch = inputs["inputs"]

            if isinstance(outputs, dict) and "action_loss" in outputs and "action_mask" in outputs:
                is_human_batch = batch.get("is_human")
                if is_human_batch is not None:
                    groups = _action_loss_by_group(
                        outputs["action_loss"].detach(),
                        outputs["action_mask"].detach(),
                        is_human_batch,
                    )
                    for name, (num, den) in groups.items():
                        group_action_num[name] += num
                        group_action_den[name] += den

            has_keypoint_batch = batch.get("has_keypoint")
            if (
                enable_keypoint_head
                and self.args.local_rank in (-1, 0)
                and "viz_image" in batch
                and has_keypoint_batch is not None
                and bool((has_keypoint_batch >= 0.5).any())
                and viz_candidates_seen < self.keypoint_viz_max_images * 20
            ):
                with torch.inference_mode():
                    # Reuse the backbone/state features compute_loss's forward pass
                    # already computed above instead of re-running the full VLM
                    # backbone a second time (that redundant rerun, once per eval
                    # batch with keypoint data, was the actual cause of eval blowing
                    # up training wall time). action_input is intentionally empty:
                    # get_action_with_features only reads it for the RTC "action" in
                    # action_input gate, which we don't want here (this is a real,
                    # non-teacher-forced rollout for visualization).
                    action_out = unwrapped_model.action_head.get_action_with_features(
                        backbone_features=outputs["backbone_features"],
                        state_features=outputs["state_features"],
                        embodiment_id=batch["embodiment_id"],
                        backbone_output=BatchFeature(
                            data={
                                "image_mask": outputs["image_mask"],
                                "backbone_attention_mask": outputs["backbone_attention_mask"],
                            }
                        ),
                        action_input=BatchFeature(data={}),
                        options={"return_keypoints": True},
                    )
                viz_candidates_seen = self._collect_keypoint_viz(
                    batch, action_out, viz_images_gt, viz_images_pred, viz_candidates_seen
                )

        # Single reduce at the end — avoids per-step all_gather deadlock
        # when ranks process different numbers of batches.
        if self.args.world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_action_loss, op=dist.ReduceOp.SUM)
            for name in ("human", "robot"):
                dist.all_reduce(group_action_num[name], op=dist.ReduceOp.SUM)
                dist.all_reduce(group_action_den[name], op=dist.ReduceOp.SUM)
            if enable_keypoint_head:
                dist.all_reduce(total_kp_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_kp_active_loss, op=dist.ReduceOp.SUM)

        denom = total_steps.clamp(min=1)
        mean_loss = (total_loss / denom).item()
        num_samples = int(total_steps.item()) * self.args.per_device_eval_batch_size

        metrics = {
            f"{metric_key_prefix}_loss": mean_loss,
            f"{metric_key_prefix}_action_loss": (total_action_loss / denom).item(),
        }
        for name in ("human", "robot"):
            if group_action_den[name] > 0:
                metrics[f"{metric_key_prefix}_action_loss_{name}"] = (
                    group_action_num[name] / group_action_den[name]
                ).item()
        if enable_keypoint_head:
            metrics[f"{metric_key_prefix}_keypoint_loss"] = (total_kp_loss / denom).item()
            metrics[f"{metric_key_prefix}_keypoint_active_loss"] = (
                total_kp_active_loss / denom
            ).item()

        if enable_keypoint_head and self.args.local_rank in (-1, 0):
            logging.info(
                f"Keypoint viz: reservoir-sampled {len(viz_images_gt)}/"
                f"{self.keypoint_viz_max_images} image pairs from {viz_candidates_seen} "
                f"candidates for {metric_key_prefix}."
            )
            if viz_images_gt:
                self._log_keypoint_viz(viz_images_gt, viz_images_pred, metric_key_prefix)
            if self.keypoint_video_episodes > 0 and self.keypoint_video_max_frames > 0:
                self._log_keypoint_episode_video(model, unwrapped_model, metric_key_prefix)

        return EvalLoopOutput(
            predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples
        )

    def _collect_keypoint_viz(
        self,
        inputs: dict,
        action_out: dict,
        gt_images: list,
        pred_images: list,
        num_seen: int,
    ) -> int:
        """Reservoir-sample (Algorithm R) up to keypoint_viz_max_images GT-vs-
        predicted overlay pairs, uniformly over every has_keypoint candidate seen
        across the eval pass so far — not just the first ones encountered, so a
        long eval set isn't dominated by whichever shard happens to load first.

        `inputs` carries the GT keypoint targets and the raw-frame thumbnail (see
        ShardedSingleStepDataset._render_viz_thumbnail); `action_out` carries the
        model's own flow-matching rollout keypoint prediction (get_action with
        options={"return_keypoints": True}). `num_seen` is the running candidate
        count from prior batches this eval pass; returns the updated count.
        """
        has_keypoint = inputs.get("has_keypoint")
        viz_image = inputs.get("viz_image")
        if has_keypoint is None or viz_image is None or "keypoint_pred" not in action_out:
            return num_seen

        kp_target = inputs["keypoint_target"]
        active_target = inputs["keypoint_active_target"]
        kp_pred = action_out["keypoint_pred"]
        active_pred = action_out["keypoint_active_pred"]
        num_objects = kp_pred.shape[2]
        kp_per_object = kp_pred.shape[3]

        for i in range(viz_image.shape[0]):
            if has_keypoint[i].item() < 0.5:
                continue
            img = viz_image[i].detach().cpu().numpy()
            gt_kp = (
                kp_target[i]
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1, num_objects, kp_per_object, 2)
            )
            gt_active = active_target[i].detach().float().cpu().numpy()
            pred_kp = kp_pred[i].detach().float().cpu().numpy()
            pred_active = active_pred[i].detach().float().cpu().numpy()
            gt_img = render_keypoint_overlay(img, gt_kp, gt_active)
            pred_img = render_keypoint_overlay(img, pred_kp, pred_active)

            if num_seen < self.keypoint_viz_max_images:
                gt_images.append(gt_img)
                pred_images.append(pred_img)
            else:
                j = random.randint(0, num_seen)
                if j < self.keypoint_viz_max_images:
                    gt_images[j] = gt_img
                    pred_images[j] = pred_img
            num_seen += 1
        return num_seen

    def _log_keypoint_viz(self, gt_images: list, pred_images: list, metric_key_prefix: str) -> None:
        if wandb.run is None:
            logging.info(
                f"Keypoint viz: {len(gt_images)} image pairs collected but wandb.run is None "
                "(use_wandb not set?) — skipping W&B image log."
            )
            return
        # One panel, one key: each entry is GT|Pred side by side for the same sample,
        # so the list's built-in index slider pages through matched pairs and the
        # run's step slider pages through eval calls.
        combined = [
            wandb.Image(combine_gt_pred(gt, pred), caption=f"sample {i}")
            for i, (gt, pred) in enumerate(zip(gt_images, pred_images))
        ]
        wandb.log({f"{metric_key_prefix}/keypoint": combined}, step=self.state.global_step)

    def _pick_keypoint_video_episodes(self) -> list:
        """Pick keypoint_video_episodes distinct held-out episodes PER dataset_path
        (every ShardedSingleStepDataset in the eval split — see eval_set_split_ratio
        / validation_path — that carries a "keypoint" modality), not
        keypoint_video_episodes total pooled across all of them: otherwise a
        multi-dataset mix could end up rendering every video from the same one
        dataset by chance, starving the rest of coverage. Sampling within each
        dataset is without replacement, capped to however many held-out episodes
        that dataset actually has.
        """
        eval_dataset = getattr(self, "eval_dataset", None)
        candidates = getattr(eval_dataset, "datasets", None) if eval_dataset is not None else None
        if not candidates:
            return []
        picks = []
        for ds in candidates:
            if "keypoint" not in getattr(ds, "modality_configs", {}):
                continue
            episode_indices = getattr(ds, "episode_indices", None) or []
            if not episode_indices:
                continue
            k = min(self.keypoint_video_episodes, len(episode_indices))
            picks.extend((ds, ep_idx) for ep_idx in random.sample(episode_indices, k))
        return picks

    def _render_keypoint_episode_video(
        self, model, unwrapped_model, dataset, ep_idx: int
    ) -> np.ndarray | None:
        """Roll the model's keypoint head forward over a held-out episode's frames
        (strided down to keypoint_video_max_frames) and render a GT-vs-predicted
        overlay video, the per-episode analogue of _collect_keypoint_viz's
        single-frame panels — see keypoint_viz.render_keypoint_overlay for the
        overlay.mp4-style visualization itself.

        Loads the full episode once (dense, so keypoint_target's forward-looking
        delta indices stay correct — see LeRobotEpisodeLoader / extract_step_data)
        and only spends model rollout compute on the strided subset of frames.
        """
        episode_data = dataset.episode_loader[ep_idx]
        effective_len = dataset.get_effective_episode_length(ep_idx)
        if effective_len <= 0:
            return None
        stride = max(1, effective_len // self.keypoint_video_max_frames)
        step_indices = list(range(0, effective_len, stride))[: self.keypoint_video_max_frames]

        frames = []
        for start in range(0, len(step_indices), self.keypoint_video_batch_size):
            chunk = step_indices[start : start + self.keypoint_video_batch_size]
            datapoints = [dataset.get_datapoint(episode_data, t) for t in chunk]
            batch = self.data_collator(datapoints)["inputs"]
            batch = self._prepare_inputs(batch)
            if "viz_image" not in batch or "keypoint_target" not in batch:
                return None
            # Deliberately NOT unwrapped_model.get_action(inputs=batch, ...): that
            # runs the VLM backbone directly, bypassing model.forward — the only
            # place Accelerate's bf16 handling (whether an autocast patch or
            # DeepSpeed's own precision management) actually gets applied — so any
            # backbone_trainable_params_fp32 layer produces real fp32 activations,
            # which flash-attn rejects outright. Instead, get backbone_features via
            # the WRAPPED model's forward (the exact call compute_loss already makes
            # successfully every eval batch), then hand them to
            # action_head.get_action_with_features for the actual rollout — same
            # split _collect_keypoint_viz above uses, and it never touches the
            # backbone a second time.
            with torch.inference_mode():
                outputs = model(inputs=batch)
            if not isinstance(outputs, dict) or "backbone_features" not in outputs:
                return None
            with torch.inference_mode():
                action_out = unwrapped_model.action_head.get_action_with_features(
                    backbone_features=outputs["backbone_features"],
                    state_features=outputs["state_features"],
                    embodiment_id=batch["embodiment_id"],
                    backbone_output=BatchFeature(
                        data={
                            "image_mask": outputs["image_mask"],
                            "backbone_attention_mask": outputs["backbone_attention_mask"],
                        }
                    ),
                    action_input=BatchFeature(data={}),
                    options={"return_keypoints": True},
                )
            if "keypoint_pred" not in action_out:
                return None
            num_objects = action_out["keypoint_pred"].shape[2]
            kp_per_object = action_out["keypoint_pred"].shape[3]
            for i in range(len(chunk)):
                img = batch["viz_image"][i].detach().cpu().numpy()
                gt_kp = (
                    batch["keypoint_target"][i]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1, num_objects, kp_per_object, 2)
                )
                gt_active = batch["keypoint_active_target"][i].detach().float().cpu().numpy()
                pred_kp = action_out["keypoint_pred"][i].detach().float().cpu().numpy()
                pred_active = action_out["keypoint_active_pred"][i].detach().float().cpu().numpy()
                gt_img = render_keypoint_overlay(img, gt_kp, gt_active)
                pred_img = render_keypoint_overlay(img, pred_kp, pred_active)
                frames.append(combine_gt_pred(gt_img, pred_img))

        if not frames:
            return None
        return np.stack(frames).transpose(0, 3, 1, 2)  # (T, C, H, W), RGB, for wandb.Video

    def _log_keypoint_episode_video(self, model, unwrapped_model, metric_key_prefix: str) -> None:
        if wandb.run is None:
            return
        episodes = self._pick_keypoint_video_episodes()
        if not episodes:
            return
        # One key per episode (rather than a list under one key, as with
        # wandb.Image): W&B's list-panel paging is Image-specific, Video isn't
        # guaranteed to render the same way. Key includes the dataset (basename of
        # dataset_path) since episodes are now picked per dataset_path — with a
        # multi-dataset mix there can be several videos per eval call.
        logs = {}
        for dataset, ep_idx in episodes:
            video = self._render_keypoint_episode_video(model, unwrapped_model, dataset, ep_idx)
            if video is not None:
                dataset_name = os.path.basename(str(getattr(dataset, "dataset_path", "dataset")))
                logs[f"{metric_key_prefix}/keypoint_episode_video/{dataset_name}/ep{ep_idx}"] = (
                    wandb.Video(
                        video,
                        fps=self.keypoint_video_fps,
                        format="mp4",
                        caption=f"{dataset_name} episode_{ep_idx}",
                    )
                )
        if not logs:
            logging.info(
                "Keypoint episode video: no held-out episode with keypoint data produced a "
                f"video for {metric_key_prefix}."
            )
            return
        wandb.log(logs, step=self.state.global_step)

    def get_eval_dataloader(self, eval_dataset=None):
        """Return a plain DataLoader for eval to avoid accelerate's NCCL broadcast on CPU tensors."""
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("No eval dataset provided")

        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        persistent_workers = self.args.dataloader_num_workers > 0
        dataloader_params = {
            "batch_size": self.args.per_device_eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                logging.warning(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            logging.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.
        """

        # Use parent implementation to preserve built-in functionality.
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # Record last loss for testing purposes.
        self.loss = loss

        # --------------------------------------------------------------
        # Loss component logging: pure action (flow-matching) loss, separate from
        # "loss" (which is action + keypoint combined once the aux head is on), plus
        # the keypoint aux head's own components.
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and isinstance(outputs, dict)
        ):
            log_values = {}
            if "action_loss_scalar" in outputs:
                log_values["action_loss"] = (
                    self._nested_gather(outputs["action_loss_scalar"].detach()).mean().item()
                )
            # Per-group (human/robot) action-loss breakdown — co-training runs share
            # one embodiment tag, so this can't come from embodiment_id.
            if "action_loss" in outputs and "action_mask" in outputs:
                is_human_batch = inputs["inputs"].get("is_human")
                if is_human_batch is not None:
                    groups = _action_loss_by_group(
                        outputs["action_loss"].detach(),
                        outputs["action_mask"].detach(),
                        is_human_batch,
                    )
                    for name, (num, den) in groups.items():
                        log_values[f"action_loss_{name}"] = (num / den).item()
            if "keypoint_loss" in outputs:
                log_values["keypoint_loss"] = (
                    self._nested_gather(outputs["keypoint_loss"].detach()).mean().item()
                )
                log_values["keypoint_active_loss"] = (
                    self._nested_gather(outputs["keypoint_active_loss"].detach()).mean().item()
                )
            if log_values and self.args.local_rank in (-1, 0):
                self.log(log_values)

        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            if self.args.local_rank in (-1, 0):
                self.log({"train_accuracy": acc_mean})

                # Log a sample of ground-truth vs predicted action tokens from
                # the first batch element so users can verify the model is
                # learning the right behaviors.
                shifted_labels = inputs["labels"][:1, 1:].cpu()
                shifted_preds = preds[:1, :-1]
                mask_0 = shifted_labels[0] != -100
                gt_tokens = shifted_labels[0][mask_0][:20]
                if self.action_offset is not None:
                    gt_tokens = gt_tokens - self.action_offset
                gt_sample = gt_tokens.tolist()
                pred_sample = shifted_preds[0][mask_0[: shifted_preds.shape[1]]][:20].tolist()
                logging.info(
                    "Step %d — GT vs Pred (first 20 action tokens, batch[0]):\n"
                    "  GT:   %s\n  Pred: %s",
                    self.state.global_step,
                    gt_sample,
                    pred_sample,
                )

        return (loss, outputs) if return_outputs else loss
