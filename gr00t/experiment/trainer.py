# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import os
from typing import Optional

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        super().__init__(**kwargs)
        # Allowlist numpy globals for safe RNG state unpickling in PyTorch 2.1+
        torch.serialization.add_safe_globals(
            [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to apply _set_static_graph() once before the first backward.

        The equivariant backbone runs the same LLM module N times per forward pass
        (once per rotation group element). DDP fires a gradient-ready hook once per
        parameter per backward traversal; with N traversals through the same parameters
        the hook fires N times and raises 'marked ready twice'.

        _set_static_graph() tells DDP the computation graph is fixed and parameters
        may be marked ready more than once per step. Applied lazily here because
        `model` in training_step is the actual DDP-wrapped model (Accelerate may
        delay wrapping beyond _wrap_model).
        """
        if not getattr(self, "_static_graph_set", False):
            # Walk through Accelerate / DDP wrappers to find the DDP instance
            candidate = model
            while candidate is not None:
                if hasattr(candidate, "_set_static_graph"):
                    candidate._set_static_graph()
                    break
                candidate = getattr(candidate, "module", None)
            self._static_graph_set = True
        return super().training_step(model, inputs, num_items_in_batch)

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        self.model.eval()
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()
        self.model.train()
        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
