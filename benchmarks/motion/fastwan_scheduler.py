# SPDX-License-Identifier: Apache-2.0
"""DMD sampling adapter for Diffusers' image-conditioned Wan 5B pipeline.

Follows FastVideo's clean-prediction / re-noising equations and training-noise
lookup at commit a01c50a8063386895c6d8895a1e2509d11d55284:
  fastvideo/pipelines/basic/wan/stages/dmd.py
  fastvideo/models/utils.py (pred_noise_to_pred_video)
  fastvideo/models/wan/definition.py (DMD_TRAINING_NOISE_SHIFT)
This is a benchmark adapter, not FastVideo's native runtime or sparse attention.
"""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


class FastWanDMDScheduler(FlowMatchEulerDiscreteScheduler):
    ladder = (1000, 757, 522)

    def set_timesteps(self, num_inference_steps=3, device=None, **kwargs):
        if num_inference_steps != len(self.ladder):
            raise ValueError("This checkpoint is tested only with its trained three-step ladder.")
        training = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=8.0)
        targets = torch.tensor(self.ladder, dtype=torch.float32)
        indices = (training.timesteps[:, None] - targets[None, :]).abs().argmin(dim=0)
        self.sigmas = training.sigmas[indices].to(device)
        self.timesteps = targets.to(device)
        self.num_inference_steps = num_inference_steps
        self._step_index = 0

    def step(self, model_output, timestep, sample, return_dict=False, **kwargs):
        index = self._step_index
        sigma = self.sigmas[index].double()
        clean = (sample.double() - sigma * model_output.double()).to(model_output.dtype)
        if index < len(self.ladder) - 1:
            noise = randn_tensor(clean.shape, generator=self.generator,
                                 device=clean.device, dtype=clean.dtype)
            next_sigma = self.sigmas[index + 1]
            clean = ((1 - next_sigma) * clean + next_sigma * noise).to(noise.dtype)
        self._step_index += 1
        return (clean,)
