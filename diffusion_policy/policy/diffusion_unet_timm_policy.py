from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def add_remove_noise(self, batch, noise_level):
        # this function is an implementation of checking if something is in or out of distribution
        # it adds the specified noise to the batch, then denoises it again and returns the denoised actions 
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # TODO: repeat samples by tiling 
        # HACK : features missing
        #   1) local conditioning 

        # should return action similarity across X diffusion steps 
        # for now, hard code at []

        # # train on multiple diffusion samples per obs
        # if self.train_diffusion_n_samples != 1:
        #     # repeat obs features and actions multiple times along the batch dimension
        #     # each sample will later have a different noise sample, effecty training 
        #     # more diffusion steps per each obs encoder forward pass
        #     global_cond = torch.repeat_interleave(global_cond, 
        #         repeats=self.train_diffusion_n_samples, dim=0)
        #     nactions = torch.repeat_interleave(nactions, 
        #         repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # # input perturbation by adding additonal noise to alleviate exposure bias
        # # reference: https://github.com/forever208/DDPM-IP
        # noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        assert 0 <= noise_level < self.noise_scheduler.config.num_train_timesteps
        timesteps = noise_level * torch.ones((nactions.shape[0],), dtype = torch.long, device = trajectory.device)
        # timesteps = torch.randint(
        #     0, self.noise_scheduler.config.num_train_timesteps, 
        #     (nactions.shape[0],), device=trajectory.device
        # ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        
        # now, time to denoise! 

        scheduler = self.noise_scheduler
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        start_index = (scheduler.timesteps >= noise_level).nonzero()[-1][0]

        for t in scheduler.timesteps[start_index:]: # this goes from high to low 
            # 1. apply conditioning

            # 2. predict model output
            model_output = self.model(noisy_trajectory, t, 
                local_cond=None, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            noisy_trajectory = scheduler.step(
                model_output, t, noisy_trajectory, 
                generator=None,
                **self.kwargs
                ).prev_sample
        
        return noisy_trajectory 

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_obs_representations(self, batch):
        # this function will return a list of intermediate representation
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        
        global_cond = self.obs_encoder(nobs)
        return global_cond 
    

    def compute_loss(self, batch, return_intermediates = False,  parallel_intermediates = None, reduce_loss = True):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images

        if parallel_intermediates is None: 
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            # input perturbation by adding additonal noise to alleviate exposure bias
            # reference: https://github.com/forever208/DDPM-IP
            noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (nactions.shape[0],), device=trajectory.device
            ).long()
        else:
            # this will link the u-net training of another u-net forward pass
            # critical that we link them or else the noise will create a lot of variance in the training 
            noise_new = parallel_intermediates["noise_new"]
            noise = parallel_intermediates["noise"]
            timesteps = parallel_intermediates["step"]

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond,
            return_intermediates = return_intermediates
        )
        if return_intermediates:
            pred, representations = pred # unpack additional value 

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        if reduce_loss:
            loss = loss.mean()

        if return_intermediates:
            intermediates = {"u_net_representations" : representations, "global_cond": global_cond, "noise_new" : noise_new, "noise" : noise, "step" : timesteps, "pred" : pred}
            return loss, intermediates # allows for additional calculations on the intermediate representations 
            
        return loss



    def forward(self, batch, return_intermediates = False, parallel_intermediates = None, reduce_loss = True):
        # return intermediates: return intermediate representations (might be customized later)
        # noise: supplied noise seed for parallel runs (needed for u-net regularization)
        # reduce_loss: either return a mean loss or return a vector of losses, useful for parsing losses 
        return self.compute_loss(batch, return_intermediates = return_intermediates,  parallel_intermediates =  parallel_intermediates, reduce_loss = reduce_loss)