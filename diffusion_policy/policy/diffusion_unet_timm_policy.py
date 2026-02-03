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

# for DEBUG 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.set_printoptions(sci_mode=False)
# copied 
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def so3_distance(R1, R2):
    """
    Geodesic distance between two rotations.
    R1, R2: (..., 3, 3)
    returns: (...,)
    """
    R_rel = R1.transpose(-1, -2) @ R2
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta


# def se3_distance_weighted_sum(R1, t1, R2, t2, lambda_rot=1.0):
#     """
#     Simple weighted-sum SE(3) distance.
#     """
#     d_trans = torch.norm(t1 - t2, dim=-1)
#     d_rot = so3_distance(R1, R2)

#     return d_trans + lambda_rot * d_rot


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
        # NOTE: THIS is not the function used to compute OOD score; it is just a debugging tool. The real function is below. 
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

  

        trajectory = nactions
        # Sample noise that we'll add to the images

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # # input perturbation by adding additonal noise to alleviate exposure bias
        # # reference: https://github.com/forever208/DDPM-IP
        # noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        assert 0 <= noise_level < self.noise_scheduler.config.num_train_timesteps
        timesteps = noise_level * torch.ones((nactions.shape[0],), dtype = torch.long, device = trajectory.device)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        
        # now, time to denoise! 
        scheduler = self.noise_scheduler
        # set step value
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
    
    def _visualize_actions(self, noisy_trajectory_original, noisy_trajectory, noise, trajectory, INDEX_TO_VISUALIZE): 
        p_noisy_orig = noisy_trajectory_original[INDEX_TO_VISUALIZE, :, 0:3].detach().cpu().numpy()
        p_noisy = noisy_trajectory[INDEX_TO_VISUALIZE, :, 0:3].detach().cpu().numpy()
        p_noise = noise[INDEX_TO_VISUALIZE, :, 0:3].detach().cpu().numpy()
        orig_traj = trajectory[INDEX_TO_VISUALIZE, :, 0:3].detach().cpu().numpy() 
        # Stack them for axis limits
        points = np.concatenate([p_noisy_orig, p_noisy, p_noise, orig_traj], axis = 0)
        mins = points.min(axis=0) - 0.1
        maxs = points.max(axis=0) + 0.1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p_noisy_orig[:, 0], p_noisy_orig[:, 1], p_noisy_orig[:, 2], color='blue', label='original + noise')
        ax.scatter(p_noisy[:, 0], p_noisy[:, 1], p_noisy[:, 2], color='red', label='original + noise DENOISED')
        ax.scatter(p_noise[:, 0], p_noise[:, 1], p_noise[:, 2], color='green', label='noise')
        ax.scatter(orig_traj[:, 0], orig_traj[:, 1], orig_traj[:, 2], color='black', label='original')
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])
        ax.set_zlim([mins[2], maxs[2]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.tight_layout()

        # Render rotating video
        import os
        import imageio

        frames = []
        fps = 24
        n_frames = 72
        video_path = os.path.join("visuals/", 'rotating_3d.mp4')
        for i, angle in enumerate(np.linspace(0, 360, n_frames)):
            ax.view_init(elev=30, azim=angle)
            frame_path = os.path.join("visuals/", f"frame_{i:03d}.png")
            plt.savefig(frame_path)
            frames.append(imageio.imread(frame_path))
        # Save video
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Saved rotating 3d video to {video_path}")

        # Optionally, cleanup images
        for i in range(n_frames):
            try:
                os.remove(os.path.join("visuals/", f"frame_{i:03d}.png"))
            except Exception:
                pass

    def compute_ood_score(self, batch, noise_level, do_stop = False):
        # in: batch 
        # out: OOD score per batch element 
        # do_stop is a debugging technique 

        # this function is an implementation of checking if something is in or out of distribution
        # it adds the specified noise to the batch, then denoises it again and returns the denoised actions 
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # HACK : features missing
        #   1) local conditioning 

        trajectory = nactions
        # Sample noise that we'll add to the images

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random timestep for each image
        assert 0 <= noise_level < self.noise_scheduler.config.num_train_timesteps
        timesteps = noise_level * torch.ones((nactions.shape[0],), dtype = torch.long, device = trajectory.device)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        noisy_trajectory_original = noisy_trajectory.clone() # for graphing DEBUG
        # now, time to denoise! 

        scheduler = self.noise_scheduler
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        start_index = (scheduler.timesteps >= noise_level).nonzero()[-1][0]
        with torch.no_grad(): # so we won't explode the vram 
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
        

      
        noisy_trajectory_unnorm = self.normalizer['action'].unnormalize(noisy_trajectory).cpu()
        trajectory_unnorm = batch['action'].cpu()  # This is already unnormalized

        # N x T x D -> N X T X 3 -> N X 3T 
        assert len(noisy_trajectory_unnorm.shape) == 3, "the logic here is hard-coded to work for chunked actions"

        # this is taking the per-element difference in the chunk and then averaging it across time. 
        # more correct than taking the flattened norm 
        d_trans = torch.norm(noisy_trajectory_unnorm[..., 0:3] - trajectory_unnorm[..., 0:3], dim = -1)
        d_trans = torch.mean(d_trans, dim = 1)

        # d_trans = torch.mean(torch.square(noisy_trajectory_unnorm[..., 0:3].flatten(start_dim = 1) - trajectory_unnorm[..., 0:3].flatten(start_dim = 1)), dim = 1)
        # d_trans = torch.mean(torch.square(noisy_trajectory_unnorm.flatten(start_dim = 1) - trajectory_unnorm.flatten(start_dim = 1)), dim = 1)
        
        if do_stop:
            # DEBUGGING PURPOSE ONLY TO PROBE INSIDE THE OOD FUNCTION
            INDEX_TO_VISUALIZE = 0 
            plt.imsave("visuals/state.png", np.transpose(batch["obs"]["agentview_rgb"][INDEX_TO_VISUALIZE, 0].detach().cpu().numpy(), (1,2,0)))
            self._visualize_actions(noisy_trajectory_original, noisy_trajectory, noise, trajectory, INDEX_TO_VISUALIZE)
            import ipdb 
            ipdb.set_trace()
       
        N, T = noisy_trajectory_unnorm.shape[0], noisy_trajectory_unnorm.shape[1]

        # N X T X D -> NT X D 
        noisy_trajectory_unnorm = noisy_trajectory_unnorm.view(N * T, -1) # torch.flatten(noisy_trajectory_unnorm, start_dim=0, end_dim=1)
        trajectory_unnorm = trajectory_unnorm.view(N * T, -1) # torch.flatten(trajectory_unnorm, start_dim=0, end_dim=1)

        # NT X D -> NT X 6 
        n_traj_rot_matrx = rotation_6d_to_matrix(noisy_trajectory_unnorm[:, 3:9])
        traj_rot_matrx = rotation_6d_to_matrix(trajectory_unnorm[:, 3:9])
        d_rot = so3_distance(n_traj_rot_matrx, traj_rot_matrx) # NT 
        d_rot = d_rot.view(N, T) 
        d_rot_mean = torch.mean(d_rot, dim = 1) # size N 

        lmd = 0.1
        return d_trans + lmd * d_rot_mean 
        # return torch.mean(torch.square(noisy_trajectory.flatten(start_dim = 1) - trajectory.flatten(start_dim = 1)), dim = 1).detach()
        # return total_error.detach() 

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

    def compute_obs_representations(self, batch, select_keys = None):
        # this function will return a list of intermediate representation
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        if select_keys is not None:
            global_cond = self.obs_encoder.feature_by_key(nobs, select_keys)
        else:
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