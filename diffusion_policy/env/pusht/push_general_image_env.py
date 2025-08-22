from gym import spaces
try:
    from umi_day.common import import_umi_source
    from diffusion_policy.env.pusht.push_general_env import PushGeneralEnv
except: # local running
    from push_general_env import PushGeneralEnv

import numpy as np
import cv2

class PushGeneralImageEnv(PushGeneralEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            environments = None,
            load_env = None): # load_env allows you to specify the current environemnt. Not needed to initialize 
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False,
            environments = environments)

        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None

        if load_env is not None: 
            print("Loading task ", load_env)
            self.load_env(load_env)

    
    def _get_obs(self, render_goal = True):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache

if __name__ == "__main__":
    import imageio 
    import matplotlib.pyplot as plt 

    env = PushGeneralImageEnv(environments = "assets/letters/environments.json")
    # env = PushGeneralImageEnv(environments = "assets/procedural/t_cross_envs.json")
    target_obj = "S"
    env.load_env(target_obj)
    img_list = list()
    for i in range(100):
        env.seed(i)
        env.reset()
        obs = env._get_obs()
        img = np.transpose(obs["image"], (1, 2, 0))
        img_list.append(img.astype(np.float32)) # D, D, 3

    # avg_img = np.mean(np.stack(img_list, axis = 0), axis=0)
    # plt.imsave(f"{target_obj}.png", avg_img)

    # Convert to grayscale to simplify presence detection
    gray_imgs = [np.mean(img[10 : -10, 10 : -10], axis=2) for img in img_list]

    # Threshold: 1 if pixel belongs to shape, 0 if background
    mask_imgs = [(gray < 0.99).astype(np.float32) for gray in gray_imgs]  # adjust threshold

    # Sum over all masks
    heatmap = np.sum(mask_imgs, axis=0)  # each pixel shows how often it was occupied

    # Normalize for visualization
    heatmap /= np.max(heatmap)

    plt.imsave(f"{target_obj}_heatmap.png", heatmap, cmap="hot")

    #
    # letter_list = ["T", "L", "J"]
    # for letter in letter_list:
    #     print(letter)
    #     env.load_env(letter)
    #     env._setup()
    #     import ipdb
    #     ipdb.set_trace()
    #     obs = env._get_obs()
    #     img = np.transpose(obs["image"], (1, 2, 0))
    #     plt.imsave(f"{letter}.png", img)

    # for i in range(11):
    #     env.load_env(f"t_cross_{i}")
    #     env._setup()
    #     obs = env._get_obs()
    #     img = np.transpose(obs["image"], (1, 2, 0))
    #     plt.imsave(f"t_cross_{i}.png", img)
