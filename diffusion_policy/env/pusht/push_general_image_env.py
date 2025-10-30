from gym import spaces
try:
    from umi_day.common import import_umi_source
    from diffusion_policy.env.pusht.push_general_env import PushGeneralEnv
except: # local running
    from push_general_env import PushGeneralEnv

import numpy as np
import cv2
import tqdm

class PushGeneralImageEnv(PushGeneralEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            environments = None,
            load_env = None,
            use_old = False): # load_env allows you to specify the current environemnt. Not needed to initialize
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False,
            environments = environments,
            use_old = use_old)

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
            print("Loading task environment ", load_env)
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


def compute_grey_area(img, show_debug=False):

    # Convert to HSV
    img *= 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define grey range in HSV
    # Grey = low saturation, mid brightness (adjust if shape brightness differs)
    # lower_grey = np.array([0, 0, 50])     # H doesn't matter, S low, V not too dark

    lower_grey = np.array([0, 0, 0])     # H doesn't matter, S low, V not too dark
    upper_grey = np.array([179, 25, 180]) # S small range, V conservative

    # Threshold for grey area
    mask = cv2.inRange(hsv, lower_grey, upper_grey)

    # Optional morphological cleanup
    # kernel = np.ones((5,5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Compute area in pixels
    area_pixels = cv2.countNonZero(mask)

    # If you want contour area instead:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum(cv2.contourArea(c) for c in contours)

    if show_debug:
        cv2.imshow('Image', img)
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return area_pixels, contour_area

# unit test for checking that all the seeds will create things well within the bounds of the game area
def check_within_bounds(env, iterations = 1000):
    img_list = list()
    area_list = list()
    for i in tqdm.tqdm(range(iterations)):
        env.seed(i)
        env.reset()
        obs = env._get_obs()
        img = np.transpose(obs["image"], (1, 2, 0))
        area_pixels, contour_area = compute_grey_area(img)
        img_list.append(img.astype(np.float32)) # D, D, 3
        area_list.append(area_pixels)
        plt.imsave("test.png", img / 255)
    mean_area = sum(area_list) / iterations
    area_list = np.array(area_list)
    minimum_std = 8
    print(np.std(area_list))
    z_scores = (area_list - mean_area) / max(minimum_std, np.std(area_list))

    selected_scores = np.where(np.abs(z_scores) >= 3)[0]
    if selected_scores.shape[0] > 0:
        print(f"TEST FAILED! YOU HAVE {selected_scores.shape[0]} that violate. I'm saving 20 examples")
        for i in range(min(20, selected_scores.shape[0])):
            plt.imsave(f"violation_{i}_{z_scores[selected_scores[i]]}.png", img_list[selected_scores[i]] / 255)
    else:
        print("TEST PASSED!")
if __name__ == "__main__":
    import imageio 
    import matplotlib.pyplot as plt 

    env = PushGeneralImageEnv(environments = "assets/letters/environments.json", use_old = False)
    # env = PushGeneralImageEnv(environments = "assets/procedural/t_cross_envs.json")
    target_obj = "O"
    env.load_env(target_obj)


    check_within_bounds(env, iterations = 1000)
