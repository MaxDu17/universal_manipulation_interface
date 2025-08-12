import json 
import os 
import numpy as np 

import imageio 
import matplotlib.pyplot as plt 
from push_general_image_env import PushGeneralImageEnv

root_dir = "assets/procedural/"


def generate_t_to_cross(num_iterations):
    num_iterations = 10
    env_dict = {} 
    for iteration in range(num_iterations + 1):
        end = 1.5 # due to symmetry
        base_name = f"t_cross_{iteration}"
        filename = os.path.join(root_dir, f"{base_name}.json")
        env_dict[base_name] = {"file" : filename, "scale": 30 }

        with open(filename, "w") as f:
            step = (iteration / num_iterations) * end
            params =  {
                "horizontal" : [[-2, step + 1],
                [ 2, step + 1],
                [ 2, step],
                [-2, step]],
                "vertical": [[-0.5, 1],
                [-0.5, 4],
                [ 0.5, 4],
                [ 0.5, 1]]
            }
            json.dump(params, f)
    with open(os.path.join(root_dir, "t_cross_envs.json"), "w") as f:
        json.dump(env_dict, f, indent = 2)

num_iterations = 10 
generate_t_to_cross(num_iterations)
input("Press enter to generate!")

env = PushGeneralImageEnv(environments = os.path.join("assets", "procedural", "t_cross_envs.json"))
envs = {}
with open(os.path.join("assets", "procedural", "t_cross_envs.json"), "r") as f:
    envs = json.load(f)

for name, env_name in envs.items():
    env.load_env(name)
    env._setup()
    obs = env._get_obs()
    img = np.transpose(obs["image"], (1, 2, 0))
    plt.imsave(f"{name}.png", img)
