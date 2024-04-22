import sys
import os
import math
import time

import numpy as np
from isaacgym import gymapi, gymutil
import torch
from cc.udp import UDP

from frostgym.asset import CartPoleAsset
from frostgym.env import BaseIsaacGymConfig, BaseIsaacGymEnvironment



cfg = BaseIsaacGymConfig()
cfg.asset = CartPoleAsset()


env = BaseIsaacGymEnvironment(cfg)



# keys: dict_keys(['root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel'])

frame_idx = 0


dof_positions = torch.zeros(env.num_dofs)


while True:
    print("Frame: ", frame_idx)

    frame_idx += 1
    
    dof_positions[0] = 1
    
    

    env.step(dof_positions)

print("Done")
self.stop()

