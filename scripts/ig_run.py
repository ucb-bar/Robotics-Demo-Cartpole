import sys
import os
import math
import time

import numpy as np
from isaacgym import gymapi, gymutil
import torch
from cc.udp import UDP

from frostgym.asset import CartPoleAsset
from frostgym.env import SimulationEnv



udp_comm = UDP(recv_addr=("0.0.0.0", 6001), send_addr=("10.0.10.12", 6000))


asset = CartPoleAsset()

env = SimulationEnv()

env.loadAsset(asset)


# keys: dict_keys(['root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel'])

frame_idx = 0


dof_positions = np.zeros(env.num_dofs)


while True:
    # step the physics
    # env.gym.simulate(env.sim)
    # env.gym.fetch_results(env.sim, True)

    dof_positions[0] = 0.1
    
    

    env.step(dof_positions)

print("Done")
self.stop()

