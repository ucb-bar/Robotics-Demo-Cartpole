import os
import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from frostgym.gym_env import CartPoleEnvironment, SanityCheckCartPoleEnvironment
from frostgym.agent import PolicyGradientAgent

from params import *


# prepare environment
env = SanityCheckCartPoleEnvironment(0, 0, 0, 0)

# prepare policy
agent = PolicyGradientAgent(env.n_obs, env.n_acs, lr=lr)

# create 3D plot of obs[0] vs time vs reward
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

theta_0 = np.array([])
theta_2 = np.array([])
rewardss = np.array([])

for i_param in range(4000):

    agent.actor.K.detach()[0] = (np.random.random() - 0.5) * 5
    # agent.actor.K.detach()[1] = np.random.random() - 0.5
    agent.actor.K.detach()[2] = (np.random.random() - 0.5) * 20 - 20
    # agent.actor.K.detach()[3] = np.random.random() - 0.5


    if i_param % 100 == 0:
        print("******** EPOCH {} ********".format(i_param))

    trajectories = env.sampleNTrajectories(agent, n_trajectories=20, max_path_length=200)


    trajectory_dicts = {k: [traj[k].numpy() for traj in trajectories] for k in trajectories[0].keys()}


    reward_avg = 0
    for i in range(len(trajectory_dicts["obs"])):
        reward_avg += len(trajectory_dicts["rewards"][i])
    
    reward_avg /= len(trajectory_dicts["obs"])
        

    # for i in range(len(trajectory_dicts["obs"])):

    #     # train_info = agent.update(
    #     #     trajectory_dicts["obs"],
    #     #     trajectory_dicts["acs"],
    #     #     trajectory_dicts["rewards"],
    #     #     trajectory_dicts["terminated"]
    #     # )

    #     traj_obs = np.array(trajectory_dicts["obs"][i])
    #     traj_rewards = np.array(trajectory_dicts["rewards"][i])

    #     print(traj_obs.shape)
    #     print(traj_rewards.shape)
    #     # calculate cumulative rewards
    #     for i in range(1, traj_rewards.shape[0]):
    #         traj_rewards[i] += traj_rewards[i-1]

    #     # ax.plot(traj_obs[:, 0], traj_obs[:, 2], zs=traj_rewards[:], alpha=0.1)


    theta_0 = np.append(theta_0, agent.actor.K.detach()[0])
    theta_2 = np.append(theta_2, agent.actor.K.detach()[2])
    rewardss = np.append(rewardss, reward_avg)



# color by reward

dot_size = 2
ax.scatter(theta_0, theta_2, zs=rewardss, alpha=0.8, c=rewardss, cmap='inferno', s=dot_size)


#save the data
data = np.array([theta_0, theta_2, rewardss])
np.save("data_full.npy", data)


ax.set_xlabel("theta[0]")
ax.set_ylabel("theta[2]")
ax.set_zlabel("Reward")

plt.show()


