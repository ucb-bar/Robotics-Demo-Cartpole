import sys
import os
import math
import time
from typing import Sequence

import numpy as np
import gym
import torch

from .agent import PolicyGradientAgent
from .asset import GenericAsset



class GymBaseEnvironment:
    def __init__(self, env_name: str, render_mode="rgb_array"):
        self.render_mode = render_mode

        self.env = gym.make(env_name, render_mode=self.render_mode)

        self.n_obs = self.env.observation_space.shape[0]
        self.n_acs = self.env.action_space.shape[0]
    
    def reset(self) -> torch.Tensor:
        obs, info = self.env.reset()
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, acs: torch.Tensor) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(acs.to("cpu").detach().numpy())
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(terminated),
            torch.tensor(truncated),
            info)

    def sampleTrajectory(self, policy: PolicyGradientAgent, max_path_length: int) -> dict:
        obs_traj = []
        next_obs_traj = []
        acs_traj = []
        rewards_traj = []
        terminated_traj = []

        obs = self.reset()

        rollout_done = False
        steps = 0
        while not rollout_done:
            obs_batch = obs.unsqueeze(dim=0)
            acs_batch = policy.getAction(obs_batch)
            acs = acs_batch[0]

            next_obs, reward, terminated, truncated, info = self.step(acs)
            steps += 1

            rollout_done = terminated or truncated or steps >= max_path_length

            obs_traj.append(obs)
            next_obs_traj.append(next_obs)
            acs_traj.append(acs)
            rewards_traj.append(reward)
            terminated_traj.append(torch.tensor(rollout_done))

            obs = next_obs

        traj = {
            "obs": torch.stack(obs_traj),
            "next_obs": torch.stack(next_obs_traj),
            "acs": torch.stack(acs_traj),
            "rewards": torch.stack(rewards_traj),
            "terminated": torch.stack(terminated_traj)
        }

        return traj

    def sampleTrajectories(self, policy: PolicyGradientAgent, min_timesteps_per_batch: int, max_path_length: int) -> tuple:
        trajectories = []
        timesteps_this_batch = 0
        while timesteps_this_batch < min_timesteps_per_batch:
            traj = self.sampleTrajectory(policy, max_path_length)
            trajectories.append(traj)

            timesteps_this_batch += len(traj["obs"])

        return trajectories, timesteps_this_batch

    def sampleNTrajectories(self, policy: torch.nn.Module, n_trajectories: int, max_path_length: int) -> tuple:
        trajectories = []
        for _ in range(n_trajectories):
            trajectory = self.sampleTrajectory(policy, max_path_length)
            trajectories.append(trajectory)
        
        return trajectories

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def computeMetrics(self, writer, train_trajectories: Sequence[dict], eval_trajectories: Sequence[dict]) -> dict:
        train_returns = [traj["rewards"].sum().item() for traj in train_trajectories]
        eval_returns = [traj["rewards"].sum().item() for traj in eval_trajectories]

        train_episode_lengths = [len(traj["rewards"]) for traj in train_trajectories]
        eval_episode_lengths = [len(traj["rewards"]) for traj in eval_trajectories]

        metrics = {
            "train_returns": train_returns,
            "eval_returns": eval_returns,
            "train_episode_lengths": train_episode_lengths,
            "eval_episode_lengths": eval_episode_lengths
        }

        return metrics

