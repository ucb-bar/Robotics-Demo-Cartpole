
from typing import Sequence

import numpy as np
import gym
import torch

from .agent import PolicyGradientAgent


class Environment:
    def __init__(self):
        pass

    def reset(self) -> torch.Tensor:
        pass

    def step(self, acs: torch.Tensor) -> tuple:
        pass
    
    def render(self):
        pass
    
    def close(self):
        pass

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



class SanityCheckCartPoleEnvironment(Environment):
    def __init__(self):
        self.n_obs = 4
        self.n_acs = 1

        class Spec:
            max_episode_steps = 1000
        self.spec = Spec()

        g = 9.8
        M = 0.5
        m = 0.2
        b = 0.1
        I = 0.006
        l = 0.3

        self.dt = 0.02
        self.t: int
        self.x: torch.Tensor

        p = I*(M+m)+M*m*l**2

        self.A = torch.tensor([
            [0, 1, 0, 0],
            [0, -(I+m*l**2)*b/p, (m**2*g*l**2)/p, 0],
            [0, 0, 0, 1],
            [0, -(m*l*b)/p, m*g*l*(M+m)/p, 0],
        ], dtype=torch.float32)
    
        self.B = torch.tensor([
            [0],
            [(I+m*l**2)/p],
            [0],
            [m*l/p],
        ], dtype=torch.float32)

        self.C = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        self.D = torch.tensor([
            [0],
            [0],
            [0],
            [0],
        ], dtype=torch.float32)

        self.reset()
        

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.x = torch.tensor([0, 0, 0.2, 0], dtype=torch.float32)

        return self.x

    def step(self, u: torch.Tensor) -> tuple:
        self.t += 1
        
        dx_t = self.A @ self.x + self.B @ u
        y_t = self.C @ self.x

        truncated = False
        if self.t >= 1000:
            truncated = True

        terminated = False
        if torch.abs(self.x[0]) > 5 or torch.abs(self.x[2]) > np.radians(15):
            terminated = True

        self.x += dx_t * self.dt
        return (
            y_t,
            torch.tensor(self.t, dtype=torch.float32),
            terminated,
            truncated,
            {}
        )



class CartPoleEnvironment(Environment):
    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode

        self.env = gym.make("InvertedPendulum-v4", render_mode=self.render_mode)

        self.spec = self.env.spec

        self.n_obs = self.env.observation_space.shape[0]
        self.n_acs = self.env.action_space.shape[0]
    
    def reset(self) -> torch.Tensor:
        obs, info = self.env.reset(seed=0)
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, acs: torch.Tensor) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(acs.to("cpu").detach().numpy())
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            terminated,
            truncated,
            info)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
