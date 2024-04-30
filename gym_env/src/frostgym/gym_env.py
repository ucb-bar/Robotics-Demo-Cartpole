
from typing import Sequence

import numpy as np
import gym
import torch

from .agent import PolicyGradientAgent, AffineM2P1LinearModule, AffineM2P2LinearModule


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

        if type(policy.actor) == AffineM2P1LinearModule or type(policy.actor) == AffineM2P2LinearModule:
            policy.actor.t = 0

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
    def __init__(self, alpha: float = .0, beta: float = .0, gamma: float = .0, sigma: float = .02):
        self.n_obs = 4
        self.n_acs = 1

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

        self.dt = 0.02


        g = 9.8
        M = 0.5
        m = 0.2
        l = 0.3
        I = m * l**2 # 0.006 -> 0.018
        b = 0.1



        self.steps = 0
        self.x: torch.Tensor = (
            torch.zeros(self.n_obs, dtype=torch.float32)
              + self.sigma * torch.randn(self.n_obs, dtype=torch.float32)
              )


        p = I*(M+m)+M*m*l**2

        self.A = torch.tensor([
            [0,               1,               0, 0],
            [0, -(I+m*l**2)*b/p, (m**2*g*l**2)/p, 0],
            [0,               0,               0, 1],
            [0,      -(m*l*b)/p,   m*g*l*(M+m)/p, 0],
        ], dtype=torch.float32)

        self.B = torch.tensor([
            [           0],
            [(I+m*l**2)/p],
            [           0],
            [       m*l/p],
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

        
    def reset(self) -> torch.Tensor:
        self.steps = 0
        self.x: torch.Tensor = (
            torch.zeros(self.n_obs, dtype=torch.float32)
              + self.sigma * torch.randn(self.n_obs, dtype=torch.float32)
              )

        C_t = self.C + self.gamma * torch.randn(size=())
        y_t = C_t @ self.x
        info = {}

        return y_t, info

    def step(self, u: torch.Tensor) -> tuple:
        self.steps += 1
        reward = torch.tensor(1., dtype=torch.float32)

        A_t = self.A + self.alpha * torch.randn(size=())
        B_t = self.B + self.beta * torch.randn(size=())
        C_t = self.C + self.gamma * torch.randn(size=())
        D_t = self.D

        # clip action
        u = torch.clamp(u, -3, 3)
        
        dx_t = A_t @ self.x + B_t @ u
        y_t = C_t @ self.x

        truncated = self.steps >= 2000

        terminated = bool(not torch.isfinite(y_t).all() or (np.abs(y_t[1]) > 1))

        self.x += dx_t * self.dt
        return (
            y_t,
            reward,
            terminated,
            truncated,
            {}
        )


class CartPoleEnvironment(SanityCheckCartPoleEnvironment):
    def __init__(self, alpha: float = .0, beta: float = .0, gamma: float = .0, sigma: float = .02, render_mode="rgb_array"):
        super().__init__(alpha, beta, gamma, sigma)

        self.render_mode = render_mode

        self.env = gym.make("InvertedPendulum-v4", render_mode=self.render_mode)

        self.spec = self.env.spec

        self.n_obs = self.env.observation_space.shape[0]
        self.n_acs = self.env.action_space.shape[0]
    
    def reset(self) -> torch.Tensor:
        obs, info = super().reset()
        self.env.reset(seed=0)
        return torch.tensor(obs, dtype=torch.float32)
    

    def step(self, acs: torch.Tensor) -> tuple:
        obs, reward, terminated, _, _ = super().step(acs)

        self.env.data.qpos[0] = -obs[0]
        self.env.data.qpos[1] = obs[2]
        self.env.data.qvel[0] = -obs[1]
        self.env.data.qvel[1] = obs[3]

        _, _, _, truncated, info = self.env.step(acs.to("cpu").detach().numpy())
        
        reward = torch.tensor(reward, dtype=torch.float32)

        return (
            obs,
            reward,
            terminated,
            truncated,
            info)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
