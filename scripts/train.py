import os
import datetime

import isaacgym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from frostgym.gym_env import CartPoleEnvironment, SanityCheckCartPoleEnvironment
from frostgym.agent import PolicyGradientAgent

from params import *


# prepare environment
env = CartPoleEnvironment(alpha, beta, gamma, sigma)

# prepare policy
agent = PolicyGradientAgent(env.n_obs, env.n_acs, lr=lr)


exp_name = "{}_alp{}_bet{}_gam{}_sig{}".format(
    agent.actor.__class__.__name__,
    alpha,
    beta,
    gamma,
    sigma
)
log_dir = "logs/{}_{}".format(exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

writer = SummaryWriter(log_dir)



for epoch in range(n_epochs+1):
    print("******** EPOCH {} ********".format(epoch))
    trajectories, n_env_steps = env.sampleTrajectories(agent, min_timesteps_per_batch=train_batch_size, max_path_length=max_episode_length)
    total_env_steps += n_env_steps

    trajectory_dicts = {k: [traj[k] for traj in trajectories] for k in trajectories[0].keys()}

    train_info = agent.update(
        trajectory_dicts["obs"],
        trajectory_dicts["acs"],
        trajectory_dicts["rewards"],
        trajectory_dicts["terminated"]
    )

    # log
    if epoch % log_freq == 0:
        eval_trajectories, _ = env.sampleTrajectories(agent, min_timesteps_per_batch=eval_batch_size, max_path_length=max_episode_length)

        metrics = env.computeMetrics(writer, trajectories, eval_trajectories)

        print("Epoch: {}, Train return: {:.2f}, Eval return: {:.2f}, lr: {:.5f}".format(
            epoch,
            np.mean(metrics["train_returns"]),
            np.mean(metrics["eval_returns"]),
            agent.optimizer.param_groups[0]["lr"]
        ))

        writer.add_scalar("eval/average_return", np.mean(metrics["eval_returns"]), epoch)
        writer.add_scalar("eval/std_return", np.std(metrics["eval_returns"]), epoch)
        writer.add_scalar("eval/max_return", np.max(metrics["eval_returns"]), epoch)
        writer.add_scalar("eval/min_return", np.min(metrics["eval_returns"]), epoch)
        writer.add_scalar("eval/average_episode_length", np.mean(metrics["eval_episode_lengths"]), epoch)

        writer.add_scalar("train/average_return", np.mean(metrics["train_returns"]), epoch)
        writer.add_scalar("train/std_return", np.std(metrics["train_returns"]), epoch)
        writer.add_scalar("train/max_return", np.max(metrics["train_returns"]), epoch)
        writer.add_scalar("train/min_return", np.min(metrics["train_returns"]), epoch)
        writer.add_scalar("train/average_episode_length", np.mean(metrics["train_episode_lengths"]), epoch)

        writer.add_scalar("train/actor_loss", train_info["actor_loss"], epoch)
        writer.add_scalar("train/lr", agent.optimizer.param_groups[0]["lr"], epoch)

        writer.add_scalar("train/env_steps", total_env_steps, epoch)


env.close()

agent.save(os.path.join(log_dir, checkpoint_name))
