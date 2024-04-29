import torch
import matplotlib.pyplot as plt

from frostgym.gym_env import SanityCheckCartPoleEnvironment
from frostgym.agent import PolicyGradientAgent


# prepare environment
env = SanityCheckCartPoleEnvironment()

# run trained policy
obs = env.reset()
done = False

state_traj = []
action_traj = []





while not done:
    u = torch.tensor([0.001], dtype=torch.float32)
    obs, reward, terminated, truncated, info = env.step(u)

    state_traj.append(obs)
    action_traj.append(u)

    print(obs)

    done = terminated or truncated

    if done:
        obs = env.reset()


state_traj = torch.stack(state_traj)
action_traj = torch.stack(action_traj)

plt.plot(state_traj[:, 0], label="theta")
plt.plot(state_traj[:, 1], label="theta_dot")
plt.plot(action_traj, label="u")

plt.legend()

plt.show()
