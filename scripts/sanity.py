import numpy as np
import torch
import control
import matplotlib.pyplot as plt

from frostgym.gym_env import SanityCheckCartPoleEnvironment, CartPoleEnvironment


# prepare environment
# env = SanityCheckCartPoleEnvironment()
env = CartPoleEnvironment(render_mode="human")

A, B, C, D = env.A.numpy(), env.B.numpy(), env.C.numpy(), env.D.numpy()

sys_ss = control.StateSpace(A, B, C, D)

poles = control.pole(sys_ss)
print("Poles: ", poles)

Q = np.eye(4)
R = np.eye(1)

K, S, E = control.lqr(A, B, Q, R)
print("K: ", K)
K = torch.tensor(K, dtype=torch.float32)


obs = env.reset()
done = False

state_traj = []
action_traj = []


while not done:
    x = obs

    u = -K @ x
    acs = u

    print(obs)
    obs, reward, terminated, truncated, info = env.step(acs)

    state_traj.append(obs)
    action_traj.append(u)

    done = terminated or truncated

    if done:
        obs = env.reset()
        print("Resetting")


state_traj = torch.stack(state_traj)
action_traj = torch.stack(action_traj)

plt.plot(state_traj[:, 0], label="theta")
plt.plot(state_traj[:, 1], label="theta_dot")
plt.plot(action_traj, label="u")

plt.legend()

plt.show()
