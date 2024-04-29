import numpy as np
import control
import matplotlib.pyplot as plt

class Parameters:
    g = 9.8
    M = 0.5
    m = 0.2
    b = 0.1
    I = 0.006
    l = 0.3

    p = I*(M+m)+M*m*l**2

    A = np.array([
        [0, 1, 0, 0],
        [0, -(I+m*l**2)*b/p, (m**2*g*l**2)/p, 0],
        [0, 0, 0, 1],
        [0, -(m*l*b)/p, m*g*l*(M+m)/p, 0],
    ])

    B = np.array([
        [0],
        [(I+m*l**2)/p],
        [0],
        [m*l/p],
    ])

    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    D = np.array([
        [0],
        [0],
        [0],
        [0],
    ])


p = Parameters()

A, B, C, D = p.A, p.B, p.C, p.D

sys_ss = control.StateSpace(A, B, C, D)

poles = control.pole(sys_ss)
print("Poles: ", poles)


Q = np.eye(4)
R = np.array([1.0])

K, S, E = control.lqr(A, B, Q, R)
print("K: ", K)



# obs = env.reset()

# while not done:
#     u = -torch.matmul(K, obs)
#     obs, reward, terminated, truncated, info = env.step(u)

#     state_traj.append(obs)
#     action_traj.append(u)

#     print(obs)

#     done = terminated or truncated

#     if done:
#         obs = env.reset()


# state_traj = torch.stack(state_traj)
# action_traj = torch.stack(action_traj)

# plt.plot(state_traj[:, 0], label="theta")
# plt.plot(state_traj[:, 1], label="theta_dot")
# plt.plot(action_traj, label="u")

# plt.legend()

# plt.show()
