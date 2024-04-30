
import numpy as np
import matplotlib.pyplot as plt


data = np.load("data.npy")

theta_0, theta_2, rewards = data

# create 3D plot of obs[0] vs time vs reward
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
dot_size = 2
ax.scatter(theta_0, theta_2, zs=rewards, alpha=0.8, c=rewards, cmap='inferno', s=dot_size)

ax.set_xlabel("theta[0]")
ax.set_ylabel("theta[2]")
ax.set_zlabel("Reward")

plt.show()


