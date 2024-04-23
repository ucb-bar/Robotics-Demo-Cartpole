import numpy as np
import isaacgym
import torch
from cc.udp import UDP, UDPTx, UDPRx

from frostgym.env import GymBaseEnvironment
from frostgym.agent import PolicyGradientAgent


recv_addr = ("0.0.0.0", 8000)
send_addr = ("127.0.0.1", 8001)

udp = UDP(recv_addr=recv_addr, send_addr=send_addr)


# prepare environment
env = GymBaseEnvironment("CartPole-v1", render_mode="human")

# run trained policy
obs = env.reset()
done = False

obs = obs.detach().numpy()
udp.sendNumpy(obs)

while not done:
    acs = udp.recvNumpy(bufsize=8, dtype=np.float32, timeout=None)
    acs = torch.tensor(acs)

    acs = acs.multinomial(num_samples=1)[0]

    # acs = agent.getAction(obs.unsqueeze(dim=0))[0]

    print(acs)
    obs, reward, terminated, truncated, info = env.step(acs)


    obs = obs.detach().numpy()
    udp.sendNumpy(obs)

    env.render()

    if terminated or truncated:
        obs = env.reset()
