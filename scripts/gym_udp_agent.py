import isaacgym
import torch
from cc.udp import UDP, UDPTx, UDPRx

from frostgym.env import GymBaseEnvironment
from frostgym.agent import PolicyGradientAgent


recv_addr = ("0.0.0.0", 8001)
send_addr = ("127.0.0.1", 8000)

udp = UDP(recv_addr=recv_addr, send_addr=send_addr)

n_obs = 4
n_acs = 2

# prepare policy
agent = PolicyGradientAgent(n_obs, n_acs)
agent.load("cartpole_agent.pt")


# run trained policy

while True:
    obs = udp.recvNumpy(timeout=None)
    obs = torch.tensor(obs).unsqueeze(dim=0)

    acs = agent.actor.forward(obs)
    
    acs = acs.detach().numpy()
    print(acs, acs.dtype)
    udp.sendNumpy(acs)
