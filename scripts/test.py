
import torch

from frostgym.gym_env import CartPoleEnvironment
from frostgym.agent import PolicyGradientAgent

from params import *

# prepare environment
env = CartPoleEnvironment(alpha, beta, gamma, sigma, render_mode="human")

# prepare policy
agent = PolicyGradientAgent(env.n_obs, env.n_acs)
agent.load(checkpoint_name)

print(agent.actor.state_dict())

# run trained policy
obs = env.reset()
done = False

K = torch.tensor([[-1.,         -2.04076318, 20.36715237,  3.93021507]], dtype=torch.float32)

while not done:
    acs = agent.getAction(obs.unsqueeze(dim=0))[0]

    acs = -K @ obs

    print(acs)
    
    obs, reward, terminated, truncated, info = env.step(acs)

    env.render()

    if terminated or truncated:
        obs = env.reset()
