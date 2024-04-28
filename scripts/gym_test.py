
from frostgym.gym_env import GymBaseEnvironment
from frostgym.agent import PolicyGradientAgent


# prepare environment
env = GymBaseEnvironment("CartPole-v1", render_mode="human")

# prepare policy
agent = PolicyGradientAgent(env.n_obs, env.n_acs)
agent.load("checkpoints/cartpole_agent.pt")

# run trained policy
obs = env.reset()
done = False

while not done:
    acs = agent.getAction(obs.unsqueeze(dim=0))[0]
    obs, reward, terminated, truncated, info = env.step(acs)

    env.render()

    if terminated or truncated:
        obs = env.reset()
