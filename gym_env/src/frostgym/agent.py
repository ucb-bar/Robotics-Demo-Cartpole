from typing import Sequence

import torch
from torch.utils.tensorboard import SummaryWriter



class MLP(torch.nn.Module):
    def __init__(self, n_obs, n_acs):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(n_obs, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, n_acs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)

        return x



class PolicyGradientAgent:
    def __init__(self, n_obs, n_acs, lr=0.01, gamma=0.99):
        self.lr = lr
        self.gamma = gamma

        # create actor model
        self.actor = MLP(n_obs, n_acs)
        
        # create optimizer
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
    
    def calculateDiscountedReward(self, rewards: Sequence[torch.Tensor]) -> torch.Tensor:
        discounted_rewards = []
        for t in range(len(rewards)):
            discounted_reward = 0
            for t_prime in range(t, len(rewards)):
                discounted_reward += rewards[t_prime] * (self.gamma ** (t_prime - t))
            discounted_rewards.append(discounted_reward)
        return torch.tensor(discounted_rewards, dtype=torch.float32)

    def calculateQValues(self, rewards: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        # use rewardt-to-go
        q_values = []
        for trajectory in rewards:
            q_values.append(self.calculateDiscountedReward(trajectory))

        return q_values
    
    def estimateAdvantage(self, obs: torch.Tensor, rewards: torch.Tensor, q_values: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        advantages = q_values
        return advantages

    def updateActor(self, obs: torch.Tensor, acs: torch.Tensor, advantages: torch.Tensor) -> dict:
        loss = -(torch.distributions.Categorical(self.actor.forward(obs)).log_prob(acs).unsqueeze(dim=1).mean(dim=-1) * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = {
            "actor_loss": loss.item()
            }

        return info

    def update(self, obs: Sequence[torch.Tensor], acs: Sequence[torch.Tensor], rewards: Sequence[torch.Tensor], terminated: Sequence[torch.Tensor]) -> dict:
        q_values: Sequence[torch.Tensor] = self.calculateQValues(rewards)

        # flatten the sequences
        obs = torch.cat(obs)
        acs = torch.cat(acs)
        rewards = torch.cat(rewards)
        terminated = torch.cat(terminated)
        q_values = torch.cat(q_values)

        advantage = self.estimateAdvantage(obs, rewards, q_values, terminated)

        loss = self.updateActor(obs, acs, advantage)

        return loss

    def getAction(self, obs: torch.Tensor) -> torch.Tensor:
        # convert probability to one-hot action
        action_probs = self.actor.forward(obs)
        action = action_probs.multinomial(num_samples=1)[0]
        return action

    def save(self, path: str):
        torch.save(self.actor.state_dict(), path)

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(path))