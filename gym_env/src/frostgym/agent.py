from typing import Sequence, List

import torch
from torch.utils.tensorboard import SummaryWriter


class MLP(torch.nn.Module):
    def __init__(self, n_obs, n_acs):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(n_obs, 24)
        self.fc2 = torch.nn.Linear(24, 24)
        self.fc3 = torch.nn.Linear(24, n_acs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.softmax(self.fc3(x), dim=-1)

        x = self.fc3(x)

        return x


class M1P1LinearModule(torch.nn.Module):
    """
    Implements the policy function F(Y_(t)) = K * Y_t.
    """
    def __init__(self, n_obs, n_acs):
        super().__init__()
        self.K: torch.nn.Parameter = torch.nn.Parameter(data=torch.zeros(size=(n_obs, n_acs)), requires_grad=True)

    def forward(self, y_t: torch.Tensor) -> torch.Tensor:
        u_t: torch.Tensor = y_t @ self.K
        return u_t



class PolicyGradientAgent:
    def __init__(self, n_obs: int, n_acs: int, lr: float = 0.01, gamma: float = 0.99):
        self.lr = lr
        self.gamma = gamma

        # create actor model
        # self.actor = MLP(n_obs, n_acs)
        self.actor = M1P1LinearModule(n_obs, n_acs)

        # create logstd
        self.logstd = torch.nn.Parameter(torch.zeros(n_acs))
        self.parameters = list(self.actor.parameters()) + [self.logstd]

        # create optimizer
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

    
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
        # action_prob = torch.distributions.Categorical(self.actor.forward(obs))
        action_prob = torch.distributions.Normal(loc=self.actor.forward(obs), scale=torch.exp(self.logstd))

        # loss = -(action_prob.log_prob(acs).unsqueeze(dim=1).mean(dim=-1) * advantages).mean()
        loss = -(action_prob.log_prob(acs).mean(dim=-1) * advantages).mean()

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
        # action_probs = self.actor.forward(obs)
        # action = action_probs.multinomial(num_samples=1)[0]

        # action_prob = torch.distributions.Categorical(self.actor.forward(obs))
        action_prob = torch.distributions.Normal(loc=self.actor.forward(obs), scale=torch.exp(self.logstd))

        action = action_prob.sample()
        return action

    def save(self, path: str):
        torch.save(self.actor.state_dict(), path)

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(path))