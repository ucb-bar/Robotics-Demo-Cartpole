from typing import Sequence, List

import torch


class MLP(torch.nn.Module):
    def __init__(self, n_obs, n_acs):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(n_obs, 8)
        self.fc2 = torch.nn.Linear(8, 8)
        self.fc3 = torch.nn.Linear(8, n_acs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TransformerNet(torch.nn.Module):
    def __init__(self, n_obs, n_acs, device="cuda"):
        super(TransformerNet, self).__init__()
        self.device = device

        self.memory = torch.nn.Parameter(torch.zeros(size=(8, 8), device=self.device), requires_grad=True)
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=n_obs, nhead=4, device=self.device)
        self.fc1 = torch.nn.Linear(n_obs, 8, device=self.device)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model=8, nhead=4, device=self.device)
        self.fc2 = torch.nn.Linear(8, n_acs, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        x = self.encoder.forward(x)
        x = torch.relu(self.fc1.forward(x))
        x = self.decoder.forward(x, self.memory)
        x = self.fc2.forward(x)

        x = x.to("cpu")
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



class AffineM2P1LinearModule(torch.nn.Module):
    """
    Implements the policy function F(Y_(t)) = {K_0 + K_1 * Y_t} if t is zero, else {K_2 + K_3 * Y_t}.
    """
    def __init__(self, n_obs, n_acs):
        super().__init__()
        self.theta_0: torch.nn.Parameter = torch.nn.Parameter(data=torch.zeros(size=(n_acs, )), requires_grad=True)
        self.theta_1: torch.nn.Parameter = torch.nn.Parameter(data=torch.zeros(size=(n_obs, n_acs)), requires_grad=True)
        self.theta_2: torch.nn.Parameter = torch.nn.Parameter(data=torch.zeros(size=(n_obs, n_acs)), requires_grad=True)
        self.t = 0
        self.y_t_1: torch.Tensor = None

    def forward(self, y_t: torch.Tensor) -> torch.Tensor:
        u_t: torch.Tensor
        
        if self.t == 0:
            u_t = self.theta_0 + y_t @ self.theta_1
        else:
            u_t = self.theta_0 + y_t @ self.theta_1 + self.y_t_1 @ self.theta_2

        self.t += 1
        self.y_t_1 = y_t
        
        return u_t


class AffineM2P2LinearModule(torch.nn.Module):
    """
    Implements the policy function F(Y_(t)) = {K_0 + K_1 * Y_t} if t is even, {K_2 + K_3 * Y_t} if t is odd.
    """
    def __init__(self, n_obs, n_acs):
        super().__init__()
        self.theta_0: torch.nn.Parameter = torch.nn.Parameter(data=torch.zeros(size=(n_acs, )), requires_grad=True)
        self.theta_1: torch.nn.Parameter = torch.nn.Parameter(data=torch.zeros(size=(n_obs, n_acs)), requires_grad=True)
        self.theta_2: torch.nn.Parameter = torch.nn.Parameter(data=torch.zeros(size=(n_acs, )), requires_grad=True)
        self.theta_3: torch.nn.Parameter = torch.nn.Parameter(data=torch.zeros(size=(n_obs, n_acs)), requires_grad=True)
        self.t = 0
        
    def forward(self, y_t: torch.Tensor) -> torch.Tensor:
        u_t: torch.Tensor
        
        if self.t % 2 == 0: # t is even
            u_t = self.theta_0 + y_t @ self.theta_1
        else:
            u_t = self.theta_2 + y_t @ self.theta_3
        
        self.t += 1

        return u_t





class PolicyGradientAgent:
    def __init__(self, n_obs: int, n_acs: int, lr: float = 0.01, gamma: float = 0.99):
        self.lr = lr
        self.gamma = gamma

        # create actor model
        # self.actor = M1P1LinearModule(n_obs, n_acs)
        # self.actor = AffineM2P1LinearModule(n_obs, n_acs)
        # self.actor = AffineM2P2LinearModule(n_obs, n_acs)
        self.actor = MLP(n_obs, n_acs)
        # self.actor = TransformerNet(n_obs, n_acs)

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