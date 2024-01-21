import math
import torch

from torch import optim, nn
from .dqn import DQN


class Agent:
    def __init__(self, config):
        self._load(config)

        self.config = config

    def _load(self, config):
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.steps = config.get("steps", 0)

        self.device = "cpu"

        inp, oup, network_units = (
            config["inp"],
            config["oup"],
            config["network_units"],
        )
        self.policy_network = DQN(inp, oup, network_units).to(self.device)
        self.target_network = DQN(inp, oup, network_units).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.num_states, self.num_actions = inp, oup

        self.optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.SmoothL1Loss(reduction="none")

    def select_action(self, states, greedy=False):
        """Selects epsilon greedy action"""
        eps = 0 if greedy else self._eps()
        n = len(states)
        s = torch.rand(n, device=self.device)
        rand_actions = torch.randint(0, self.num_actions, (n,), device=self.device)
        with torch.no_grad():
            Q = self.policy_network(states)
            greedy_actions = torch.argmax(Q, dim=-1)

        return torch.where(s > eps, greedy_actions, rand_actions)

    def _eps(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.steps / self.epsilon_decay
        )

    def optimize(self, s, a, r, s_n, terminated, mask):
        with torch.no_grad():
            max_Q_s_n = self.target_network(s_n).max(axis=-1).values

        # mask out terminal states
        max_Q_s_n = torch.where(terminated == 1, torch.zeros_like(max_Q_s_n), max_Q_s_n)
        # r + gamma * max[Q(s', a)]
        target = r + self.gamma * max_Q_s_n

        # Q(s, a)
        Q = self.policy_network(s).gather(1, a.unsqueeze(-1))

        loss = self.criterion(Q, target.unsqueeze(-1))
        # apply mask on loss
        loss = torch.where(mask == 1, torch.zeros_like(loss), loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

        self.steps += 1

        return loss.item()

    def update_target_net(self) -> None:
        target_sd = self.target_network.state_dict()
        policy_sd = self.policy_network.state_dict()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for key in policy_sd:
            target_sd[key] = policy_sd[key] * self.tau + target_sd[key] * (1 - self.tau)
        self.target_network.load_state_dict(target_sd)
