import math
import torch
from collections import deque

from torch import optim, nn, autograd
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

    def _eps(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.steps / self.epsilon_decay
        )

    def update_target_net(self) -> None:
        target_sd = self.target_network.state_dict()
        policy_sd = self.policy_network.state_dict()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for key in policy_sd:
            target_sd[key] = policy_sd[key] * self.tau + target_sd[key] * (1 - self.tau)
        self.target_network.load_state_dict(target_sd)

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

    def optimize(self, *args):
        pass


class OneStepAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

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


class NStepAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.n = config["n"]

        self.history = deque([], maxlen=self.n)

        self.tpn_idx = torch.zeros((self.batch_size, 1))
        self.s_tpn = torch.zeros((self.batch_size, self.num_states))
        self.r_tpn = torch.zeros((self.batch_size, 1))

    def update_history(self, s, a, r, terminated, mask):
        snapshot = {"s": s, "a": a, "r": r, "T": terminated, "mask": mask}
        self.history.append(snapshot)
        # mapping = {"s": s, "a": a, "r": r, "T": terminated, "mask": mask}

        # # push to fixed size queue
        # for key, x in mapping.items():
        #     for idx in range(self.batch_size):
        #         self.history[key][idx].append(x[idx])

    def _get_tpn_idx_and_r(self, env_idx):
        """returns the index for item corresponding to t+n in history"""
        end = min(len(self.history), self.n)
        total_r = 0
        for i in range(end - 1):
            m = self.history[i + 1]["mask"][env_idx]
            r = self.history[i]["r"][env_idx]
            # next state transition is invalid
            total_r += self.gamma**i * r
            if m == 1:
                return i + 1, total_r

        return end - 1, total_r

    def optimize(self, s, a, r, s_n, terminated, mask):
        self.update_history(s, a, r, terminated, mask)

        s, a, r, terminated, mask = self.history[0].values()
        # get s_t+n and r_t+n-1
        for env_idx in range(self.batch_size):
            tpn_idx, r_tpn = self._get_tpn_idx_and_r(env_idx)
            s_tpn = self.history[tpn_idx]["s"][env_idx]
            self.tpn_idx[env_idx] = tpn_idx
            self.s_tpn[env_idx] = s_tpn
            self.r_tpn[env_idx] = r_tpn

        with torch.no_grad():
            max_Q_s_tpn = self.target_network(self.s_tpn).max(axis=-1).values

        # mask out terminal states
        max_Q_s_tpn = torch.where(
            terminated == 1, torch.zeros_like(max_Q_s_tpn), max_Q_s_tpn
        )

        # G_t:t+n = G_t:t+n-1 + gamma^n * max[Q(s_t+n, a)]
        target = (
            self.r_tpn.squeeze() + self.gamma ** self.tpn_idx.squeeze() * max_Q_s_tpn
        )

        # Q(s, a)
        Q = self.policy_network(s).gather(1, a.unsqueeze(-1))

        loss = self.criterion(Q, target.unsqueeze(-1))
        # mask out invalid transitions
        loss = torch.where(mask == 1, torch.zeros_like(loss), loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

        self.steps += 1

        return loss.item()
