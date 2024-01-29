import math
import torch

from torch import optim, nn
from .dqn import DQN

__all__ = ["Agent"]


def Agent(config):
    if config["n"] == 1:
        return OneStepAgent(config)
    else:
        return NStepAgent(config)


class _Agent:
    def __init__(self, config):
        self._load(config)
        self.config = config

    def _load(self, config):
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.tau = config["target_network"]["tau"]
        self.target_net_delay = config["target_network"]["delay"]

        self.use_double = config["double"]

        self.epsilon_start = config["epsilon"]["start"]
        self.epsilon_end = config["epsilon"]["end"]
        self.epsilon_decay = config["epsilon"]["decay"]
        self.steps = config.get("steps", 0)

        self.device = "cpu"

        noisy = config["noisy_net"]
        inp, oup, network_units = (
            config["inp"],
            config["oup"],
            config["network_units"],
        )
        self.policy_network = DQN(inp, oup, network_units, noisy).to(self.device)
        self.target_network = DQN(inp, oup, network_units, noisy).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.noisy = noisy

        self.num_states, self.num_actions = inp, oup

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

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
            if self.tau == 0:
                target_sd[key] = policy_sd[key]
            else:
                target_sd[key] = policy_sd[key] * self.tau + target_sd[key] * (
                    1 - self.tau
                )
        self.target_network.load_state_dict(target_sd)

    def select_action(self, states, greedy=False):
        """Selects epsilon greedy action"""
        eps = 0 if greedy or self.noisy else self._eps()
        n = len(states)
        s = torch.rand(n, device=self.device)
        rand_actions = torch.randint(0, self.num_actions, (n,), device=self.device)
        with torch.no_grad():
            Q = self.policy_network(states)
            greedy_actions = torch.argmax(Q, dim=-1)

        return torch.where(s > eps, greedy_actions, rand_actions)

    def optimize(self, *args):
        pass

    def _update_history(self, *args):
        pass


class OneStepAgent(_Agent):
    def __init__(self, config):
        super().__init__(config)

    def optimize(self, s, a, r, s_n, terminated, mask):
        if self.noisy:
            self.policy_network.reset_noise()
            self.target_network.reset_noise()
        with torch.no_grad():
            if self.use_double:
                greedy_a = self.policy_network(s_n).argmax(axis=-1)
                max_Q_s_n = (
                    self.target_network(s_n).gather(1, greedy_a.unsqueeze(-1)).squeeze()
                )
            else:
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

        if self.steps % self.target_net_delay == 0:
            self.update_target_net()

        return loss.item()


class NStepAgent(_Agent):
    def __init__(self, config):
        super().__init__(config)

        self.n = config["n"]
        self._init_history()

    def _init_history(self):
        """Initiates the tensors for tracking episode history"""
        b, n = self.batch_size, self.n
        # we use a deque for state because it's cheaper & we don't need access to
        # intermediate values for compute, its shape is n x b x |S|
        # self.s = deque([torch.rand((b, self.num_states))], maxlen=n)
        self.s = torch.rand((b, n, self.num_states))
        self.s_n = torch.rand((b, n, self.num_states))
        self.a = torch.zeros((b, n, 1), dtype=int)
        self.r = torch.zeros((b, n, 1), dtype=torch.float32)
        # mask: 1 if (s, a, s') is an invalid transition
        self.m = torch.zeros((b, n, 1), dtype=torch.float32)
        # terminate: 1 if (s, a, s') and s' is successful
        self.t = torch.zeros((b, n, 1), dtype=torch.float32)

        # tracks G_t:t+n for every state
        self.G = torch.zeros((b, n, 1))
        # 1 if state s is part of the current episode
        self.episode_mask = torch.zeros((b, n, 1), dtype=torch.float32)
        # indicates the relative distance (to the right) to the end of episode
        # if the distance is larger than n we can simply apply min() when retrieving
        self.episode_end_idx = torch.zeros((b, n, 1), dtype=int)

        self.discount = self.gamma ** torch.arange(n - 1, -1, -1)
        self.discount = self.discount.unsqueeze(0).unsqueeze(-1)

    def _update_history(self, s, s_n, a, r, m, t):
        # reshape tensors
        s = s.unsqueeze(dim=1)
        s_n = s_n.unsqueeze(dim=1)
        a = a[..., None, None]
        r = r[..., None, None]
        t = t[..., None, None]
        m = m[..., None, None]

        # push everything to queue
        # self.s.append(s)
        self.s = torch.cat((self.s[:, 1:], s), dim=1)
        self.s_n = torch.cat((self.s_n[:, 1:], s_n), dim=1)
        self.a = torch.cat((self.a[:, 1:], a), dim=1)
        self.r = torch.cat((self.r[:, 1:], r), dim=1)
        self.m = torch.cat((self.m[:, 1:], m), dim=1)
        self.t = torch.cat((self.t[:, 1:], t), dim=1)

        self.G = torch.cat((self.G[:, 1:], torch.zeros_like(self.G[:, 0:1])), dim=1)

        inv_m = (m - 1) * -1  # 0 if episode ended, 1 if episode is continuing
        self.episode_mask *= inv_m  # zeros the row if episode ended
        self.episode_end_idx += self.episode_mask.long()
        self.episode_mask = torch.cat((self.episode_mask[:, 1:], inv_m), dim=1)

        self.episode_end_idx = torch.cat(
            (
                self.episode_end_idx[:, 1:],
                torch.zeros_like(self.episode_end_idx[:, 0:1]),
            ),
            dim=1,
        )

        # add reward to ongoing episodes
        self.G += self.discount * r * self.episode_mask

    def optimize(self, s, a, r, s_n, terminated, mask):
        self._update_history(s, s_n, a, r, mask, terminated)

        s_t = self.s[:, 0]
        a_t = self.a[:, 0]
        m_t = self.m[:, 0]

        G_tpn = self.G[:, 0]
        # (b, 1, 1)
        tpn_idx = self.episode_end_idx[:, 0].unsqueeze(-1)
        tpn_idx = torch.where(
            tpn_idx > self.n, torch.ones_like(tpn_idx) * (self.n), tpn_idx
        )
        # (b, |S|)
        s_tpn = self.s_n.gather(1, tpn_idx.repeat(1, 1, self.num_states)).squeeze(1)
        # (b, 1)
        t_tpn = self.t.gather(1, tpn_idx).squeeze(1)
        with torch.no_grad():
            # a_tpn = self.a.gather(1, self.episode_end_idx[:, 0].unsqueeze(-1)).squeeze()
            max_Q_s_tpn = self.target_network(s_tpn).max(axis=-1).values.unsqueeze(-1)

        # mask out terminal states
        max_Q_s_tpn = torch.where(
            t_tpn == 1, torch.zeros_like(max_Q_s_tpn), max_Q_s_tpn
        )
        target = G_tpn + self.gamma ** tpn_idx.squeeze(1) * max_Q_s_tpn

        # Q(s_t, a_t)
        Q = self.policy_network(s_t).gather(1, a_t)

        loss = self.criterion(Q, target)
        # apply mask on loss
        loss = torch.where(m_t == 1, torch.zeros_like(loss), loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

        self.steps += 1

        return loss.item()
