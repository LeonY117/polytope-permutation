import torch
import numpy as np

from typing import Dict

from .mechanism.utils import load_puzzle, generate_state_from_moves, normalize_state
from .mechanism.reduce import iterate_reduce_sequence
from .mechanism.permute import permute_with_swap


class PuzzleEnv:
    def __init__(self, env_config):
        self._load_puzzle(env_config["puzzle_name"])
        self._load_config(env_config)

        self.states = np.empty((self.num_envs, self.state_size), dtype=np.float32)
        self.curr_steps = np.zeros(self.num_envs, dtype=int)
        self.rewards = torch.zeros((self.num_envs), dtype=torch.float32)
        self.cum_rewards = torch.zeros((self.num_envs), dtype=torch.float32)
        # make sure we have access to the ground truth moves
        self.gt_moves = [[]] * self.num_envs

        # make a list of possible shuffle counts:
        l, r = self.reset_config["shuffle_range"]
        self.success_history = {n: [0] * self.num_envs for n in range(l, r)}

        # for exporting purposes
        self.config = env_config
        self.puzzle_name = env_config["puzzle_name"]

        self.reset()
        self.reset_indices = []

    def _load_puzzle(self, puzzle_name):
        move_dict, final_state = load_puzzle(puzzle_name, puzzle_dir="./puzzles")
        self.final_state = normalize_state(np.array(final_state, np.float32))
        self.state_size = len(self.final_state)
        # we just want to identify the move by an index:
        action_names, swaps = [], []
        for name, swap in move_dict.items():
            action_names.append(name)
            swaps.append(swap)

        # The available actions
        self.move_dict = {a: s for a, s in zip(action_names, swaps)}
        self.swaps = swaps
        self.action_names = np.array(action_names)
        self.num_actions = len(self.swaps)

        print(
            f"Loaded {puzzle_name}: {self.num_actions} actions, {self.state_size} states"
        )

    def _load_config(self, config):
        self.num_envs = config["num_envs"]
        self.max_steps = config["max_steps"]

        self.reset_config = config["reset_config"]
        self.sampler = _Uniform_sampler(*self.reset_config["shuffle_range"])
        self.extra_generation = self.reset_config["extra_generation"]

        self.time_cost = config["reward_config"]["time"]
        self.success_reward = config["reward_config"]["success"]

    def step(self, actions):
        self.reset(self.reset_indices)
        # invalid state transitions that start with a terminal state
        mask = torch.zeros(self.num_envs)
        mask[self.reset_indices] = 1

        self.compute_next_state(actions, mask)

        completed, failed = self.compute_termination()
        self._log_completion(completed, failed)
        self.compute_reward(completed)
        self.reset_indices = completed + failed

        terminated = torch.zeros(self.num_envs)
        terminated[completed] = 1

        return self.states, self.rewards, terminated, mask

    def reset(self, indices=None):
        """Iteratively reinitialize & shuffle state at indices"""
        if indices == None:
            indices = list(range(self.num_envs))

        # sample n in one batch
        ns = self.sampler.sample((len(indices),))

        for i, n in zip(indices, ns):
            # sample n_max moves
            non_reduced_moves = np.random.choice(
                self.action_names, n.item() + self.extra_generation, replace=True
            )
            # reduce moves
            reduced_moves = iterate_reduce_sequence(non_reduced_moves, self.puzzle_name)
            # slice n moves
            reduced_moves = reduced_moves[: n.item()]
            # generate state from move
            state = generate_state_from_moves(
                reduced_moves, self.move_dict, self.final_state.copy(), inverse=False
            )
            self.states[i, :] = np.array(state, dtype=np.float32)
            self.gt_moves[i] = reduced_moves
            self.curr_steps[i] = 0
            self.cum_rewards[i] = 0

        return self.states

    def compute_next_state(self, actions, mask):
        for i in range(self.num_envs):
            if mask[i]:
                continue
            action = self.swaps[actions[i]]
            self.states[i] = permute_with_swap(self.states[i], action)
            self.curr_steps[i] += 1

    def compute_reward(self, completed):
        # step
        self.rewards = torch.ones_like(self.rewards) * self.time_cost

        for idx in completed:
            self.rewards[idx] += self.success_reward

        self.cum_rewards += self.rewards

    def compute_termination(self):
        completed, terminated = [], []
        for i in range(self.num_envs):
            success = np.array_equal(self.states[i], self.final_state)
            if success:
                completed.append(i)
                continue
            if self.curr_steps[i] >= self.max_steps:
                terminated.append(i)
        return completed, terminated

    def _log_completion(self, success, terminated):
        window_size = self.num_envs
        for success_idx in success:
            n = len(self.gt_moves[success_idx])
            if n != 0:
                self.success_history[n].append(self.curr_steps[success_idx])
        for fail_idx in terminated:
            n = len(self.gt_moves[fail_idx])
            if n != 0:
                self.success_history[n].append(0)

        # truncate window
        for n in self.success_history.keys():
            self.success_history[n] = self.success_history[n][-window_size:]

    def get_completion_rate(self) -> (Dict, Dict, float):
        completion_rate_per_n = {}
        completion_length_per_n = {}
        average_completion_rate = 0
        for n, history in self.success_history.items():
            nonzero_count = np.count_nonzero(history) + 1e-8
            completion_rate_per_n[n] = nonzero_count / len(history)
            completion_length_per_n[n] = (
                sum([h for h in history if h != 0]) / nonzero_count
            )

        average_completion_rate = np.average(list(completion_rate_per_n.values()))

        return completion_rate_per_n, completion_length_per_n, average_completion_rate

    def get_cumulative_reward(self):
        return self.cum_rewards


class _Uniform_sampler:
    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high

    def sample(self, size) -> int:
        return torch.randint(self.low, self.high, size)
