import torch
import numpy as np

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
        self.success_history, self.success_samples = [], 0

        # for exporting purposes
        # self.config = env_config
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

        self.time_cost = config["reward_config"]["time"]

    def step(self, actions):
        self.reset(self.reset_indices)

        terminated = torch.zeros(self.num_envs)
        terminated[self.reset_indices] = 1

        self.compute_next_state(actions)
        completed, failed = self.compute_termination()
        self._log_completion(failed, completed)
        self.compute_reward(completed)

        self.reset_indices = completed + failed

        return self.states, self.rewards, terminated

    def reset(self, indices=None):
        """Iteratively reinitialize & shuffle state at indices"""
        if indices == None:
            indices = list(range(self.num_envs))
        # sample n in one batch
        ns = self.sampler.sample((len(indices),))

        for i, n in zip(indices, ns):
            # sample n moves
            non_reduced_moves = np.random.choice(
                self.action_names, n.item(), replace=True
            )
            # reduce moves
            reduced_moves = iterate_reduce_sequence(non_reduced_moves, self.puzzle_name)
            # generate state from move
            state = generate_state_from_moves(
                reduced_moves, self.move_dict, self.final_state.copy(), inverse=False
            )
            self.states[i, :] = np.array(state, dtype=np.float32)
            self.gt_moves[i] = reduced_moves
            self.curr_steps[i] = 0
            self.cum_rewards[i] = 0

        return self.states

    def compute_next_state(self, actions):
        for i in range(self.num_envs):
            action = self.swaps[actions[i]]
            self.states[i] = permute_with_swap(self.states[i], action)
            self.curr_steps[i] += 1

    def compute_reward(self, completed):
        # step
        self.rewards = torch.ones_like(self.rewards) * self.time_cost

        for idx in completed:
            self.rewards[idx] += len(self.gt_moves[idx]) * 10

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

    def _log_completion(self, terminated, success):
        window_size = self.num_envs
        success_count, fail_count = len(success), len(terminated)

        self.success_history += [1] * success_count
        self.success_history += [0] * fail_count

        self.success_history = self.success_history[-window_size:]

    def get_completion_rate(self):
        return sum(self.success_history) / (len(self.success_history) + 1e-8)

    def get_cumulative_reward(self):
        return self.cum_rewards


class _Uniform_sampler:
    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high

    def sample(self, size) -> int:
        return torch.randint(self.low, self.high, size)
