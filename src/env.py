import torch
import numpy as np

from mechanism.utils import load_puzzle, generate_state_from_moves, normalize_state
from mechanism.reduce import iterate_reduce_sequence
from mechanism.permute import permute_with_swap


class PuzzleEnv:
    def __init__(self, env_config):
        self._load_puzzle(env_config["puzzle_name"])
        self._load_config(env_config)

        self.states = np.empty((self.num_envs, self.state_size), dtype=np.float32)
        self.curr_steps = np.zeros(self.num_envs, dtype=int)
        self.rewards = torch.zeros((self.num_envs), dtype=torch.float32)
        # make sure we have access to the ground truth moves
        self.gt_moves = [[]] * self.num_envs

        # for exporting purposes
        # self.config = env_config
        self.puzzle_name = env_config["puzzle_name"]

        self.reset()

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
        self.compute_next_state(actions)
        completed, terminated = self.compute_termination()
        self.compute_reward(completed)

        reset_indices = completed + terminated
        self.reset(reset_indices)
        terminated = torch.zeros(self.num_envs)
        terminated[reset_indices] = 1

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

    def compute_next_state(self, actions):
        for i in range(self.num_envs):
            action = self.swaps[actions[i]]
            self.states[i] = permute_with_swap(self.states[i], action)
            self.curr_steps[i] += 1

    def compute_reward(self, completed):
        # step
        self.rewards = self.rewards + self.time_cost

        for idx in completed:
            self.rewards[idx] += len(self.gt_moves[idx])

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


class _Uniform_sampler:
    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high

    def sample(self, size) -> int:
        return torch.randint(self.low, self.high, size)
