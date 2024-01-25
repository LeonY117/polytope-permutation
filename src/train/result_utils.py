import os
import json

import torch

from tqdm import tqdm


class ResultManager:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

        self.directory = _create_dir(self.env.config["exp_name"])

    def evaluate(self, steps, greedy=True):
        history = {
            "eval": True,
            "reward": [],
            "avg_success": [],
            "success_per_n": {
                f"{n}": []
                for n in range(*self.env.config["reset_config"]["shuffle_range"])
            },
            "success_length_per_n": {
                f"{n}": []
                for n in range(*self.env.config["reset_config"]["shuffle_range"])
            },
        }
        s = self.env.reset()
        s = torch.tensor(s, device=self.agent.device)
        for _ in tqdm(range(steps)):
            a = self.agent.select_action(s, greedy=greedy)
            s_n, r, terminated, mask = self.env.step(a)
            s_n = torch.tensor(s_n, device=self.agent.device)
            s = s_n
            history["reward"].append(
                self.env.get_cumulative_reward().sum().item() / self.env.num_envs
            )
            (
                success_rate_per_n,
                success_length_per_n,
                average_success,
            ) = self.env.get_completion_rate()
            history["avg_success"].append(average_success)
            for n in success_rate_per_n.keys():
                history["success_per_n"][f"{n}"].append(success_rate_per_n[n])
                history["success_length_per_n"][f"{n}"].append(success_length_per_n[n])

        return history

    def save_model(self) -> None:
        print("saving networks...")
        torch.save(
            self.agent.policy_network,
            os.path.join(self.directory, "policy_net.pt"),
        )
        torch.save(
            self.agent.target_network,
            os.path.join(self.directory, "target_net.pt"),
        )
        torch.save(
            self.agent.optimizer,
            os.path.join(self.directory, "optimizer.pt"),
        )
        print("saved policy network and target networks")

        return

    def save_config(self) -> None:
        print("saving configs...")
        env_config = self.env.config
        agent_config = self.agent.config
        self.agent.config["steps"] = self.agent.steps

        with open(os.path.join(self.directory, "env_config.json"), "w+") as f:
            json.dump(env_config, f, indent=2)

        with open(os.path.join(self.directory, "agent_config.json"), "w+") as f:
            json.dump(agent_config, f, indent=2)

        print("saved env and agent configs")

        return

    def save(self):
        self.save_config()
        self.save_model()


def save_history(history, path):
    with open(os.path.join(path, "history.json"), "w+") as f:
        json.dump(history, f)


def _create_dir(folder_name, root_dir="results/"):
    dir_path = os.path.join(root_dir, folder_name)
    if folder_name not in os.listdir(root_dir):
        print(f"creating new experiment folder {folder_name}")
        os.mkdir(dir_path)
    else:
        num_items = len(os.listdir(dir_path))
        print(f"WARNING: folder {folder_name} already exists with {num_items} items")
    return dir_path
