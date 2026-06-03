# -*- coding: utf-8 -*-
"""Dataset generator for 6G network environment."""

import numpy as np
import h5py
import json
import os
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List

from environment import SixGNetworkEnv


class SixGDatasetGenerator:
    """Generates dataset from the 6G environment in multiple formats."""

    def __init__(self, output_dir: str = "./6g_dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.scenarios = {
            "train_normal": "normal",
            "train_congestion": "congestion",
            "train_high_mobility": "high_mobility",
            "train_outage": "outage",
            "zero_shot_test": "zero_shot_test"
        }

    def generate_episode(
        self,
        scenario: str,
        episode_id: int,
        max_steps: int = 100,
        policy: str = "random"
    ) -> Dict:
        """Generate a single episode."""
        env = SixGNetworkEnv(
            num_base_stations=3,
            num_users=20,
            scenario=scenario,
            seed=episode_id
        )

        state = env.reset(scenario=scenario)

        episode_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "throughputs": [],
            "sinr_values": [],
            "buffer_occupancies": [],
            "info": {
                "scenario": scenario,
                "episode_id": episode_id,
                "max_steps": max_steps,
                "num_base_stations": env.num_bs,
                "num_users": env.num_users,
                "timestamp": datetime.now().isoformat()
            }
        }

        for step in range(max_steps):
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "heuristic":
                action = np.clip(env.buffer_occupancy / 500, 0, 1)
            else:
                action = np.ones(env.num_bs) * 0.5

            next_state, reward, done, truncated, info = env.step(action)

            episode_data["states"].append(state.copy())
            episode_data["actions"].append(action.copy())
            episode_data["rewards"].append(reward)
            episode_data["throughputs"].append(info["throughput"].copy())
            episode_data["sinr_values"].append(info["average_sinr"].copy())
            episode_data["buffer_occupancies"].append(info["buffer_occupancy"].copy())

            state = next_state

        episode_data["states"] = np.array(episode_data["states"])
        episode_data["actions"] = np.array(episode_data["actions"])
        episode_data["rewards"] = np.array(episode_data["rewards"])
        episode_data["throughputs"] = np.array(episode_data["throughputs"])
        episode_data["sinr_values"] = np.array(episode_data["sinr_values"])
        episode_data["buffer_occupancies"] = np.array(episode_data["buffer_occupancies"])

        return episode_data

    def generate_full_dataset(
        self,
        episodes_per_scenario: Dict[str, int] = None,
        policy: str = "random"
    ):
        """Generate complete dataset with multiple scenarios."""
        if episodes_per_scenario is None:
            episodes_per_scenario = {
                "train_normal": 500,
                "train_congestion": 300,
                "train_high_mobility": 300,
                "train_outage": 200,
                "zero_shot_test": 200
            }

        all_datasets = {}

        for scenario_name, num_episodes in episodes_per_scenario.items():
            scenario_type = self.scenarios[scenario_name]
            scenario_episodes = []

            for episode_id in range(num_episodes):
                episode_data = self.generate_episode(
                    scenario=scenario_type,
                    episode_id=episode_id,
                    max_steps=100,
                    policy=policy
                )
                scenario_episodes.append(episode_data)

            all_datasets[scenario_name] = scenario_episodes

        return all_datasets

    def save_dataset(self, dataset: Dict, formats: List[str] = ["hdf5", "json", "pickle"]):
        """Save dataset in multiple formats."""
        saved_files = []

        if "hdf5" in formats:
            h5_path = os.path.join(self.output_dir, "6g_dataset.h5")
            self._save_as_hdf5(dataset, h5_path)
            saved_files.append(h5_path)

        if "json" in formats:
            json_path = os.path.join(self.output_dir, "6g_dataset_metadata.json")
            self._save_metadata_as_json(dataset, json_path)
            saved_files.append(json_path)

        if "pickle" in formats:
            pkl_path = os.path.join(self.output_dir, "6g_dataset.pkl")
            self._save_as_pickle(dataset, pkl_path)
            saved_files.append(pkl_path)

        npz_path = os.path.join(self.output_dir, "6g_dataset_episodes.npz")
        self._save_as_npz(dataset, npz_path)
        saved_files.append(npz_path)

        return saved_files

    def _save_as_hdf5(self, dataset: Dict, filepath: str):
        """Save dataset as HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['num_scenarios'] = len(dataset)

            for scenario_name, episodes in dataset.items():
                scenario_group = f.create_group(scenario_name)
                scenario_group.attrs['num_episodes'] = len(episodes)
                scenario_group.attrs['episode_length'] = len(episodes[0]['states'])

                for ep_idx, episode in enumerate(episodes):
                    ep_group = scenario_group.create_group(f'episode_{ep_idx}')

                    for key in episode['info']:
                        ep_group.attrs[key] = episode['info'][key]

                    for array_name in ['states', 'actions', 'rewards', 'throughputs', 'sinr_values', 'buffer_occupancies']:
                        ep_group.create_dataset(array_name, data=episode[array_name], compression='gzip')

    def _save_metadata_as_json(self, dataset: Dict, filepath: str):
        """Save dataset metadata as JSON."""
        metadata = {
            "created": datetime.now().isoformat(),
            "total_episodes": sum(len(eps) for eps in dataset.values()),
            "scenarios": {}
        }

        for scenario_name, episodes in dataset.items():
            metadata["scenarios"][scenario_name] = {
                "num_episodes": len(episodes),
                "episode_length": len(episodes[0]['states']),
                "state_dim": episodes[0]['states'].shape[1:],
                "action_dim": episodes[0]['actions'].shape[1],
                "sample_throughput_mean": float(np.mean([np.mean(e['throughputs']) for e in episodes])),
                "sample_reward_mean": float(np.mean([np.mean(e['rewards']) for e in episodes]))
            }

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _save_as_pickle(self, dataset: Dict, filepath: str):
        """Save dataset as pickle file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)

    def _save_as_npz(self, dataset: Dict, filepath: str):
        """Save as compressed numpy archive."""
        npz_dict = {}

        for scenario_name, episodes in dataset.items():
            for ep_idx, episode in enumerate(episodes):
                prefix = f"{scenario_name}_ep{ep_idx}"
                npz_dict[f"{prefix}_states"] = episode['states']
                npz_dict[f"{prefix}_actions"] = episode['actions']
                npz_dict[f"{prefix}_rewards"] = episode['rewards']
                npz_dict[f"{prefix}_throughputs"] = episode['throughputs']

        np.savez_compressed(filepath, **npz_dict)