# -*- coding: utf-8 -*-
"""Dataset loader for 6G network HDF5 data."""

import h5py
import numpy as np


class SixGDatasetLoader:
    """HDF5 dataset loader for 6G network data."""

    @staticmethod
    def load_hdf5(filepath):
        """Load dataset from HDF5 file."""
        dataset = {}
        with h5py.File(filepath, 'r') as f:
            for scenario_name in f.keys():
                if scenario_name.startswith('_'):
                    continue
                dataset[scenario_name] = []
                for ep_name in f[scenario_name].keys():
                    ep = f[f"{scenario_name}/{ep_name}"]
                    episode = {
                        'states': ep['states'][:],
                        'actions': ep['actions'][:],
                        'rewards': ep['rewards'][:],
                        'throughputs': ep['throughputs'][:],
                    }
                    dataset[scenario_name].append(episode)
        return dataset

    @staticmethod
    def extract_normal_states(dataset):
        """Extract normal states from the training dataset."""
        normal_episodes = dataset['train_normal']
        all_states = []
        for episode in normal_episodes:
            all_states.extend(episode['states'].reshape(-1, 15))
        return np.array(all_states)