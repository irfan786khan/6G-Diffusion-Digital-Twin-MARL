# -*- coding: utf-8 -*-
"""Configuration parameters for the 6G Digital Twin system."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvironmentConfig:
    """Configuration for the 6G network environment."""
    num_base_stations: int = 3
    num_users: int = 20
    max_steps: int = 100
    bandwidth_mhz: float = 20.0
    carrier_frequency_ghz: float = 28.0


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion model."""
    state_dim: int = 15
    hidden_dim: int = 256
    num_timesteps: int = 100
    epochs: int = 35
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 0.01


@dataclass
class QMIXConfig:
    """Configuration for the QMIX agent."""
    num_agents: int = 3
    agent_state_dim: int = 5
    num_actions: int = 11
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.95
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    n_step: int = 3
    buffer_capacity: int = 50000
    batch_size: int = 128
    tau: float = 0.005


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    episodes_per_scenario: dict = None
    policy: str = "random"
    output_dir: str = "./6g_dataset"

    def __post_init__(self):
        if self.episodes_per_scenario is None:
            self.episodes_per_scenario = {
                "train_normal": 500,
                "train_congestion": 300,
                "train_high_mobility": 300,
                "train_outage": 200,
                "zero_shot_test": 200
            }


@dataclass
class TrainingConfig:
    """Configuration for MARL training."""
    num_episodes: int = 250
    synthetic_samples: int = 8000
    scale_factor: float = 12.0
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    num_test_episodes: int = 40
    scalability_user_counts: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    num_scalability_episodes: int = 30


class Config:
    """Master configuration class."""
    def __init__(self):
        self.environment = EnvironmentConfig()
        self.diffusion = DiffusionConfig()
        self.qmix = QMIXConfig()
        self.dataset = DatasetConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()

    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment.__dict__,
            'diffusion': self.diffusion.__dict__,
            'qmix': self.qmix.__dict__,
            'dataset': self.dataset.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__
        }