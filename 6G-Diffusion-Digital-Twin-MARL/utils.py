# -*- coding: utf-8 -*-
"""Utility functions for the 6G Digital Twin system."""

import numpy as np
import torch
import random
import os


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_directory(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def compute_jains_fairness(values: np.ndarray) -> float:
    """Compute Jain's fairness index."""
    if np.sum(values) == 0:
        return 0.0
    return (np.sum(values) ** 2) / (len(values) * np.sum(values ** 2) + 1e-8)


def compute_shannon_capacity(sinr: float, bandwidth_hz: float = 20e6) -> float:
    """Compute Shannon capacity in Mbps."""
    return bandwidth_hz * np.log2(1 + sinr) / 1e6


def normalize_state(state: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    """Normalize state vector."""
    if mean is None or std is None:
        mean = np.mean(state, axis=0)
        std = np.std(state, axis=0) + 1e-8
    return (state - mean) / std, mean, std


def denormalize_state(state_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Denormalize state vector."""
    return state_norm * std + mean


def compute_throughput_stats(throughputs: list) -> dict:
    """Compute throughput statistics."""
    return {
        'mean': np.mean(throughputs),
        'std': np.std(throughputs),
        'median': np.median(throughputs),
        'min': np.min(throughputs),
        'max': np.max(throughputs),
        'p25': np.percentile(throughputs, 25),
        'p75': np.percentile(throughputs, 75),
        'p90': np.percentile(throughputs, 90),
        'p95': np.percentile(throughputs, 95)
    }


def moving_average(data: list, window: int = 50) -> list:
    """Compute moving average of a list."""
    if len(data) < window:
        return data
    return [np.mean(data[i:i+window]) for i in range(len(data) - window + 1)]


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"