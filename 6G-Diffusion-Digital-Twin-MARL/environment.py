# -*- coding: utf-8 -*-
"""6G Network Environment with Enhanced Reward Shaping."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class SixGNetworkEnv(gym.Env):
    """
    6G Network Environment with Enhanced Reward Shaping.

    Includes fairness and stability metrics for improved learning.
    """

    def __init__(self, num_base_stations=3, num_users=20, scenario="normal", seed=None):
        super().__init__()
        self.num_bs = num_base_stations
        self.num_users = num_users
        self.scenario = scenario
        self.current_step = 0
        self.max_steps = 100
        self.throughput_history = deque(maxlen=10)

        if seed is not None:
            np.random.seed(seed)

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_bs,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(self.num_bs, 5), dtype=np.float32
        )
        self.reset()

    def _generate_channel_parameters(self):
        """Generate scenario-specific channel parameters."""
        scenarios = {
            "normal": (3.5, 8.0, 1.0, -90),
            "congestion": (4.0, 10.0, 0.5, -85),
            "high_mobility": (3.8, 6.0, 0.3, -88),
            "outage": (4.2, 12.0, 0.4, -80),
        }
        return scenarios.get(self.scenario, (4.5, 15.0, 0.2, -75))

    def reset(self, scenario=None, seed=None):
        """Reset environment to initial state."""
        if scenario is not None:
            self.scenario = scenario
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.throughput_history.clear()

        (self.path_loss_exp, self.shadowing_std,
         self.fading_factor, self.noise_floor) = self._generate_channel_parameters()

        self.bs_positions = np.random.rand(self.num_bs, 2) * 1000
        self.user_positions = np.random.rand(self.num_users, 2) * 1000

        if self.scenario == "high_mobility":
            self.user_speeds = np.random.uniform(5, 30, self.num_users)
        else:
            self.user_speeds = np.random.uniform(0.5, 5, self.num_users)

        self.channel_gains = self._compute_channel_gains()
        self.sinr = np.zeros((self.num_bs, self.num_users))
        self._update_sinr()
        self.buffer_occupancy = np.random.uniform(0, 500, self.num_bs)
        self.historical_throughput = np.zeros(self.num_bs)

        if self.scenario == "congestion":
            self.traffic_rate = np.random.uniform(100, 300, self.num_bs)
        else:
            self.traffic_rate = np.random.uniform(20, 80, self.num_bs)

        return self._get_state()

    def _compute_channel_gains(self):
        """Compute path loss, shadowing, and fading using 3GPP model."""
        gains = np.zeros((self.num_bs, self.num_users))
        for b in range(self.num_bs):
            for u in range(self.num_users):
                distance = max(
                    np.linalg.norm(self.bs_positions[b] - self.user_positions[u]), 10
                )
                path_loss = 20 * np.log10(distance) + 20 * np.log10(28e9) - 147.55
                path_loss += 10 * self.path_loss_exp * np.log10(distance / 100)
                shadowing = np.random.normal(0, self.shadowing_std)
                fading = np.random.rayleigh(self.fading_factor)
                gains[b, u] = fading * (10 ** ((-path_loss - shadowing) / 10))
        return gains

    def _update_sinr(self):
        """Update SINR with interference calculation."""
        if not hasattr(self, 'current_power'):
            self.current_power = np.ones(self.num_bs) * 0.5

        received_power = self.current_power.reshape(-1, 1) * self.channel_gains
        noise_power = 10 ** (self.noise_floor / 10) / 1000

        for u in range(self.num_users):
            for b in range(self.num_bs):
                signal = received_power[b, u]
                interference = np.sum([
                    received_power[i, u] for i in range(self.num_bs) if i != b
                ])
                self.sinr[b, u] = signal / (interference + noise_power + 1e-10)
        return self.sinr

    def _get_user_association(self):
        """Associate each user to the base station with highest SINR."""
        return np.argmax(self.sinr, axis=0)

    def _compute_throughput(self):
        """Compute throughput using Shannon-Hartley theorem."""
        user_association = self._get_user_association()
        throughput_per_bs = np.zeros(self.num_bs)

        for b in range(self.num_bs):
            users_served = np.where(user_association == b)[0]
            if len(users_served) > 0:
                bandwidth_per_user = 20e6 / len(users_served)
                for u in users_served:
                    throughput_per_bs[b] += (
                        bandwidth_per_user * np.log2(1 + self.sinr[b, u]) / 1e6
                    )
        return throughput_per_bs

    def _update_mobility(self):
        """Update user positions using random walk mobility model."""
        directions = np.random.uniform(0, 2 * np.pi, self.num_users)
        step_sizes = self.user_speeds * 0.1
        self.user_positions[:, 0] += step_sizes * np.cos(directions)
        self.user_positions[:, 1] += step_sizes * np.sin(directions)
        self.user_positions = np.clip(self.user_positions, 0, 1000)
        self.channel_gains = self._compute_channel_gains()

    def _update_traffic(self):
        """Update buffer occupancy with traffic arrival and service."""
        arrivals = np.random.poisson(self.traffic_rate / 10, self.num_bs)
        throughput = self._compute_throughput()
        self.buffer_occupancy += arrivals * 1500
        self.buffer_occupancy -= throughput * 1e6 / 8 * 0.1
        self.buffer_occupancy = np.maximum(0, self.buffer_occupancy)

    def _get_state(self):
        """Construct state vector with 5 features per base station."""
        avg_sinr = np.mean(self.sinr, axis=1)
        sinr_variance = np.var(self.sinr, axis=1)
        normalized_buffer = self.buffer_occupancy / 1000
        normalized_throughput = self.historical_throughput / 100
        user_association = self._get_user_association()
        num_users_per_bs = np.array([
            np.sum(user_association == b) for b in range(self.num_bs)
        ])
        normalized_users = num_users_per_bs / self.num_users

        state = np.column_stack([
            avg_sinr, sinr_variance, normalized_buffer,
            normalized_throughput, normalized_users
        ])
        return state.astype(np.float32)

    def step(self, actions):
        """Execute one timestep with enhanced reward shaping."""
        self.current_power = np.clip(actions, 0, 1)
        self._update_sinr()
        throughput = self._compute_throughput()

        self.throughput_history.append(np.mean(throughput))

        throughput_reward = np.sum(np.log(throughput + 1e-6))

        if np.sum(throughput) > 0:
            fairness = (np.sum(throughput) ** 2) / (
                self.num_bs * np.sum(throughput ** 2) + 1e-8
            )
        else:
            fairness = 0
        fairness_reward = fairness * 2.0

        if len(self.throughput_history) > 1:
            stability = -np.std(list(self.throughput_history)) * 0.1
        else:
            stability = 0

        reward = throughput_reward + fairness_reward + stability

        self.historical_throughput = throughput
        self._update_traffic()
        self._update_mobility()
        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = {
            "throughput": throughput,
            "buffer_occupancy": self.buffer_occupancy.copy(),
            "average_sinr": np.mean(self.sinr, axis=1),
            "scenario": self.scenario,
            "fairness": fairness,
        }
        return self._get_state(), reward, done, False, info