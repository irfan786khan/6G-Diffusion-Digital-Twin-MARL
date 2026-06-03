# -*- coding: utf-8 -*-
"""QMIX Agent with Prioritized Experience Replay and Dueling Networks."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class PriorityReplayBuffer:
    """Prioritized Experience Replay buffer."""

    def __init__(self, capacity=50000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done, error=None):
        priority = abs(error) + 1e-6 if error is not None else 1.0
        priority = priority ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append((state.copy(), action.copy(), reward, next_state.copy(), done))
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = (state.copy(), action.copy(), reward, next_state.copy(), done)
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None

        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class DuelingQMixNetwork(nn.Module):
    """Dueling QMIX Network with separate value and advantage streams."""

    def __init__(self, num_agents=3, agent_state_dim=5, num_actions=11, hidden_dim=128):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.shared = nn.Sequential(
            nn.Linear(agent_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        self.target_shared = nn.Sequential(
            nn.Linear(agent_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.target_value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.target_advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        self._init_target_networks()

    def _init_target_networks(self):
        """Initialize target networks with the same weights."""
        for tp, p in zip(self.target_shared.parameters(), self.shared.parameters()):
            tp.data.copy_(p.data)
        for tp, p in zip(self.target_value_stream.parameters(), self.value_stream.parameters()):
            tp.data.copy_(p.data)
        for tp, p in zip(self.target_advantage_stream.parameters(), self.advantage_stream.parameters()):
            tp.data.copy_(p.data)

    def forward(self, obs):
        """Forward pass: Q = V + (A - mean(A))."""
        agent_qs = []
        for i in range(self.num_agents):
            features = self.shared(obs[:, i, :])
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
            agent_qs.append(q.unsqueeze(1))
        return torch.cat(agent_qs, dim=1)

    def get_target_q(self, obs):
        """Get target Q-values using target networks."""
        agent_qs = []
        for i in range(self.num_agents):
            features = self.target_shared(obs[:, i, :])
            value = self.target_value_stream(features)
            advantage = self.target_advantage_stream(features)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
            agent_qs.append(q.unsqueeze(1))
        return torch.cat(agent_qs, dim=1)

    def update_target(self, tau=0.005):
        """Soft update target networks using Polyak averaging."""
        for tp, p in zip(self.target_shared.parameters(), self.shared.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(self.target_value_stream.parameters(), self.value_stream.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(self.target_advantage_stream.parameters(), self.advantage_stream.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


class AdvancedQMIXAgent:
    """
    Advanced QMIX Agent with:
    - Prioritized Experience Replay
    - Dueling Network Architecture
    - Multi-step Learning
    - Adaptive Epsilon Decay
    """

    def __init__(self, num_agents=3, agent_state_dim=5, num_actions=11,
                 lr=3e-4, gamma=0.95, epsilon_start=0.5, epsilon_end=0.05,
                 epsilon_decay=0.995, n_step=3, device='cpu'):
        self.num_agents = num_agents
        self.agent_state_dim = agent_state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_step = n_step
        self.device = device

        self.actions = np.linspace(0, 1, num_actions)

        self.qmix = DuelingQMixNetwork(num_agents, agent_state_dim, num_actions).to(device)
        self.optimizer = optim.Adam(self.qmix.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300, eta_min=1e-6)

        self.real_buffer = PriorityReplayBuffer(capacity=50000)
        self.synthetic_buffer = PriorityReplayBuffer(capacity=50000)
        self.n_step_buffer = deque(maxlen=n_step)

        self.training_losses = []
        self.epsilon_history = []

    def act(self, state, eval_mode=False):
        """Select action with epsilon-greedy exploration."""
        if state.ndim == 1:
            state = state.reshape(self.num_agents, self.agent_state_dim)

        if not eval_mode and np.random.random() < self.epsilon:
            return np.array([np.random.choice(self.actions) for _ in range(self.num_agents)])

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.qmix(state_tensor)
        best_indices = q_values.argmax(dim=2).cpu().numpy()[0]
        return np.array([self.actions[idx] for idx in best_indices])

    def add_experience(self, state, action, reward, next_state, done, is_synthetic=False):
        """Add experience with N-step return calculation."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step or done:
            n_state, n_action, _, _, _ = self.n_step_buffer[0]
            n_next_state = self.n_step_buffer[-1][3]
            n_done = self.n_step_buffer[-1][4]

            n_reward = 0
            for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
                n_reward += (self.gamma ** i) * r

            buffer = self.synthetic_buffer if is_synthetic else self.real_buffer
            buffer.push(n_state, n_action, n_reward, n_next_state, n_done)

    def train_step(self, batch_size=128):
        """Perform training step with prioritized experience replay."""
        real_sample = self.real_buffer.sample(batch_size // 2)
        synth_sample = self.synthetic_buffer.sample(batch_size // 2)

        if real_sample is None and synth_sample is None:
            return 0.0

        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
        all_indices, all_weights = [], []

        if real_sample is not None:
            (sr, ar, rr, nr, dr), ir, wr = real_sample
            all_states.extend(sr)
            all_actions.extend(ar)
            all_rewards.extend(rr)
            all_next_states.extend(nr)
            all_dones.extend(dr)
            all_indices.extend(ir)
            all_weights.extend(wr)

        if synth_sample is not None:
            (ss, as_, rs, ns, ds), is_, ws = synth_sample
            all_states.extend(ss)
            all_actions.extend(as_)
            all_rewards.extend(rs)
            all_next_states.extend(ns)
            all_dones.extend(ds)
            all_indices.extend(is_)
            all_weights.extend(ws)

        states = np.array(all_states)
        actions = np.array(all_actions)
        rewards = np.array(all_rewards)
        next_states = np.array(all_next_states)
        dones = np.array(all_dones)
        weights = np.array(all_weights)

        states = states.reshape(-1, self.num_agents, self.agent_state_dim)
        next_states = next_states.reshape(-1, self.num_agents, self.agent_state_dim)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        current_q = self.qmix(states_t)
        actions_reshaped = actions.reshape(-1, self.num_agents)
        action_indices = np.argmin(
            np.abs(actions_reshaped[:, :, None] - self.actions[None, :]), axis=-1
        )
        action_indices_t = torch.LongTensor(action_indices).unsqueeze(-1).to(self.device)
        current_q_selected = current_q.gather(2, action_indices_t).squeeze(2).sum(dim=1)

        with torch.no_grad():
            next_q = self.qmix.get_target_q(next_states_t).max(dim=2)[0].sum(dim=1)
            target_q = rewards_t + self.gamma * (1 - dones_t) * next_q

        td_errors = (target_q - current_q_selected).detach().cpu().numpy()
        loss = (weights_t * nn.functional.mse_loss(current_q_selected, target_q, reduction='none')).mean()

        if real_sample is not None:
            self.real_buffer.update_priorities(
                all_indices[:len(real_sample[1])], td_errors[:len(real_sample[1])]
            )
        if synth_sample is not None:
            self.synthetic_buffer.update_priorities(
                all_indices[len(real_sample[1]):], td_errors[len(real_sample[1]):]
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.qmix.update_target(tau=0.005)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        self.training_losses.append(loss.item())

        return loss.item()