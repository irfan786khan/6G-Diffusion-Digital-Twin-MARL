# -*- coding: utf-8 -*-
"""Diffusion Model with Residual Connections for Synthetic State Generation."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ImprovedDiffusionModel(nn.Module):
    """
    Diffusion Model with Residual Connections.

    Features: Residual blocks, LayerNorm, and cosine noise schedule.
    """

    def __init__(self, state_dim=15, hidden_dim=256, num_timesteps=100):
        super().__init__()
        self.state_dim = state_dim
        self.num_timesteps = num_timesteps

        self.input_proj = nn.Linear(state_dim + 1, hidden_dim)

        self.res_block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1)
        )
        self.res_block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1)
        )
        self.res_block3 = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1)
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        self._build_cosine_schedule()

    def _build_cosine_schedule(self):
        """Construct cosine noise schedule for improved diffusion quality."""
        s = 0.008
        t = torch.linspace(0, 1, self.num_timesteps + 1)
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        self.register_buffer('alpha_bar', alpha_bar[:-1])
        self.alpha = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.beta = 1 - self.alpha

    def forward(self, x, t):
        """Forward pass with residual connections."""
        t_norm = (t.float() / self.num_timesteps).unsqueeze(1)
        h = self.input_proj(torch.cat([x, t_norm], dim=1))
        h = h + self.res_block1(h)
        h = h + self.res_block2(h)
        h = h + self.res_block3(h)
        return self.output_proj(h)

    def add_noise(self, x0, t):
        """Add noise to clean data at timestep t."""
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def sample(self, num_samples, device='cpu'):
        """Generate synthetic states using DDIM sampling."""
        self.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, self.state_dim).to(device)
            for t in reversed(range(0, self.num_timesteps, 10)):
                t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                noise_pred = self.forward(x, t_tensor)
                alpha_bar_t = self.alpha_bar[t]
                alpha_bar_prev = self.alpha_bar[max(t - 10, 0)]
                x = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                x = x * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * torch.randn_like(x)
        return x


def train_diffusion_model(states, epochs=35, batch_size=512, device='cpu'):
    """Train the improved diffusion model."""
    mean, std = states.mean(0), states.std(0) + 1e-8
    states_norm = (states - mean) / std

    dataset = TensorDataset(torch.FloatTensor(states_norm).to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ImprovedDiffusionModel(state_dim=states.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x0 = batch[0]
            t = torch.randint(0, model.num_timesteps, (x0.shape[0],), device=device)
            x_t, noise = model.add_noise(x0, t)
            loss = nn.functional.mse_loss(model.forward(x_t, t), noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    return model, mean, std


def generate_diverse_synthetic_states(model, mean, std, num_samples=8000,
                                      scale_factor=12.0, device='cpu'):
    """Generate diverse synthetic states for multiple unseen scenarios."""
    synthetic_norm = model.sample(num_samples, device).cpu().numpy()
    synthetic = synthetic_norm * std + mean

    scenarios = []

    s_a = synthetic.copy()
    s_a[:, 2] *= scale_factor * 1.5
    s_a[:, 0] -= 2
    scenarios.append(s_a)

    s_b = synthetic.copy()
    s_b[:, 0] *= 0.4
    s_b[:, 1] *= 2.0
    scenarios.append(s_b)

    s_c = synthetic.copy()
    s_c[:, 0] += np.random.normal(0, 1.5, num_samples)
    s_c[:, 4] *= 1.5
    scenarios.append(s_c)

    s_d = synthetic.copy()
    s_d[:, 2] *= scale_factor
    s_d[:, 0] += np.random.normal(-1, 1.0, num_samples)
    s_d[:, 4] *= 1.3
    scenarios.append(s_d)

    all_synthetic = np.concatenate(scenarios, axis=0)
    all_synthetic = np.clip(all_synthetic, 0, 100)

    return all_synthetic