# -*- coding: utf-8 -*-
"""Digital twin quality evaluation using statistical metrics."""

import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.preprocessing import StandardScaler


class DigitalTwinQualityEvaluator:
    """Evaluate digital twin quality using multiple statistical metrics."""

    def __init__(self, real_states, synthetic_states):
        self.real_states = real_states
        self.synthetic_states = synthetic_states
        self.feature_names = [
            'SINR1', 'SINR2', 'SINR3',
            'SINR_Var1', 'SINR_Var2', 'SINR_Var3',
            'Buffer1', 'Buffer2', 'Buffer3',
            'TP1', 'TP2', 'TP3',
            'Users1', 'Users2', 'Users3'
        ]

        self.scaler = StandardScaler()
        self.real_norm = self.scaler.fit_transform(real_states)
        self.synth_norm = self.scaler.transform(synthetic_states)

    def compute_mean_error(self):
        """Compute mean absolute error between distributions."""
        real_mean = np.mean(self.real_norm, axis=0)
        synth_mean = np.mean(self.synth_norm, axis=0)
        return np.abs(real_mean - synth_mean)

    def compute_variance_ratio(self):
        """Compute variance ratio between synthetic and real data."""
        real_var = np.var(self.real_norm, axis=0)
        synth_var = np.var(self.synth_norm, axis=0)
        return synth_var / (real_var + 1e-8)

    def compute_kl_divergence(self, bins=30):
        """Compute KL divergence for each feature."""
        kl_div = []
        for i in range(self.real_norm.shape[1]):
            h_r, bins_e = np.histogram(self.real_norm[:, i], bins=bins, density=True)
            h_s, _ = np.histogram(self.synth_norm[:, i], bins=bins_e, density=True)
            h_r = h_r + 1e-10
            h_s = h_s + 1e-10
            h_r = h_r / np.sum(h_r)
            h_s = h_s / np.sum(h_s)
            kl_div.append(entropy(h_r, h_s))
        return np.array(kl_div)

    def compute_wasserstein_distance(self):
        """Compute Wasserstein distance for each feature."""
        wd = [
            wasserstein_distance(self.real_norm[:, i], self.synth_norm[:, i])
            for i in range(self.real_norm.shape[1])
        ]
        return np.array(wd)

    def compute_scores(self):
        """Compute overall quality scores."""
        mean_error = self.compute_mean_error()
        kl_div = self.compute_kl_divergence()
        wd = self.compute_wasserstein_distance()

        mean_score = max(0, min(100, 100 - (np.mean(mean_error) * 50)))
        kl_score = max(0, min(100, 100 - (np.mean(kl_div) * 50)))
        wd_score = max(0, min(100, 100 - (np.mean(wd) * 100)))

        overall = (mean_score + kl_score + wd_score) / 3

        return {
            'mean_error': mean_error,
            'mean_error_avg': np.mean(mean_error),
            'kl_divergence': kl_div,
            'kl_divergence_avg': np.mean(kl_div),
            'wasserstein_distance': wd,
            'wasserstein_distance_avg': np.mean(wd),
            'mean_score': mean_score,
            'kl_score': kl_score,
            'wd_score': wd_score,
            'overall_score': overall
        }

    def get_quality_rating(self, score):
        """Get quality rating based on score."""
        if score > 80:
            return "Excellent"
        elif score > 65:
            return "Good"
        elif score > 50:
            return "Medium"
        else:
            return "Needs Improvement"