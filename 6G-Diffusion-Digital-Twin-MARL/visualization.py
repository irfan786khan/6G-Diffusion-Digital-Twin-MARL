# -*- coding: utf-8 -*-
"""Visualization utilities for all plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


class ResultVisualizer:
    """Generate all visualization plots for the paper."""

    def __init__(self, output_dir='./results'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def plot_diffusion_training(self, per_step_losses, per_epoch_losses, epochs):
        """Plot diffusion model training loss."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(per_step_losses, alpha=0.5, linewidth=0.5, color='blue')
        axes[0].set_xlabel('Training Step', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Diffusion Model Training Loss (Per Step)', fontsize=12)
        axes[0].set_yscale('log')

        axes[1].plot(range(1, epochs + 1), per_epoch_losses, 'ro-', linewidth=2, markersize=6)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Average Loss', fontsize=12)
        axes[1].set_title('Diffusion Model Training Loss (Per Epoch)', fontsize=12)
        axes[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/diffusion_training_loss.pdf', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_synthetic_distributions(self, original_synthetic, diverse_synthetic):
        """Plot synthetic data distributions."""
        features = ['Avg SINR', 'SINR Variance', 'Buffer Occ', 'Throughput', 'Users per BS']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(5):
            axes[i].hist(original_synthetic[:, i], bins=50, alpha=0.5, label='Original', color='blue')
            axes[i].hist(diverse_synthetic[:, i], bins=50, alpha=0.5, label='Diverse', color='red')
            axes[i].set_xlabel(features[i], fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'Distribution of {features[i]}', fontsize=11)
            axes[i].legend()

        axes[5].axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/synthetic_distributions.pdf', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_training_metrics(self, rewards, moving_avg, throughputs, fairness, losses, epsilon, steps):
        """Plot MARL training metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].plot(rewards, alpha=0.5, linewidth=1, label='Episode Reward')
        axes[0, 0].plot(moving_avg, 'r-', linewidth=2, label='Moving Avg (50 ep)')
        axes[0, 0].set_xlabel('Episode', fontsize=10)
        axes[0, 0].set_ylabel('Total Reward', fontsize=10)
        axes[0, 0].set_title('Training Rewards', fontsize=12)
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].plot(throughputs, 'g-', linewidth=1, alpha=0.7)
        axes[0, 1].set_xlabel('Episode', fontsize=10)
        axes[0, 1].set_ylabel('Throughput (Mbps)', fontsize=10)
        axes[0, 1].set_title('Throughput During Training', fontsize=12)
        z = np.polyfit(range(len(throughputs)), throughputs, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(p(range(len(throughputs))), "r--", linewidth=2, label=f'Trend (slope: {z[0]:.3f})')
        axes[0, 1].legend(fontsize=8)

        axes[0, 2].plot(fairness, 'b-', linewidth=1, alpha=0.7)
        axes[0, 2].set_xlabel('Episode', fontsize=10)
        axes[0, 2].set_ylabel('Fairness Index', fontsize=10)
        axes[0, 2].set_title('Fairness During Training', fontsize=12)
        axes[0, 2].set_ylim(0, 1)

        axes[1, 0].plot(losses, 'purple', linewidth=1, alpha=0.7)
        axes[1, 0].set_xlabel('Episode', fontsize=10)
        axes[1, 0].set_ylabel('Average Loss', fontsize=10)
        axes[1, 0].set_title('Training Loss', fontsize=12)
        axes[1, 0].set_yscale('log')

        axes[1, 1].plot(epsilon, 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Training Step', fontsize=10)
        axes[1, 1].set_ylabel('Epsilon', fontsize=10)
        axes[1, 1].set_title('Adaptive Epsilon Decay', fontsize=12)

        axes[1, 2].hist(throughputs, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(np.mean(throughputs), color='red', linestyle='--', linewidth=2)
        axes[1, 2].set_xlabel('Throughput (Mbps)', fontsize=10)
        axes[1, 2].set_ylabel('Frequency', fontsize=10)
        axes[1, 2].set_title('Throughput Distribution', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/marl_training_metrics.pdf', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_zero_shot_results(self, qmix_tp, random_tp, qmix_f, random_f, improvement, p_value):
        """Plot zero-shot evaluation results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        x_pos = [0, 1]

        means_tp = [np.mean(qmix_tp), np.mean(random_tp)]
        stds_tp = [np.std(qmix_tp), np.std(random_tp)]
        bars = axes[0].bar(x_pos, means_tp, yerr=stds_tp, capsize=5, alpha=0.7,
                          color=['#2ecc71', '#95a5a6'], edgecolor='black', linewidth=1.5)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(['QMIX', 'Random'], fontsize=11)
        axes[0].set_ylabel('Throughput (Mbps)', fontsize=12)
        axes[0].set_title(f'Throughput Comparison\nImprovement: {improvement:.1f}%', fontsize=13)
        for bar, mean in zip(bars, means_tp):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

        means_f = [np.mean(qmix_f), np.mean(random_f)]
        stds_f = [np.std(qmix_f), np.std(random_f)]
        bars = axes[1].bar(x_pos, means_f, yerr=stds_f, capsize=5, alpha=0.7,
                          color=['#2ecc71', '#95a5a6'], edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(['QMIX', 'Random'], fontsize=11)
        axes[1].set_ylabel('Fairness Index', fontsize=12)
        axes[1].set_title('Fairness Comparison', fontsize=13)
        axes[1].set_ylim(0, 1)
        for bar, mean in zip(bars, means_f):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

        axes[2].hist(qmix_tp, bins=20, alpha=0.5, label='QMIX', color='green', edgecolor='black')
        axes[2].hist(random_tp, bins=20, alpha=0.5, label='Random', color='gray', edgecolor='black')
        axes[2].set_xlabel('Throughput (Mbps)', fontsize=11)
        axes[2].set_ylabel('Frequency', fontsize=11)
        axes[2].set_title('Throughput Distribution', fontsize=13)
        axes[2].legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/zero_shot_results.pdf', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_interference_robustness(self, results):
        """Plot interference robustness analysis."""
        categories = ['High Interference\n(Low SINR)', 'Medium Interference\n(Medium SINR)', 'Low Interference\n(High SINR)']
        random_vals = [results['high_interference']['random'],
                       results['medium_interference']['random'],
                       results['low_interference']['random']]
        qmix_vals = [results['high_interference']['qmix'],
                     results['medium_interference']['qmix'],
                     results['low_interference']['qmix']]
        improvements = [results['high_interference']['improvement'],
                        results['medium_interference']['improvement'],
                        results['low_interference']['improvement']]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(categories))
        width = 0.35

        axes[0].bar(x - width/2, random_vals, width, label='Random', color='red', alpha=0.7)
        axes[0].bar(x + width/2, qmix_vals, width, label='QMIX', color='blue', alpha=0.7)
        axes[0].set_ylabel('Throughput (Mbps)', fontsize=12)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories, fontsize=10)
        axes[0].legend(fontsize=11)

        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = axes[1].bar(categories, improvements, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Improvement (%)', fontsize=12)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)

        for bar, imp in zip(bars, improvements):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if imp > 0 else -3),
                        f'{imp:.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/interference_robustness.pdf', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_marl_comparison(self, qmix_throughputs, iql_throughputs):
        """Plot QMIX vs IQL comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        algorithms = ['QMIX', 'IQL']
        means = [np.mean(qmix_throughputs), np.mean(iql_throughputs)]
        stds = [np.std(qmix_throughputs), np.std(iql_throughputs)]
        colors = ['#2ecc71', '#3498db']

        bars = axes[0].bar(algorithms, means, yerr=stds, capsize=8, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Throughput (Mbps)', fontsize=12)
        for bar, mean in zip(bars, means):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{mean:.2f}', ha='center', va='bottom', fontsize=11)

        axes[1].hist(qmix_throughputs, bins=30, alpha=0.5, label='QMIX', color='#2ecc71', edgecolor='black')
        axes[1].hist(iql_throughputs, bins=30, alpha=0.5, label='IQL', color='#3498db', edgecolor='black')
        axes[1].set_xlabel('Throughput (Mbps)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/marl_comparison.pdf', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_digital_twin_quality(self, kl_div, wasserstein_dist, feature_names):
        """Plot digital twin quality metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(feature_names))

        colors_kl = ['green' if k < 0.1 else 'yellowgreen' if k < 0.3 else 'orange' if k < 0.7 else 'red' for k in kl_div]
        axes[0].bar(x, kl_div, color=colors_kl, alpha=0.7, edgecolor='black')
        axes[0].axhline(y=0.1, color='green', linestyle='--', label='Excellent (<0.1)')
        axes[0].axhline(y=0.3, color='yellowgreen', linestyle='--', label='Good (<0.3)')
        axes[0].axhline(y=0.7, color='orange', linestyle='--', label='Medium (<0.7)')
        axes[0].set_xlabel('Features', fontsize=11)
        axes[0].set_ylabel('KL Divergence', fontsize=11)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        axes[0].legend(fontsize=8)

        colors_wd = ['green' if w < 0.2 else 'yellowgreen' if w < 0.4 else 'orange' if w < 0.7 else 'red' for w in wasserstein_dist]
        axes[1].bar(x, wasserstein_dist, color=colors_wd, alpha=0.7, edgecolor='black')
        axes[1].axhline(y=0.2, color='green', linestyle='--', label='Excellent (<0.2)')
        axes[1].axhline(y=0.4, color='yellowgreen', linestyle='--', label='Good (<0.4)')
        axes[1].axhline(y=0.7, color='orange', linestyle='--', label='Medium (<0.7)')
        axes[1].set_xlabel('Features', fontsize=11)
        axes[1].set_ylabel('Wasserstein Distance', fontsize=11)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/digital_twin_quality.pdf', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_scalability(self, user_counts, qmix_tp, random_tp, improvements, fairness_qmix, fairness_random):
        """Plot scalability analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(user_counts, qmix_tp, 'bs-', linewidth=2, markersize=8, label='QMIX')
        axes[0, 0].plot(user_counts, random_tp, 'ro-', linewidth=2, markersize=8, label='Random')
        axes[0, 0].set_xlabel('Number of Users', fontsize=12)
        axes[0, 0].set_ylabel('Throughput (Mbps)', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_title('Throughput vs Number of Users', fontsize=13)

        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[0, 1].bar(range(len(user_counts)), improvements, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xticks(range(len(user_counts)))
        axes[0, 1].set_xticklabels(user_counts)
        axes[0, 1].set_xlabel('Number of Users', fontsize=12)
        axes[0, 1].set_ylabel('Improvement (%)', fontsize=12)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0, 1].set_title('Performance Gain', fontsize=13)

        axes[1, 0].plot(user_counts, fairness_qmix, 'bs-', linewidth=2, markersize=8, label='QMIX')
        axes[1, 0].plot(user_counts, fairness_random, 'ro-', linewidth=2, markersize=8, label='Random')
        axes[1, 0].set_xlabel('Number of Users', fontsize=12)
        axes[1, 0].set_ylabel('Fairness Index', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title('Fairness vs Number of Users', fontsize=13)

        axes[1, 1].axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/scalability_analysis.pdf', dpi=150, bbox_inches='tight')
        plt.close()