# -*- coding: utf-8 -*-
"""Main execution script for 6G Digital Twin with MARL."""

import numpy as np
import torch
import pickle
import h5py
from tqdm import tqdm

from environment import SixGNetworkEnv
from diffusion_model import train_diffusion_model, generate_diverse_synthetic_states
from qmix_agent import AdvancedQMIXAgent
from dataset_loader import SixGDatasetLoader
from evaluation import evaluate_zero_shot, evaluate_scalability, evaluate_interference_robustness


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_marl_agent(agent, episodes, synthetic_states, num_episodes=250):
    """Train the MARL agent with hybrid buffer."""
    print("\n" + "=" * 50)
    print("Training MARL Agent with Hybrid Buffer")
    print("=" * 50)

    for ep in tqdm(episodes[:150], desc="Adding real experiences"):
        for t in range(len(ep['states']) - 1):
            agent.add_experience(
                ep['states'][t].flatten(), ep['actions'][t], ep['rewards'][t],
                ep['states'][t + 1].flatten(), t == len(ep['states']) - 2,
                is_synthetic=False
            )

    for i in tqdm(range(len(synthetic_states) - 1), desc="Adding synthetic experiences"):
        action = np.random.beta(2, 2, agent.num_agents)
        reward = np.random.normal(0, 2) - 1
        agent.add_experience(
            synthetic_states[i], action, reward, synthetic_states[i + 1],
            False, is_synthetic=True
        )

    episode_rewards = []
    episode_throughputs = []
    episode_fairness = []

    for episode in tqdm(range(num_episodes), desc="Training MARL"):
        env = SixGNetworkEnv(scenario="normal")
        state = env.reset().flatten()
        total_reward = 0
        total_throughput = 0
        total_fairness = 0
        steps = 0

        for _ in range(100):
            action = agent.act(state.reshape(3, 5))
            next_state, reward, done, _, info = env.step(action)
            agent.train_step(batch_size=128)
            total_reward += reward
            total_throughput += np.mean(info['throughput'])
            total_fairness += info['fairness']
            state = next_state.flatten()
            steps += 1
            if done:
                break

        episode_rewards.append(total_reward)
        episode_throughputs.append(total_throughput / steps)
        episode_fairness.append(total_fairness / steps)

    return episode_rewards, episode_throughputs, episode_fairness


def main():
    """Main execution pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    print("\n" + "=" * 60)
    print("6G Diffusion-Driven Digital Twin with MARL")
    print("Zero-Shot Throughput Optimization")
    print("=" * 60)

    loader = SixGDatasetLoader()
    dataset = loader.load_hdf5('./6g_dataset/6g_dataset.h5')
    normal_states = loader.extract_normal_states(dataset)

    diffusion_model, state_mean, state_std = train_diffusion_model(
        normal_states, epochs=35, batch_size=512, device=device
    )

    synthetic_states = generate_diverse_synthetic_states(
        diffusion_model, state_mean, state_std,
        num_samples=8000, scale_factor=12.0, device=device
    )

    agent = AdvancedQMIXAgent(
        num_agents=3, agent_state_dim=5, num_actions=11,
        lr=3e-4, gamma=0.95, epsilon_start=0.5, epsilon_end=0.05,
        epsilon_decay=0.995, n_step=3, device=device
    )

    normal_episodes = dataset['train_normal']
    rewards_history, throughputs_history, fairness_history = train_marl_agent(
        agent, normal_episodes, synthetic_states, num_episodes=250
    )

    test_episodes = dataset['zero_shot_test']
    zero_shot_results = evaluate_zero_shot(agent, test_episodes, num_test=40)

    print("\n" + "=" * 50)
    print("Zero-Shot Test Results")
    print("=" * 50)
    print(f"QMIX Throughput: {zero_shot_results['qmix_mean']:.2f} ± {zero_shot_results['qmix_std']:.2f} Mbps")
    print(f"Random Throughput: {zero_shot_results['random_mean']:.2f} ± {zero_shot_results['random_std']:.2f} Mbps")
    print(f"Improvement: {zero_shot_results['improvement']:.1f}%")
    print(f"p-value: {zero_shot_results['p_value']:.4e}")

    user_counts = [5, 10, 15, 20, 25, 30]
    scalability_results = evaluate_scalability(agent, user_counts, num_episodes=30)

    interference_results = evaluate_interference_robustness(agent, test_episodes)

    torch.save(diffusion_model.state_dict(), 'advanced_diffusion_model.pth')
    torch.save(agent.qmix.state_dict(), 'advanced_qmix_model.pth')

    history = {
        'rewards': rewards_history,
        'throughputs': throughputs_history,
        'fairness': fairness_history,
        'losses': agent.training_losses,
        'epsilon_history': agent.epsilon_history,
        'zero_shot_results': zero_shot_results,
        'scalability_results': scalability_results,
        'interference_results': interference_results
    }
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    print("\n" + "=" * 50)
    print("Execution Complete")
    print("=" * 50)
    print("Files saved:")
    print("  - advanced_diffusion_model.pth")
    print("  - advanced_qmix_model.pth")
    print("  - training_history.pkl")
    print(f"\nBest Zero-Shot Throughput Improvement: {zero_shot_results['improvement']:.1f}%")


if __name__ == "__main__":
    main()