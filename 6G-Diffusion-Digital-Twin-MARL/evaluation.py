# -*- coding: utf-8 -*-
"""Evaluation functions for zero-shot testing and scalability analysis."""

import numpy as np
from scipy import stats
from environment import SixGNetworkEnv


def evaluate_zero_shot(agent, test_episodes, num_test=40):
    """Perform zero-shot evaluation on unseen scenarios."""
    qmix_throughputs = []
    random_throughputs = []
    qmix_fairness = []
    random_fairness = []
    qmix_rewards = []
    random_rewards = []

    for i in range(min(num_test, len(test_episodes))):
        env = SixGNetworkEnv(scenario="zero_shot_test", seed=i)
        state = env.reset().flatten()
        ep_throughput = []
        ep_fairness = []
        ep_reward = []

        for _ in range(100):
            action = agent.act(state.reshape(3, 5), eval_mode=True)
            next_state, reward, done, _, info = env.step(action)
            ep_throughput.append(np.mean(info['throughput']))
            ep_fairness.append(info['fairness'])
            ep_reward.append(reward)
            state = next_state.flatten()

        qmix_throughputs.append(np.mean(ep_throughput))
        qmix_fairness.append(np.mean(ep_fairness))
        qmix_rewards.append(np.sum(ep_reward))

        env = SixGNetworkEnv(scenario="zero_shot_test", seed=i + 1000)
        state = env.reset().flatten()
        ep_throughput = []
        ep_fairness = []
        ep_reward = []

        for _ in range(100):
            action = np.random.uniform(0, 1, 3)
            next_state, reward, done, _, info = env.step(action)
            ep_throughput.append(np.mean(info['throughput']))
            ep_fairness.append(info['fairness'])
            ep_reward.append(reward)
            state = next_state.flatten()

        random_throughputs.append(np.mean(ep_throughput))
        random_fairness.append(np.mean(ep_fairness))
        random_rewards.append(np.sum(ep_reward))

    qmix_mean = np.mean(qmix_throughputs)
    qmix_std = np.std(qmix_throughputs)
    random_mean = np.mean(random_throughputs)
    random_std = np.std(random_throughputs)
    improvement = (qmix_mean - random_mean) / (random_mean + 1e-8) * 100
    t_stat, p_value = stats.ttest_ind(qmix_throughputs, random_throughputs)

    results = {
        'qmix_throughputs': qmix_throughputs,
        'random_throughputs': random_throughputs,
        'qmix_fairness': qmix_fairness,
        'random_fairness': random_fairness,
        'qmix_rewards': qmix_rewards,
        'random_rewards': random_rewards,
        'qmix_mean': qmix_mean,
        'qmix_std': qmix_std,
        'random_mean': random_mean,
        'random_std': random_std,
        'improvement': improvement,
        'p_value': p_value
    }

    return results


def evaluate_scalability(agent, user_counts, num_episodes=30, scenario="zero_shot_test"):
    """Evaluate agent scalability with increasing number of users."""
    results = {
        'user_counts': user_counts,
        'qmix_throughput': [],
        'random_throughput': [],
        'qmix_fairness': [],
        'random_fairness': [],
        'qmix_sinr': [],
        'random_sinr': [],
        'qmix_reward': [],
        'random_reward': []
    }

    for num_users in user_counts:
        qmix_throughputs = []
        qmix_fairness = []
        qmix_sinrs = []
        qmix_rewards = []

        for ep in range(num_episodes):
            env = SixGNetworkEnv(num_base_stations=3, num_users=num_users,
                                scenario=scenario, seed=ep)
            state = env.reset()
            ep_throughput = []
            ep_fairness = []
            ep_sinr = []
            ep_reward = []

            for step in range(100):
                action = agent.act(state.flatten(), eval_mode=True)
                next_state, reward, done, _, info = env.step(action)
                ep_throughput.append(np.mean(info['throughput']))
                ep_fairness.append(info['fairness'])
                ep_sinr.append(np.mean(info['average_sinr']))
                ep_reward.append(reward)
                state = next_state
                if done:
                    break

            qmix_throughputs.append(np.mean(ep_throughput))
            qmix_fairness.append(np.mean(ep_fairness))
            qmix_sinrs.append(np.mean(ep_sinr))
            qmix_rewards.append(np.sum(ep_reward))

        random_throughputs = []
        random_fairness = []
        random_sinrs = []
        random_rewards = []

        for ep in range(num_episodes):
            env = SixGNetworkEnv(num_base_stations=3, num_users=num_users,
                                scenario=scenario, seed=ep + 1000)
            state = env.reset()
            ep_throughput = []
            ep_fairness = []
            ep_sinr = []
            ep_reward = []

            for step in range(100):
                action = np.random.uniform(0, 1, 3)
                next_state, reward, done, _, info = env.step(action)
                ep_throughput.append(np.mean(info['throughput']))
                ep_fairness.append(info['fairness'])
                ep_sinr.append(np.mean(info['average_sinr']))
                ep_reward.append(reward)
                state = next_state
                if done:
                    break

            random_throughputs.append(np.mean(ep_throughput))
            random_fairness.append(np.mean(ep_fairness))
            random_sinrs.append(np.mean(ep_sinr))
            random_rewards.append(np.sum(ep_reward))

        results['qmix_throughput'].append(np.mean(qmix_throughputs))
        results['random_throughput'].append(np.mean(random_throughputs))
        results['qmix_fairness'].append(np.mean(qmix_fairness))
        results['random_fairness'].append(np.mean(random_fairness))
        results['qmix_sinr'].append(np.mean(qmix_sinrs))
        results['random_sinr'].append(np.mean(random_sinrs))
        results['qmix_reward'].append(np.mean(qmix_rewards))
        results['random_reward'].append(np.mean(random_rewards))

    return results


def evaluate_interference_robustness(agent, test_episodes):
    """Evaluate agent robustness across different interference levels."""
    def get_avg_sinr(episode):
        states = episode['states']
        sinr_values = states[:, :, 0]
        return np.mean(sinr_values)

    episode_sinr = [(i, get_avg_sinr(test_episodes[i])) for i in range(len(test_episodes))]
    episode_sinr.sort(key=lambda x: x[1])

    num_episodes = len(test_episodes)
    low_sinr_episodes = [episode_sinr[i][0] for i in range(num_episodes // 3)]
    mid_sinr_episodes = [episode_sinr[i][0] for i in range(num_episodes // 3, 2 * num_episodes // 3)]
    high_sinr_episodes = [episode_sinr[i][0] for i in range(2 * num_episodes // 3, num_episodes)]

    def evaluate_on_episodes(episode_indices, use_agent=True):
        throughputs = []
        for ep_idx in episode_indices:
            env = SixGNetworkEnv(scenario="zero_shot_test", seed=ep_idx)
            state = env.reset()
            ep_tp = []

            for step in range(100):
                if use_agent:
                    action = agent.act(state.flatten(), eval_mode=True)
                else:
                    action = np.random.uniform(0, 1, 3)
                next_state, _, done, _, info = env.step(action)
                ep_tp.append(np.mean(info['throughput']))
                state = next_state
                if done:
                    break
            throughputs.append(np.mean(ep_tp))
        return np.mean(throughputs), np.std(throughputs)

    random_low, random_low_std = evaluate_on_episodes(low_sinr_episodes, use_agent=False)
    qmix_low, qmix_low_std = evaluate_on_episodes(low_sinr_episodes, use_agent=True)

    random_mid, random_mid_std = evaluate_on_episodes(mid_sinr_episodes, use_agent=False)
    qmix_mid, qmix_mid_std = evaluate_on_episodes(mid_sinr_episodes, use_agent=True)

    random_high, random_high_std = evaluate_on_episodes(high_sinr_episodes, use_agent=False)
    qmix_high, qmix_high_std = evaluate_on_episodes(high_sinr_episodes, use_agent=True)

    results = {
        'high_interference': {
            'random': random_low, 'random_std': random_low_std,
            'qmix': qmix_low, 'qmix_std': qmix_low_std,
            'improvement': (qmix_low - random_low) / random_low * 100
        },
        'medium_interference': {
            'random': random_mid, 'random_std': random_mid_std,
            'qmix': qmix_mid, 'qmix_std': qmix_mid_std,
            'improvement': (qmix_mid - random_mid) / random_mid * 100
        },
        'low_interference': {
            'random': random_high, 'random_std': random_high_std,
            'qmix': qmix_high, 'qmix_std': qmix_high_std,
            'improvement': (qmix_high - random_high) / random_high * 100
        }
    }

    return results