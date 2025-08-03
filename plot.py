import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 用于更方便计算EMA

def ema_filter(data, window=10):
    """Calculate Exponential Moving Average to smooth data."""
    return pd.Series(data).ewm(span=window, adjust=False).mean().values

def plot_stats(episode_rewards, save_path):
    plt.figure(figsize=(10, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(len(episode_rewards)):
        smoothed_returns = ema_filter(episode_rewards[i], window=10)
        plt.plot(episode_rewards[i], alpha=0.3, color=colors[i % len(colors)])
        plt.plot(smoothed_returns, color=colors[i % len(colors)], label=f'Agent {i + 1}')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards (MADDPG) | Smooth: ema (window=10)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved at {save_path}")

# episode_rewards = np.load('saved/dqn/rewards.npy')
# plot_stats(episode_rewards, save_path='saved/dqn/reward_plot.png')
episode_rewards = np.load('saved/dqn/rewards.npy', allow_pickle=True).item()
# convert to list of lists for plotting
episode_rewards = [episode_rewards[i] for i in range(len(episode_rewards))]
# convert np.float64() to float for plotting
episode_rewards = [[float(reward) for reward in episode] for episode in episode_rewards]
np.save('saved/dqn/1rewards.npy', episode_rewards)  # Save the processed rewards
plot_stats(episode_rewards, save_path='saved/dqn/reward_plot.png')