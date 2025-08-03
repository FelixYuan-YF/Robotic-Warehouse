import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import pandas as pd

from dqn import QLearningAgent
from make_env import load_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN agent in a multi-agent environment.")
    # werahouse related arguments
    parser.add_argument('--shelf_columns', type=int,
                        default=3, help='Number of shelf columns')
    parser.add_argument('--column_height', type=int,
                        default=8, help='Height of each shelf column')
    parser.add_argument('--shelf_rows', type=int, default=1,
                        help='Number of rows in the shelf')
    parser.add_argument('--n_agents', type=int, default=2,
                        help='Number of agents in the environment')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')
    # Training related arguments
    parser.add_argument('--num_episodes', type=int,
                        default=5000, help='Number of training episodes')
    parser.add_argument('--save_dir', type=str, default='saved/dqn',
                        help='Path to save the output models and rewards')
    return parser.parse_args()


def ema_filter(data, window=10):
    """Calculate Exponential Moving Average to smooth data."""
    return pd.Series(data).ewm(span=window, adjust=False).mean().values


def plot_stats(episode_rewards, save_path):
    plt.figure(figsize=(10, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in episode_rewards:
        smoothed_returns = ema_filter(episode_rewards[i], window=10)
        plt.plot(episode_rewards[i], alpha=0.3, color=colors[i % len(colors)])
        plt.plot(smoothed_returns, color=colors[i % len(
            colors)], label=f'Agent {i + 1}')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards (DQN) | Smooth: ema (window=10)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved at {save_path}")


def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    env, num_agents, state_size, action_size = load_env(
        shelf_columns=args.shelf_columns,
        column_height=args.column_height,
        shelf_rows=args.shelf_rows,
        n_agents=args.n_agents,
        max_steps=args.max_steps
    )
    agents = [QLearningAgent(state_size, action_size)
              for _ in range(num_agents)]
    episode_rewards = {i: [] for i in range(num_agents)}

    pbar = tqdm(total=args.num_episodes,
                desc="Training Progress", unit="episode")

    for episode in range(args.num_episodes):
        state = env.reset()[0]
        episode_reward = np.zeros(num_agents)

        for step in range(env.max_steps):
            actions = [agent.select_action(state[i])
                       for i, agent in enumerate(agents)]
            next_state, reward, dones, _ = env.step(actions)

            for i in range(num_agents):
                agents[i].store_transition(
                    state[i], actions[i], reward[i], next_state[i], dones[i])
                agents[i].train_step()
                episode_reward[i] += reward[i]

            state = next_state

            if any(dones):
                break

        for i in range(num_agents):
            episode_rewards[i].append(episode_reward[i])

        pbar.set_postfix(
            {f'Agent {i:.2f}': episode_reward[i] for i in range(num_agents)})
        pbar.update(1)

        # Save intermediate results every 100 episodes
        if (episode + 1) % 100 == 0:
            for i in range(num_agents):
                agents[i].save_model(os.path.join(
                    args.save_dir, f'agent_{i}.pth'))
            np.save(os.path.join(args.save_dir,
                    f'rewards_episode_{episode+1}.npy'), episode_rewards)
            print(f"\nIntermediate results saved (Episode {episode+1})")

    pbar.close()

    # Save final models and results
    for i in range(num_agents):
        agents[i].save_model(os.path.join(args.save_dir, f'agent_{i}.pth'))
    np.save(os.path.join(args.save_dir, 'rewards.npy'), episode_rewards)

    # Generate training result plots
    plot_stats(episode_rewards, os.path.join(
        args.save_dir, 'training_rewards.png'))

    # Print training statistics
    print(f"\nTraining completed!")
    print(
        f"Average rewards: {[np.mean(rewards) for rewards in episode_rewards]}")
    print(f"Max rewards: {[np.max(rewards) for rewards in episode_rewards]}")


if __name__ == "__main__":
    main()
