import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import pandas as pd

from maddpg import MADDPG
from make_env import load_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MADDPG agent in a multi-agent environment.")
    # Warehouse related arguments
    parser.add_argument('--shelf_columns', type=int, default=3, help='Number of shelf columns')
    parser.add_argument('--column_height', type=int, default=8, help='Height of each shelf column')
    parser.add_argument('--shelf_rows', type=int, default=1, help='Number of rows in the shelf')
    parser.add_argument('--n_agents', type=int, default=2, help='Number of agents in the environment')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--render_mode', type=str, default='rgb_array', choices=['human', 'rgb_array'], help='Render mode for the environment')
    # Training related arguments
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--save_dir', type=str, default='saved/maddpg', help='Path to save the output models and rewards')
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
        plt.plot(smoothed_returns, color=colors[i % len(colors)], label=f'Agent {i + 1}')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards (MADDPG) | Smooth: ema (window=10)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved at {save_path}")

def discrete_action_from_continuous(continuous_action, num_discrete_actions):
    """Convert continuous actions to discrete actions"""
    # Use softmax probability distribution to select actions
    if len(continuous_action.shape) > 0:
        # Apply softmax to continuous actions
        probs = np.exp(continuous_action) / np.sum(np.exp(continuous_action))
        # Select action according to probability
        discrete_action = np.random.choice(num_discrete_actions, p=probs)
    else:
        # For single-value case
        discrete_action = int(np.clip(continuous_action, 0, num_discrete_actions - 1))
    
    return discrete_action

def main():
    args = parse_args()
    env, num_agents, state_dim, action_dim = load_env(
        shelf_columns=args.shelf_columns,
        column_height=args.column_height,
        shelf_rows=args.shelf_rows,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        render_mode=args.render_mode
    )

    # Create MADDPG agents
    agent = MADDPG(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        max_action=1.0,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.98,
        tau=0.005,
        buffer_size=int(1e6),
        batch_size=128,
        discrete_action=True
    )

    episode_rewards = [[] for _ in range(num_agents)]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Starting MADDPG training, Environment: {num_agents} agents, State dimension: {state_dim}, Action dimension: {action_dim}")
    
    pbar = tqdm(total=args.num_episodes, desc="Training Progress", unit="episode")
    
    for episode in range(args.num_episodes):
        states, _ = env.reset()
        episode_reward = [0 for _ in range(num_agents)]
        
        # Reset noise
        agent.reset_noise()
        
        # Calculate exploration rate (decays over training)
        exploration_rate = max(0.05, 1.0 - episode / (args.num_episodes * 0.8))
        
        for step in range(args.max_steps):
            # Select actions
            continuous_actions = agent.act(states, add_noise=True, noise_scale=exploration_rate)
            
            # Convert continuous actions to discrete actions
            discrete_actions = []
            for i, cont_action in enumerate(continuous_actions):
                discrete_action = discrete_action_from_continuous(cont_action, action_dim)
                discrete_actions.append(discrete_action)
            
            # Execute actions
            next_states, rewards, dones, info = env.step(discrete_actions)
            
            # Store experience
            agent.step(states, continuous_actions, rewards, next_states, dones)
            
            # Update states
            states = next_states
            
            # Accumulate rewards
            for i in range(num_agents):
                episode_reward[i] += rewards[i]
            
            # Check if episode is done
            if any(dones):
                break
        
        # Record episode results
        for i in range(num_agents):
            episode_rewards[i].append(episode_reward[i])
        
        # Update progress bar
        pbar.set_postfix({f'Agent {i:.2f}': episode_reward[i] for i in range(num_agents)})
        pbar.update(1)
        
        # Save intermediate results every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.save_models(args.save_dir)
            np.save(os.path.join(args.save_dir, f'rewards_episode_{episode+1}.npy'), episode_rewards)
            print(f"\nIntermediate results saved (Episode {episode+1})")

    pbar.close()

    # Save final models and results
    agent.save_models(args.save_dir)
    np.save(os.path.join(args.save_dir, 'rewards.npy'), episode_rewards)

    # Generate training result plots
    plot_stats(episode_rewards, save_path=os.path.join(args.save_dir, 'training_rewards.png'))
    
    # Print training statistics
    print(f"\nTraining completed!")
    print(f"Average rewards: {[np.mean(rewards) for rewards in episode_rewards]}")
    print(f"Max rewards: {[np.max(rewards) for rewards in episode_rewards]}")

if __name__ == "__main__":
    main()