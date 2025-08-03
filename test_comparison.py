#!/usr/bin/env python3
"""
通用测试脚本，用于测试训练好的DQN和MADDPG模型
支持对比两种算法的性能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm

from dqn import QLearningAgent
from maddpg import MADDPG
from make_env import load_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test and compare DQN and MADDPG agents.")
    # Environment related arguments
    parser.add_argument('--shelf_columns', type=int, default=3, help='Number of shelf columns')
    parser.add_argument('--column_height', type=int, default=8, help='Height of each shelf column')
    parser.add_argument('--shelf_rows', type=int, default=1, help='Number of rows in the shelf')
    parser.add_argument('--n_agents', type=int, default=2, help='Number of agents in the environment')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    # Testing related arguments
    parser.add_argument('--dqn_model_dir', type=str, default='saved/dqn',
                        help='Path to DQN trained models')
    parser.add_argument('--maddpg_model_dir', type=str, default='saved/maddpg',
                        help='Path to MADDPG trained models')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'maddpg', 'both'], default='both',
                        help='Which algorithm to test')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during testing')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between steps when rendering (seconds)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save comparison results')
    return parser.parse_args()


def discrete_action_from_continuous(continuous_action, num_discrete_actions):
    """Convert continuous actions to discrete actions for MADDPG"""
    if len(continuous_action.shape) > 0:
        probs = np.exp(continuous_action) / np.sum(np.exp(continuous_action))
        discrete_action = np.argmax(probs)  # Deterministic for testing
    else:
        discrete_action = int(np.clip(continuous_action, 0, num_discrete_actions - 1))
    return discrete_action


def test_dqn_agents(env, agents, num_episodes):
    """Test DQN agents."""
    num_agents = len(agents)
    episode_rewards = {i: [] for i in range(num_agents)}
    episode_lengths = []
    success_count = 0
    
    print("Testing DQN agents...")
    for episode in tqdm(range(num_episodes), desc="DQN Testing"):
        state = env.reset()[0]
        episode_reward = np.zeros(num_agents)
        episode_length = 0
        
        for step in range(env.max_steps):
            # Select actions (no exploration during testing)
            actions = []
            for i, agent in enumerate(agents):
                old_epsilon = agent.epsilon
                agent.epsilon = 0.0  # Greedy policy
                action = agent.select_action(state[i])
                agent.epsilon = old_epsilon
                actions.append(action)
            
            next_state, reward, dones, _ = env.step(actions)
            
            for i in range(num_agents):
                episode_reward[i] += reward[i]
            
            state = next_state
            episode_length += 1
            
            if any(dones):
                if sum(episode_reward) > 0:
                    success_count += 1
                break
        
        for i in range(num_agents):
            episode_rewards[i].append(episode_reward[i])
        episode_lengths.append(episode_length)
    
    return episode_rewards, episode_lengths, success_count


def test_maddpg_agent(env, agent, num_episodes):
    """Test MADDPG agent."""
    num_agents = agent.num_agents
    episode_rewards = [[] for _ in range(num_agents)]
    episode_lengths = []
    success_count = 0
    
    print("Testing MADDPG agents...")
    for episode in tqdm(range(num_episodes), desc="MADDPG Testing"):
        states, _ = env.reset()
        episode_reward = [0 for _ in range(num_agents)]
        episode_length = 0
        
        for step in range(env.max_steps):
            # Select actions (no noise during testing)
            continuous_actions = agent.act(states, add_noise=False)
            
            # Convert to discrete actions
            discrete_actions = []
            for i, cont_action in enumerate(continuous_actions):
                discrete_action = discrete_action_from_continuous(cont_action, env.action_space[0].n)
                discrete_actions.append(discrete_action)
            
            next_states, rewards, dones, info = env.step(discrete_actions)
            
            for i in range(num_agents):
                episode_reward[i] += rewards[i]
            
            states = next_states
            episode_length += 1
            
            if any(dones):
                if sum(episode_reward) > 0:
                    success_count += 1
                break
        
        for i in range(num_agents):
            episode_rewards[i].append(episode_reward[i])
        episode_lengths.append(episode_length)
    
    return episode_rewards, episode_lengths, success_count


def print_comparison_statistics(dqn_results=None, maddpg_results=None):
    """Print comparison statistics between algorithms."""
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*80)
    
    if dqn_results:
        dqn_rewards, dqn_lengths, dqn_success = dqn_results
        dqn_total_rewards = [sum(dqn_rewards[i][ep] for i in range(len(dqn_rewards))) 
                            for ep in range(len(dqn_rewards[0]))]
        
        print("\nDQN Results:")
        print(f"  Success rate: {dqn_success/len(dqn_total_rewards)*100:.1f}%")
        print(f"  Mean total reward: {np.mean(dqn_total_rewards):.3f} ± {np.std(dqn_total_rewards):.3f}")
        print(f"  Mean episode length: {np.mean(dqn_lengths):.2f} ± {np.std(dqn_lengths):.2f}")
    
    if maddpg_results:
        maddpg_rewards, maddpg_lengths, maddpg_success = maddpg_results
        maddpg_total_rewards = [sum(maddpg_rewards[i][ep] for i in range(len(maddpg_rewards))) 
                               for ep in range(len(maddpg_rewards[0]))]
        
        print("\nMADDPG Results:")
        print(f"  Success rate: {maddpg_success/len(maddpg_total_rewards)*100:.1f}%")
        print(f"  Mean total reward: {np.mean(maddpg_total_rewards):.3f} ± {np.std(maddpg_total_rewards):.3f}")
        print(f"  Mean episode length: {np.mean(maddpg_lengths):.2f} ± {np.std(maddpg_lengths):.2f}")
    
    if dqn_results and maddpg_results:
        print("\nComparison:")
        dqn_total_rewards = [sum(dqn_rewards[i][ep] for i in range(len(dqn_rewards))) 
                            for ep in range(len(dqn_rewards[0]))]
        maddpg_total_rewards = [sum(maddpg_rewards[i][ep] for i in range(len(maddpg_rewards))) 
                               for ep in range(len(maddpg_rewards[0]))]
        
        reward_diff = np.mean(maddpg_total_rewards) - np.mean(dqn_total_rewards)
        success_diff = maddpg_success - dqn_success
        length_diff = np.mean(maddpg_lengths) - np.mean(dqn_lengths)
        
        print(f"  Reward difference (MADDPG - DQN): {reward_diff:.3f}")
        print(f"  Success difference (MADDPG - DQN): {success_diff} episodes")
        print(f"  Episode length difference (MADDPG - DQN): {length_diff:.2f} steps")
        
        better = "MADDPG" if reward_diff > 0 else "DQN"
        print(f"  Better performing algorithm: {better}")


def plot_comparison_results(dqn_results=None, maddpg_results=None, save_dir=None):
    """Plot comparison results between algorithms."""
    if not save_dir:
        save_dir = "comparison_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data
    algorithms = []
    all_rewards = []
    all_lengths = []
    labels = []
    colors = ['skyblue', 'lightcoral']
    
    if dqn_results:
        dqn_rewards, dqn_lengths, _ = dqn_results
        dqn_total_rewards = [sum(dqn_rewards[i][ep] for i in range(len(dqn_rewards))) 
                            for ep in range(len(dqn_rewards[0]))]
        algorithms.append("DQN")
        all_rewards.append(dqn_total_rewards)
        all_lengths.append(dqn_lengths)
        labels.append("DQN")
    
    if maddpg_results:
        maddpg_rewards, maddpg_lengths, _ = maddpg_results
        maddpg_total_rewards = [sum(maddpg_rewards[i][ep] for i in range(len(maddpg_rewards))) 
                               for ep in range(len(maddpg_rewards[0]))]
        algorithms.append("MADDPG")
        all_rewards.append(maddpg_total_rewards)
        all_lengths.append(maddpg_lengths)
        labels.append("MADDPG")
    
    if len(algorithms) == 0:
        return
    
    # Plot reward comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Boxplot comparison
    if len(algorithms) > 1:
        ax1.boxplot(all_rewards, labels=labels, patch_artist=True, 
                   boxprops=dict(facecolor=colors[0]), 
                   medianprops=dict(color='red', linewidth=2))
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Algorithm Performance Comparison')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.hist(all_rewards[0], bins=20, alpha=0.7, color=colors[0], edgecolor='black')
        ax1.axvline(x=np.mean(all_rewards[0]), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_rewards[0]):.2f}')
        ax1.set_xlabel('Total Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{labels[0]} Performance Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Episode length comparison
    if len(algorithms) > 1:
        ax2.boxplot(all_lengths, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor=colors[1]),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length Comparison')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.hist(all_lengths[0], bins=20, alpha=0.7, color=colors[1], edgecolor='black')
        ax2.axvline(x=np.mean(all_lengths[0]), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_lengths[0]):.1f}')
        ax2.set_xlabel('Episode Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{labels[0]} Episode Length Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual episode comparison if both algorithms tested
    if len(algorithms) == 2:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Reward progression
        plt.subplot(2, 1, 1)
        for i, (rewards, label, color) in enumerate(zip(all_rewards, labels, colors)):
            plt.plot(rewards, alpha=0.7, color=color, label=label)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Episode length progression
        plt.subplot(2, 1, 2)
        for i, (lengths, label, color) in enumerate(zip(all_lengths, labels, colors)):
            plt.plot(lengths, alpha=0.7, color=color, label=label)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Length Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'progress_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Comparison plots saved to {save_dir}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load environment
    render_mode = 'human' if args.render else 'rgb_array'
    env, num_agents, state_size, action_size = load_env(
        shelf_columns=args.shelf_columns,
        column_height=args.column_height,
        shelf_rows=args.shelf_rows,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        render_mode=render_mode
    )
    
    print(f"Environment setup: {num_agents} agents, {state_size}D state, {action_size} actions")
    
    dqn_results = None
    maddpg_results = None
    
    # Test DQN if requested
    if args.algorithm in ['dqn', 'both']:
        if os.path.exists(args.dqn_model_dir):
            try:
                # Load DQN agents
                dqn_agents = []
                for i in range(num_agents):
                    agent = QLearningAgent(state_size, action_size)
                    model_path = os.path.join(args.dqn_model_dir, f'agent_{i}.pth')
                    if os.path.exists(model_path):
                        agent.load_model(model_path)
                        dqn_agents.append(agent)
                    else:
                        print(f"Warning: DQN model file '{model_path}' not found!")
                        break
                
                if len(dqn_agents) == num_agents:
                    dqn_results = test_dqn_agents(env, dqn_agents, args.num_episodes)
                    print("DQN testing completed successfully!")
                
            except Exception as e:
                print(f"Error testing DQN: {e}")
        else:
            print(f"DQN model directory '{args.dqn_model_dir}' not found!")
    
    # Test MADDPG if requested
    if args.algorithm in ['maddpg', 'both']:
        if os.path.exists(args.maddpg_model_dir):
            try:
                # Load MADDPG agent
                maddpg_agent = MADDPG(
                    num_agents=num_agents,
                    state_dim=state_size,
                    action_dim=action_size,
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
                
                maddpg_agent.load_models(args.maddpg_model_dir)
                maddpg_results = test_maddpg_agent(env, maddpg_agent, args.num_episodes)
                print("MADDPG testing completed successfully!")
                
            except Exception as e:
                print(f"Error testing MADDPG: {e}")
        else:
            print(f"MADDPG model directory '{args.maddpg_model_dir}' not found!")
    
    # Print and save results
    print_comparison_statistics(dqn_results, maddpg_results)
    plot_comparison_results(dqn_results, maddpg_results, args.output_dir)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'comparison_results.npz')
    save_data = {}
    if dqn_results:
        save_data['dqn_rewards'] = dqn_results[0]
        save_data['dqn_lengths'] = dqn_results[1]
        save_data['dqn_success'] = dqn_results[2]
    if maddpg_results:
        save_data['maddpg_rewards'] = maddpg_results[0]
        save_data['maddpg_lengths'] = maddpg_results[1]
        save_data['maddpg_success'] = maddpg_results[2]
    
    np.savez(results_file, **save_data)
    print(f"\nComparison results saved to {results_file}")


if __name__ == "__main__":
    main()
