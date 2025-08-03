import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm

from maddpg import MADDPG
from make_env import load_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a trained MADDPG agent in a multi-agent environment.")
    # Environment related arguments
    parser.add_argument('--shelf_columns', type=int, default=3, help='Number of shelf columns')
    parser.add_argument('--column_height', type=int, default=8, help='Height of each shelf column')
    parser.add_argument('--shelf_rows', type=int, default=1, help='Number of rows in the shelf')
    parser.add_argument('--n_agents', type=int, default=2, help='Number of agents in the environment')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    # Testing related arguments
    parser.add_argument('--model_dir', type=str, default='saved/maddpg',
                        help='Path to load the trained models')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during testing')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between steps when rendering (seconds)')
    parser.add_argument('--save_gif', action='store_true',
                        help='Save test episodes as GIF (requires render=True)')
    parser.add_argument('--gif_episodes', type=int, default=5,
                        help='Number of episodes to save as GIF')
    return parser.parse_args()


def discrete_action_from_continuous(continuous_action, num_discrete_actions):
    """Convert continuous actions to discrete actions"""
    # Use softmax probability distribution to select actions
    if len(continuous_action.shape) > 0:
        # Apply softmax to continuous actions
        probs = np.exp(continuous_action) / np.sum(np.exp(continuous_action))
        # Select action according to probability (deterministic for testing)
        discrete_action = np.argmax(probs)
    else:
        # For single-value case
        discrete_action = int(np.clip(continuous_action, 0, num_discrete_actions - 1))
    
    return discrete_action


def test_agents(env, agent, num_episodes, render=False, delay=0.1, save_gif=False, gif_episodes=5, model_dir=None):
    """Test the trained MADDPG agents."""
    num_agents = agent.num_agents
    episode_rewards = [[] for _ in range(num_agents)]
    episode_lengths = []
    success_count = 0
    
    frames = [] if save_gif else None
    gif_episode_count = 0
    
    print(f"Testing MADDPG agents for {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes), desc="Testing Progress"):
        states, _ = env.reset()
        episode_reward = [0 for _ in range(num_agents)]
        episode_length = 0
        
        # Record frames for GIF
        if save_gif and gif_episode_count < gif_episodes:
            if hasattr(env, 'render'):
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        
        for step in range(env.max_steps):
            # Select actions (no noise during testing)
            continuous_actions = agent.act(states, add_noise=False)
            
            # Convert continuous actions to discrete actions
            discrete_actions = []
            for i, cont_action in enumerate(continuous_actions):
                discrete_action = discrete_action_from_continuous(cont_action, env.action_space[0].n)
                discrete_actions.append(discrete_action)
            
            # Execute actions
            next_states, rewards, dones, info = env.step(discrete_actions)
            
            # Accumulate rewards
            for i in range(num_agents):
                episode_reward[i] += rewards[i]
            
            states = next_states
            episode_length += 1
            
            # Render if requested
            if render:
                if hasattr(env, 'render'):
                    env.render()
                time.sleep(delay)
            
            # Record frames for GIF
            if save_gif and gif_episode_count < gif_episodes:
                if hasattr(env, 'render'):
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
            
            # Check if episode is done
            if any(dones):
                # Check for success (you may need to adjust this condition based on your environment)
                if sum(episode_reward) > 0:  # Simple success criterion
                    success_count += 1
                break
        
        # Record episode results
        for i in range(num_agents):
            episode_rewards[i].append(episode_reward[i])
        episode_lengths.append(episode_length)
        
        if save_gif and gif_episode_count < gif_episodes:
            gif_episode_count += 1
    
    # Save GIF if requested
    if save_gif and frames:
        try:
            import imageio
            gif_path = os.path.join(model_dir, 'test_episodes.gif') if model_dir else 'test_episodes.gif'
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"GIF saved to {gif_path}")
        except ImportError:
            print("Warning: imageio not installed, cannot save GIF")
    
    return episode_rewards, episode_lengths, success_count


def print_statistics(episode_rewards, episode_lengths, success_count, num_episodes):
    """Print test statistics."""
    print("\n" + "="*60)
    print("MADDPG TEST RESULTS")
    print("="*60)
    
    print(f"Total episodes: {num_episodes}")
    print(f"Successful episodes: {success_count} ({success_count/num_episodes*100:.1f}%)")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print()
    
    print("Agent Performance:")
    for i in range(len(episode_rewards)):
        rewards = episode_rewards[i]
        print(f"  Agent {i}:")
        print(f"    Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"    Max reward: {np.max(rewards):.3f}")
        print(f"    Min reward: {np.min(rewards):.3f}")
    
    total_rewards = [sum(episode_rewards[i][ep] for i in range(len(episode_rewards))) 
                    for ep in range(num_episodes)]
    print(f"\nTeam Performance:")
    print(f"  Mean total reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"  Max total reward: {np.max(total_rewards):.3f}")
    print(f"  Min total reward: {np.min(total_rewards):.3f}")


def plot_test_results(episode_rewards, episode_lengths, save_dir):
    """Plot test results."""
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'test_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot rewards
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Individual agent rewards
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(len(episode_rewards)):
        ax1.plot(episode_rewards[i], color=colors[i % len(colors)], 
                label=f'Agent {i}', alpha=0.7)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('MADDPG Test Results - Individual Agent Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths
    ax2.plot(episode_lengths, color='purple', alpha=0.7)
    ax2.axhline(y=np.mean(episode_lengths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'test_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    total_rewards = [sum(episode_rewards[i][ep] for i in range(len(episode_rewards))) 
                    for ep in range(len(episode_rewards[0]))]
    
    plt.hist(total_rewards, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(x=np.mean(total_rewards), color='red', linestyle='--', 
               label=f'Mean: {np.mean(total_rewards):.2f}')
    plt.xlabel('Total Team Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Team Rewards (MADDPG)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'reward_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test plots saved to {plots_dir}")


def compare_agents_performance(episode_rewards, save_dir):
    """Create additional analysis plots for MADDPG agents."""
    plots_dir = os.path.join(save_dir, 'test_plots')
    
    # Agent comparison boxplot
    plt.figure(figsize=(10, 6))
    data_to_plot = [episode_rewards[i] for i in range(len(episode_rewards))]
    labels = [f'Agent {i}' for i in range(len(episode_rewards))]
    
    box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
    
    plt.ylabel('Reward')
    plt.title('Agent Performance Comparison (MADDPG)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'agent_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cumulative rewards
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(len(episode_rewards)):
        cumulative_rewards = np.cumsum(episode_rewards[i])
        plt.plot(cumulative_rewards, color=colors[i % len(colors)], 
                label=f'Agent {i}', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Episodes (MADDPG)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'cumulative_rewards.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' does not exist!")
        return
    
    # Load environment
    render_mode = 'human' if args.render else 'rgb_array'
    env, num_agents, state_dim, action_dim = load_env(
        shelf_columns=args.shelf_columns,
        column_height=args.column_height,
        shelf_rows=args.shelf_rows,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        render_mode=render_mode
    )
    
    # Create MADDPG agent
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
    
    # Load trained models
    try:
        agent.load_models(args.model_dir)
        print(f"Successfully loaded MADDPG models from {args.model_dir}")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    print(f"\nStarting MADDPG test with {num_agents} agents")
    print(f"Environment: {args.shelf_columns}x{args.shelf_rows} warehouse, {args.column_height} height")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Max steps per episode: {args.max_steps}")
    
    # Test the agents
    episode_rewards, episode_lengths, success_count = test_agents(
        env, agent, args.num_episodes, 
        render=args.render, delay=args.delay,
        save_gif=args.save_gif, gif_episodes=args.gif_episodes,
        model_dir=args.model_dir
    )
    
    # Print statistics
    print_statistics(episode_rewards, episode_lengths, success_count, args.num_episodes)
    
    # Plot results
    plot_test_results(episode_rewards, episode_lengths, args.model_dir)
    compare_agents_performance(episode_rewards, args.model_dir)
    
    # Save test results
    results_file = os.path.join(args.model_dir, 'test_results.npz')
    np.savez(results_file, 
             episode_rewards=episode_rewards, 
             episode_lengths=episode_lengths,
             success_count=success_count,
             num_episodes=args.num_episodes)
    print(f"\nTest results saved to {results_file}")


if __name__ == "__main__":
    main()
