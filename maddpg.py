import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc4.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x * self.max_action

class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc4.weight.data.uniform_(-0.003, 0.003)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class OUNoise:
    """Ornstein-Uhlenbeck process for adding noise to actions"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        if len(self.buffer) < self.batch_size:
            return None
            
        batch = random.sample(self.buffer, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return (
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def size(self):
        return len(self.buffer)

class MADDPG:
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=64, max_action=1.0, 
                 actor_lr=0.001, critic_lr=0.001, gamma=0.95, tau=0.01, 
                 buffer_size=100000, batch_size=64, discrete_action=True):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.discrete_action = discrete_action
        
        # Create Actor and Critic networks for each agent
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        self.noise = []

        self.step_count = 0
        
        for i in range(num_agents):
            # Actor network
            actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
            target_actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
            
            # Critic network (inputs are states and actions of all agents)
            total_state_dim = state_dim * num_agents
            total_action_dim = action_dim * num_agents
            critic = Critic(total_state_dim, total_action_dim, hidden_dim).to(device)
            target_critic = Critic(total_state_dim, total_action_dim, hidden_dim).to(device)
            
            # Optimizers
            actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
            critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
            
            # Noise
            noise = OUNoise(action_dim)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
            self.noise.append(noise)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        
        # Initialize target networks
        self._initialize_target_networks()

    def _initialize_target_networks(self):
        """Initialize target network parameters"""
        for i in range(self.num_agents):
            self._hard_update(self.target_actors[i], self.actors[i])
            self._hard_update(self.target_critics[i], self.critics[i])

    def _hard_update(self, target, source):
        """Hard update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def _soft_update(self, target, source, tau):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def act(self, states, add_noise=True, noise_scale=0.1):
        """Select actions"""
        actions = []
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
            
            # Set to evaluation mode
            self.actors[i].eval()
            with torch.no_grad():
                action = self.actors[i](state).cpu().data.numpy()[0]
            self.actors[i].train()
            
            # Add noise for exploration
            if add_noise:
                if self.discrete_action:
                    # For discrete actions, add Gaussian noise
                    noise = np.random.normal(0, noise_scale, size=action.shape)
                    action = action + noise
                else:
                    # Use OU noise
                    noise = self.noise[i].sample()
                    action = action + noise_scale * noise
            
            # Ensure action is within valid range
            action = np.clip(action, -self.max_action, self.max_action)
            actions.append(action)
        
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Store experience and train"""
        # Store experience
        self.replay_buffer.add((states, actions, rewards, next_states, dones))
        
        # If enough experience, train
        self.step_count += 1
        if self.replay_buffer.size() >= self.batch_size and self.step_count % 10 == 0:
            self.train()

    def train(self):
        """Train all agents"""
        # Sample experience
        batch = self.replay_buffer.sample()
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = batch
        
        # Update networks for each agent
        for agent_idx in range(self.num_agents):
            self._update_agent(agent_idx, states, actions, rewards, next_states, dones)
        
        # Soft update target networks
        self._update_target_networks()

    def _update_agent(self, agent_idx, states, actions, rewards, next_states, dones):
        """Update networks for a single agent"""
        batch_size = states.shape[0]
        
        # Reshape states and actions
        states_all = states.view(batch_size, -1)  # (batch_size, num_agents * state_dim)
        next_states_all = next_states.view(batch_size, -1)
        actions_all = actions.view(batch_size, -1)  # (batch_size, num_agents * action_dim)
        
        # Update Critic
        with torch.no_grad():
            # Compute target actions
            target_actions = []
            for i in range(self.num_agents):
                target_action = self.target_actors[i](next_states[:, i])
                target_actions.append(target_action)
            target_actions_all = torch.cat(target_actions, dim=1)
            
            # Compute target Q value
            target_q = self.target_critics[agent_idx](next_states_all, target_actions_all)
            target_q = rewards[:, agent_idx].unsqueeze(1) + \
                      self.gamma * target_q * (1 - dones[:, agent_idx].unsqueeze(1))
        
        # Current Q value
        current_q = self.critics[agent_idx](states_all, actions_all)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update Critic
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
        self.critic_optimizers[agent_idx].step()
        
        # Update Actor
        # Compute actor's actions
        policy_actions = []
        for i in range(self.num_agents):
            if i == agent_idx:
                policy_action = self.actors[i](states[:, i])
            else:
                policy_action = actions[:, i].detach()
            policy_actions.append(policy_action)
        policy_actions_all = torch.cat(policy_actions, dim=1)
        
        # Actor loss
        actor_loss = -self.critics[agent_idx](states_all, policy_actions_all).mean()
        
        # Update Actor
        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
        self.actor_optimizers[agent_idx].step()

    def _update_target_networks(self):
        """Soft update all target networks"""
        for i in range(self.num_agents):
            self._soft_update(self.target_actors[i], self.actors[i], self.tau)
            self._soft_update(self.target_critics[i], self.critics[i], self.tau)

    def reset_noise(self):
        """Reset noise"""
        for noise in self.noise:
            noise.reset()

    def save_models(self, save_dir):
        """Save models"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), 
                      os.path.join(save_dir, f'actor_{i}.pth'))
            torch.save(self.critics[i].state_dict(), 
                      os.path.join(save_dir, f'critic_{i}.pth'))
        
        print(f"Models saved to {save_dir}")

    def load_models(self, load_dir):
        """Load models"""
        for i in range(self.num_agents):
            actor_path = os.path.join(load_dir, f'actor_{i}.pth')
            critic_path = os.path.join(load_dir, f'critic_{i}.pth')
            
            if os.path.exists(actor_path):
                self.actors[i].load_state_dict(torch.load(actor_path, map_location=device))
                self.target_actors[i].load_state_dict(torch.load(actor_path, map_location=device))
            
            if os.path.exists(critic_path):
                self.critics[i].load_state_dict(torch.load(critic_path, map_location=device))
                self.target_critics[i].load_state_dict(torch.load(critic_path, map_location=device))
        
        print(f"Models loaded from {load_dir}")

    def remember(self, states, actions, rewards, next_states, dones):
        """Add experience to replay buffer (maintain compatibility with old interface)"""
        self.replay_buffer.add((states, actions, rewards, next_states, dones))