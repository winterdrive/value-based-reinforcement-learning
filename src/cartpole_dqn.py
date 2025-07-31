"""
DQN implementation for CartPole environment.

This script implements Deep Q-Network (DQN) with experience replay
for the CartPole-v1 environment. The agent learns to balance a pole
on a cart by taking discrete actions (left or right).

Key features:
- Epsilon-greedy exploration strategy
- Experience replay buffer
- Target network for stable training
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os
import argparse
import matplotlib.pyplot as plt

from dqn_core import DQNCartPole, ReplayBuffer, get_device, set_seed


class CartPoleDQNAgent:
    """
    DQN Agent for CartPole environment.
    
    Implements the DQN algorithm with experience replay and target network
    for learning optimal control policies in the CartPole environment.
    """
    
    def __init__(self, args):
        # Environment setup
        self.env = gym.make("CartPole-v1")
        self.state_dim = self.env.observation_space.shape[0]  # 4 for CartPole
        self.num_actions = self.env.action_space.n  # 2 for CartPole
        
        # Training parameters
        self.device = get_device()
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.target_update_frequency = args.target_update_frequency
        self.max_episode_steps = args.max_episode_steps
        self.save_dir = args.save_dir
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize networks
        self.q_net = DQNCartPole(self.state_dim, self.num_actions).to(self.device)
        self.target_net = DQNCartPole(self.state_dim, self.num_actions).to(self.device)
        
        # Initialize target network with main network weights
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(self.memory_size)
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.update_count = 0
        
        print(f"CartPole DQN Agent initialized")
        print(f"Device: {self.device}")
        print(f"State dimension: {self.state_dim}")
        print(f"Number of actions: {self.num_actions}")

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state
            
        Returns:
            Selected action (0 or 1 for CartPole)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step using a batch from replay buffer."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update tracking
        self.losses.append(loss.item())
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self, num_episodes: int):
        """
        Train the DQN agent for specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
        """
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while episode_length < self.max_episode_steps:
                # Select and execute action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition in replay buffer
                self.memory.add(state, action, reward, next_state, done)
                
                # Update state and tracking
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Train the network
                self.train_step()
                
                if done:
                    break
            
            # Update epsilon (exploration decay)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Track episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            if episode % 100 == 0 or episode == num_episodes - 1:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:7.2f} | "
                      f"Epsilon: {self.epsilon:.4f} | "
                      f"Updates: {self.update_count}")
        
        print("Training completed!")

    def evaluate(self, num_episodes: int = 10, render: bool = False) -> float:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            Average reward over evaluation episodes
        """
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        old_epsilon = self.epsilon
        self.epsilon = 0  # Disable exploration during evaluation
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while episode_length < self.max_episode_steps:
                if render:
                    self.env.render()
                
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            print(f"Eval Episode {episode + 1}: Reward = {episode_reward}, Length = {episode_length}")
        
        self.epsilon = old_epsilon  # Restore exploration
        avg_reward = np.mean(eval_rewards)
        print(f"Average evaluation reward: {avg_reward:.2f}")
        
        return avg_reward

    def save_model(self, filename: str = "cartpole_dqn.pt"):
        """Save the trained model."""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.losses = checkpoint.get('losses', [])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.update_count = checkpoint.get('update_count', 0)
        print(f"Model loaded from {filepath}")

    def plot_training_progress(self):
        """Plot training progress and save figures."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        
        # Moving average of rewards (window=100)
        if len(self.episode_rewards) >= 100:
            moving_avg = [np.mean(self.episode_rewards[i:i+100]) 
                         for i in range(len(self.episode_rewards) - 99)]
            axes[1, 0].plot(moving_avg)
            axes[1, 0].set_title('Moving Average Reward (window=100)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Average Reward')
        
        # Training losses
        if self.losses:
            axes[1, 1].plot(self.losses)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='DQN for CartPole')
    
    # Environment parameters
    parser.add_argument('--max_episode_steps', type=int, default=500,
                        help='Maximum steps per episode')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--memory_size', type=int, default=10000,
                        help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    
    # Exploration parameters
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_min', type=float, default=0.01,
                        help='Minimum epsilon value')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Epsilon decay rate')
    
    # Network update parameters
    parser.add_argument('--target_update_frequency', type=int, default=100,
                        help='Target network update frequency')
    
    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    
    # Save/load parameters
    parser.add_argument('--save_dir', type=str, default='./models/cartpole',
                        help='Directory to save models and results')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load pre-trained model')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help='Mode: train or eval')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create agent
    agent = CartPoleDQNAgent(args)
    
    if args.mode == 'train':
        # Train the agent
        agent.train(args.num_episodes)
        
        # Save the trained model
        agent.save_model()
        
        # Plot training progress
        agent.plot_training_progress()
        
        # Evaluate the trained agent
        agent.evaluate(args.eval_episodes, args.render)
        
    elif args.mode == 'eval':
        if args.load_model is None:
            print("Error: Must specify --load_model for evaluation mode")
            return
        
        # Load pre-trained model and evaluate
        agent.load_model(args.load_model)
        agent.evaluate(args.eval_episodes, args.render)


if __name__ == "__main__":
    main()
