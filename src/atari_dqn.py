"""
DQN implementation for Atari Pong environment.

This script implements Deep Q-Network (DQN) with experience replay
and optional prioritized experience replay for the Atari Pong game.
The agent learns to play Pong by processing visual observations.

Key features:
- Convolutional neural network for visual processing
- Frame stacking and preprocessing
- Experience replay buffer
- Target network for stable training
- Optional prioritized experience replay
"""

import torch
import torch.optim as optim
import numpy as np
import random
import os
import argparse
import matplotlib.pyplot as plt

from dqn_core import DQNAtari, AtariPreprocessor, ReplayBuffer, PrioritizedReplayBuffer, get_device, set_seed, create_environment

# Try to import additional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")


class AtariDQNAgent:
    """
    DQN Agent for Atari environments.
    
    Implements the DQN algorithm with convolutional neural networks
    for learning optimal policies in Atari games like Pong.
    """
    
    def __init__(self, args):
        # Environment setup
        self.env_name = args.env_name
        self.env = create_environment(self.env_name)
        self.num_actions = self.env.action_space.n
        
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
        self.use_prioritized_replay = args.use_prioritized_replay
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize preprocessing
        self.preprocessor = AtariPreprocessor(frame_stack=4)
        
        # Initialize networks
        self.q_net = DQNAtari(self.num_actions).to(self.device)
        self.target_net = DQNAtari(self.num_actions).to(self.device)
        
        # Initialize target network with main network weights
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Experience replay buffer
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                capacity=self.memory_size,
                alpha=args.priority_alpha,
                beta=args.priority_beta
            )
        else:
            self.memory = ReplayBuffer(self.memory_size)
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.update_count = 0
        self.env_step_count = 0
        
        # Initialize wandb if available
        self.use_wandb = args.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"{self.env_name}_{args.wandb_run_name}",
                config=vars(args)
            )
        
        print(f"Atari DQN Agent initialized for {self.env_name}")
        print(f"Device: {self.device}")
        print(f"Number of actions: {self.num_actions}")
        print(f"Using prioritized replay: {self.use_prioritized_replay}")

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current processed state (4x84x84)
            
        Returns:
            Selected action
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
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
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
        
        # Compute TD errors for prioritized replay
        td_errors = (current_q_values.squeeze() - target_q_values).detach()
        
        # Compute weighted loss
        loss = (weights * (td_errors ** 2)).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors.cpu().numpy())
        
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
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            
            episode_reward = 0
            episode_length = 0
            
            while episode_length < self.max_episode_steps:
                # Select and execute action
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Preprocess next state
                next_state = self.preprocessor.step(next_obs)
                
                # Store transition in replay buffer
                if self.use_prioritized_replay:
                    # Compute initial TD error for prioritized replay
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                        
                        current_q = self.q_net(state_tensor)[0, action]
                        if done:
                            target_q = reward
                        else:
                            next_q = self.target_net(next_state_tensor).max()
                            target_q = reward + self.gamma * next_q
                        
                        td_error = abs(current_q - target_q).item()
                    
                    self.memory.add((state, action, reward, next_state, done), td_error)
                else:
                    self.memory.add(state, action, reward, next_state, done)
                
                # Update state and tracking
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.env_step_count += 1
                
                # Train the network
                self.train_step()
                
                if done:
                    break
            
            # Update epsilon (exploration decay)
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Track episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Logging
            if episode % 100 == 0 or episode == num_episodes - 1:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_loss = np.mean(self.losses[-1000:]) if self.losses else 0
                
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:7.2f} | "
                      f"Epsilon: {self.epsilon:.4f} | "
                      f"Updates: {self.update_count} | "
                      f"Avg Loss: {avg_loss:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        "episode": episode,
                        "avg_reward_100": avg_reward,
                        "avg_length_100": avg_length,
                        "epsilon": self.epsilon,
                        "update_count": self.update_count,
                        "env_step_count": self.env_step_count,
                        "avg_loss": avg_loss
                    })
        
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
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            
            episode_reward = 0
            episode_length = 0
            
            while episode_length < self.max_episode_steps:
                if render:
                    self.env.render()
                
                action = self.select_action(state)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                state = self.preprocessor.step(obs)
                
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

    def save_model(self, filename: str = None):
        """Save the trained model."""
        if filename is None:
            filename = f"{self.env_name.replace('/', '_')}_dqn.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'env_step_count': self.env_step_count,
            'env_name': self.env_name
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
        self.env_step_count = checkpoint.get('env_step_count', 0)
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
        
        # Moving average of rewards
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
        filename = f"{self.env_name.replace('/', '_')}_training_progress.png"
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='DQN for Atari Games')
    
    # Environment parameters
    parser.add_argument('--env_name', type=str, default='ALE/Pong-v5',
                        help='Atari environment name')
    parser.add_argument('--max_episode_steps', type=int, default=10000,
                        help='Maximum steps per episode')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--memory_size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    
    # Exploration parameters
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_min', type=float, default=0.1,
                        help='Minimum epsilon value')
    parser.add_argument('--epsilon_decay', type=float, default=0.99995,
                        help='Epsilon decay rate')
    
    # Network update parameters
    parser.add_argument('--target_update_frequency', type=int, default=1000,
                        help='Target network update frequency')
    
    # Prioritized experience replay parameters
    parser.add_argument('--use_prioritized_replay', action='store_true',
                        help='Use prioritized experience replay')
    parser.add_argument('--priority_alpha', type=float, default=0.6,
                        help='Priority exponent alpha')
    parser.add_argument('--priority_beta', type=float, default=0.4,
                        help='Importance sampling exponent beta')
    
    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    
    # Save/load parameters
    parser.add_argument('--save_dir', type=str, default='./models/atari',
                        help='Directory to save models and results')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load pre-trained model')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help='Mode: train or eval')
    
    # Logging parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='atari-dqn',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default='dqn_run',
                        help='W&B run name')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create agent
    agent = AtariDQNAgent(args)
    
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
