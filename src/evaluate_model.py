"""
Model evaluation and testing utilities for DQN agents.

This script provides functionality to evaluate trained DQN models,
record gameplay videos, and analyze performance metrics.
"""

import torch
import numpy as np
import gymnasium as gym
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from dqn_core import DQNCartPole, DQNAtari, AtariPreprocessor, get_device, create_environment


class ModelEvaluator:
    """
    Utility class for evaluating trained DQN models.
    
    Supports both CartPole and Atari environments with features like:
    - Performance evaluation over multiple episodes
    - Video recording of gameplay
    - Performance metrics calculation and visualization
    """
    
    def __init__(self, model_path: str, env_name: str, device: str = None):
        self.model_path = model_path
        self.env_name = env_name
        self.device = get_device() if device is None else torch.device(device)
        
        # Load model checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create environment
        self.env = create_environment(env_name)
        self.num_actions = self.env.action_space.n
        
        # Determine environment type and setup
        self.is_atari = env_name.startswith("ALE/") or "atari" in env_name.lower()
        
        if self.is_atari:
            self.model = DQNAtari(self.num_actions).to(self.device)
            self.preprocessor = AtariPreprocessor()
        else:
            # Assume CartPole-like environment
            state_dim = self.env.observation_space.shape[0]
            self.model = DQNCartPole(state_dim, self.num_actions).to(self.device)
            self.preprocessor = None
        
        # Load model weights
        self.model.load_state_dict(self.checkpoint['q_net_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Environment: {env_name}")
        print(f"Device: {self.device}")
        print(f"Atari environment: {self.is_atari}")

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration probability (0 for greedy policy)
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)
        
        with torch.no_grad():
            if self.is_atari:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def evaluate_episodes(self, num_episodes: int = 10, max_steps: int = None, 
                         epsilon: float = 0.0, render: bool = False) -> Dict[str, Any]:
        """
        Evaluate the model over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            epsilon: Exploration probability
            render: Whether to render the environment
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if max_steps is None:
            max_steps = 10000 if self.is_atari else 500
        
        episode_rewards = []
        episode_lengths = []
        
        print(f"Evaluating for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            
            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                if render:
                    self.env.render()
                
                action = self.select_action(state, epsilon)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                
                if self.is_atari:
                    state = self.preprocessor.step(obs)
                else:
                    state = obs
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1:3d}: Reward = {episode_reward:7.2f}, "
                  f"Length = {episode_length:4d}")
        
        # Calculate statistics
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean([r > 0 for r in episode_rewards]) if self.is_atari else None
        }
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
        print(f"Mean Episode Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
        if results['success_rate'] is not None:
            print(f"Success Rate: {results['success_rate']:.2%}")
        print("="*50)
        
        return results

    def record_video(self, output_path: str, num_episodes: int = 1, 
                    max_steps: int = None, epsilon: float = 0.0) -> List[float]:
        """
        Record video of the agent playing.
        
        Args:
            output_path: Path to save the video
            num_episodes: Number of episodes to record
            max_steps: Maximum steps per episode
            epsilon: Exploration probability
            
        Returns:
            List of episode rewards
        """
        if max_steps is None:
            max_steps = 10000 if self.is_atari else 500
        
        # Create video environment
        video_env = gym.make(self.env_name, render_mode='rgb_array')
        
        # Video recording setup
        frames = []
        episode_rewards = []
        
        print(f"Recording {num_episodes} episode(s) to {output_path}")
        
        for episode in range(num_episodes):
            obs, _ = video_env.reset()
            
            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            
            episode_reward = 0
            episode_frames = []
            
            for step in range(max_steps):
                # Capture frame
                frame = video_env.render()
                episode_frames.append(frame)
                
                action = self.select_action(state, epsilon)
                obs, reward, terminated, truncated, _ = video_env.step(action)
                
                if self.is_atari:
                    state = self.preprocessor.step(obs)
                else:
                    state = obs
                
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            frames.extend(episode_frames)
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Frames = {len(episode_frames)}")
        
        # Save video
        self._save_video(frames, output_path)
        video_env.close()
        
        return episode_rewards

    def _save_video(self, frames: List[np.ndarray], output_path: str, fps: int = 30):
        """Save frames as video file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not frames:
            print("No frames to save")
            return
        
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to {output_path}")

    def plot_evaluation_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot evaluation results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Episode rewards
        axes[0].plot(results['episode_rewards'], marker='o')
        axes[0].axhline(y=results['mean_reward'], color='r', linestyle='--', 
                       label=f'Mean: {results["mean_reward"]:.2f}')
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[1].plot(results['episode_lengths'], marker='s', color='orange')
        axes[1].axhline(y=results['mean_length'], color='r', linestyle='--',
                       label=f'Mean: {results["mean_length"]:.2f}')
        axes[1].set_title('Episode Lengths')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Length')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

    def analyze_q_values(self, num_episodes: int = 5) -> Dict[str, Any]:
        """
        Analyze Q-value distributions during gameplay.
        
        Args:
            num_episodes: Number of episodes to analyze
            
        Returns:
            Dictionary containing Q-value statistics
        """
        all_q_values = []
        all_actions = []
        all_max_q_values = []
        
        print(f"Analyzing Q-values for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            
            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            
            episode_q_values = []
            episode_actions = []
            
            max_steps = 10000 if self.is_atari else 500
            
            for step in range(max_steps):
                with torch.no_grad():
                    if self.is_atari:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    q_values = self.model(state_tensor).cpu().numpy()[0]
                    action = q_values.argmax()
                
                episode_q_values.append(q_values)
                episode_actions.append(action)
                all_max_q_values.append(q_values.max())
                
                obs, _, terminated, truncated, _ = self.env.step(action)
                
                if self.is_atari:
                    state = self.preprocessor.step(obs)
                else:
                    state = obs
                
                if terminated or truncated:
                    break
            
            all_q_values.extend(episode_q_values)
            all_actions.extend(episode_actions)
        
        # Calculate statistics
        all_q_values = np.array(all_q_values)
        all_actions = np.array(all_actions)
        
        results = {
            'mean_q_values': np.mean(all_q_values, axis=0),
            'std_q_values': np.std(all_q_values, axis=0),
            'action_distribution': np.bincount(all_actions, minlength=self.num_actions) / len(all_actions),
            'mean_max_q_value': np.mean(all_max_q_values),
            'std_max_q_value': np.std(all_max_q_values)
        }
        
        print("\nQ-VALUE ANALYSIS")
        print("="*30)
        print(f"Mean Q-values per action: {results['mean_q_values']}")
        print(f"Action distribution: {results['action_distribution']}")
        print(f"Mean max Q-value: {results['mean_max_q_value']:.3f} ± {results['std_max_q_value']:.3f}")
        print("="*30)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DQN models')
    
    # Model and environment
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--env_name', type=str, default=None,
                        help='Environment name (auto-detected from model if not provided)')
    
    # Evaluation parameters
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum steps per episode')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Exploration probability during evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    
    # Video recording
    parser.add_argument('--record_video', action='store_true',
                        help='Record video of gameplay')
    parser.add_argument('--video_episodes', type=int, default=3,
                        help='Number of episodes to record')
    parser.add_argument('--video_path', type=str, default='./results/gameplay.mp4',
                        help='Path to save video')
    
    # Analysis
    parser.add_argument('--analyze_q_values', action='store_true',
                        help='Analyze Q-value distributions')
    parser.add_argument('--plot_results', action='store_true',
                        help='Plot evaluation results')
    parser.add_argument('--save_plots', type=str, default=None,
                        help='Path to save plots')
    
    # Output
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Auto-detect environment name if not provided
    if args.env_name is None:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        args.env_name = checkpoint.get('env_name', 'CartPole-v1')
        print(f"Auto-detected environment: {args.env_name}")
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.env_name)
    
    # Run evaluation
    results = evaluator.evaluate_episodes(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        render=args.render
    )
    
    # Record video if requested
    if args.record_video:
        video_path = os.path.join(args.results_dir, args.video_path)
        video_rewards = evaluator.record_video(
            output_path=video_path,
            num_episodes=args.video_episodes,
            max_steps=args.max_steps,
            epsilon=args.epsilon
        )
        print(f"Video recording rewards: {video_rewards}")
    
    # Analyze Q-values if requested
    if args.analyze_q_values:
        q_results = evaluator.analyze_q_values(num_episodes=5)
    
    # Plot results if requested
    if args.plot_results:
        save_path = None
        if args.save_plots:
            save_path = os.path.join(args.results_dir, args.save_plots)
        evaluator.plot_evaluation_results(results, save_path)


if __name__ == "__main__":
    main()
