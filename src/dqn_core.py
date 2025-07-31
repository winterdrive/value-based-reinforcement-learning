"""
Core DQN implementation for value-based reinforcement learning.

This module contains the main DQN algorithm implementation with support for:
- Deep Q-Network (DQN) with experience replay
- Double DQN (DDQN) for reduced overestimation bias
- Prioritized Experience Replay for improved sample efficiency
- Target network updates for stable training

The implementation supports both discrete control tasks (CartPole) and 
Atari games with visual observations.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
from collections import deque
from typing import List, Tuple

# Try to import ale_py for Atari support
try:
    import ale_py
    gym.register_envs(ale_py)
    ALE_AVAILABLE = True
except ImportError:
    ALE_AVAILABLE = False
    print("Warning: ale_py not available. Atari environments will not work.")


def init_weights(m):
    """Initialize network weights using Kaiming initialization."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQNCartPole(nn.Module):
    """
    Deep Q-Network for CartPole environment.
    
    A fully connected neural network that takes the 4-dimensional state
    (cart position, cart velocity, pole angle, pole angular velocity)
    and outputs Q-values for each action.
    """
    
    def __init__(self, state_dim: int = 4, num_actions: int = 2, hidden_dim: int = 256):
        super(DQNCartPole, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class DQNAtari(nn.Module):
    """
    Deep Q-Network for Atari environments.
    
    A convolutional neural network that processes stacked frames
    (4 x 84 x 84 grayscale images) and outputs Q-values for each action.
    Architecture based on the original DQN paper (Mnih et al., 2015).
    """
    
    def __init__(self, num_actions: int):
        super(DQNAtari, self).__init__()
        
        # Convolutional layers for visual feature extraction
        self.conv_layers = nn.Sequential(
            # First conv layer: 4x84x84 -> 32x20x20
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Second conv layer: 32x20x20 -> 64x9x9  
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Third conv layer: 64x9x9 -> 64x7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, x):
        """Forward pass through the network."""
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        
        # Extract features using convolutional layers
        features = self.conv_layers(x)
        
        # Flatten for fully connected layers
        features = features.view(features.size(0), -1)
        
        # Output Q-values
        q_values = self.fc_layers(features)
        
        return q_values


class AtariPreprocessor:
    """
    Preprocessing pipeline for Atari environments.
    
    Handles frame stacking, grayscale conversion, and resizing
    as described in the DQN paper.
    """
    
    def __init__(self, frame_stack: int = 4, frame_size: Tuple[int, int] = (84, 84)):
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)

    def preprocess_frame(self, obs: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame:
        1. Convert to grayscale
        2. Resize to 84x84
        3. Normalize to [0, 255] uint8
        """
        # Convert RGB to grayscale
        if len(obs.shape) == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
            
        # Resize to target size
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        
        return resized.astype(np.uint8)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Reset the frame stack with the initial observation."""
        frame = self.preprocess_frame(obs)
        # Fill the frame stack with the same frame
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Add a new frame to the stack."""
        frame = self.preprocess_frame(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores transitions (state, action, reward, next_state, done)
    and provides random sampling for training.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        # Separate the batch into individual components
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Implements prioritized sampling based on TD-error magnitude
    as described in Schaul et al. (2016).
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error: float):
        """Add transition with initial priority based on TD-error."""
        priority = (abs(error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
            
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], errors: List[float]):
        """Update priorities based on new TD-errors."""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


def get_device() -> torch.device:
    """Get the appropriate device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_environment(env_name: str) -> gym.Env:
    """Create and configure the specified environment."""
    if env_name.startswith("ALE/") and not ALE_AVAILABLE:
        raise ImportError(f"ale_py is required for Atari environment {env_name}")
    
    env = gym.make(env_name)
    return env


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
