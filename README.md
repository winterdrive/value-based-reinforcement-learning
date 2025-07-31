# Value-Based Reinforcement Learning

This project implements Deep Q-Network (DQN) algorithms for value-based reinforcement learning. The implementation supports both discrete control tasks (CartPole) and Atari games with visual observations (Pong, Breakout, etc.).

## Features

- **Deep Q-Network (DQN)** with experience replay
- **Target network** for stable training
- **Prioritized Experience Replay** for improved sample efficiency
- **Support for multiple environments**: CartPole and Atari games
- **Comprehensive evaluation tools** with video recording
- **Training progress visualization** and metrics tracking
- **Modular and extensible architecture**

## Technical Architecture

### Core Components

- **DQN Networks**: Separate architectures for CartPole (fully connected) and Atari (convolutional)
- **Experience Replay**: Standard and prioritized replay buffers
- **Preprocessing**: Frame stacking and resizing for Atari environments
- **Training Loop**: Epsilon-greedy exploration with decay
- **Evaluation**: Performance assessment and video recording

### Key Algorithms

1. **Deep Q-Network (DQN)**: Basic value-based RL with neural network function approximation
2. **Experience Replay**: Storing and sampling past experiences for stable learning
3. **Target Network**: Separate network for computing target Q-values
4. **Prioritized Experience Replay**: Sampling important transitions more frequently

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd value-based-reinforcement-learning
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Atari ROMs** (for Atari environments)

   ```bash
   python -c "import ale_py.roms as roms; roms.verify_install()"
   ```

## Usage

### CartPole Training

Train a DQN agent for the CartPole environment:

```bash
python src/cartpole_dqn.py --num_episodes 1000 --save_dir ./models/cartpole
```

#### CartPole Arguments

- `--num_episodes`: Number of training episodes (default: 1000)
- `--memory_size`: Replay buffer size (default: 10000)
- `--batch_size`: Training batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon_decay`: Exploration decay rate (default: 0.995)
- `--target_update_frequency`: Target network update frequency (default: 100)

### Atari Training

Train a DQN agent for Atari Pong:

```bash
python src/atari_dqn.py --env_name "ALE/Pong-v5" --num_episodes 2000 --save_dir ./models/pong
```

Train with prioritized experience replay:

```bash
python src/atari_dqn.py --env_name "ALE/Pong-v5" --use_prioritized_replay --num_episodes 2000
```

#### Atari Arguments

- `--env_name`: Atari environment name (default: "ALE/Pong-v5")
- `--num_episodes`: Number of training episodes (default: 2000)
- `--memory_size`: Replay buffer size (default: 100000)
- `--use_prioritized_replay`: Enable prioritized experience replay
- `--epsilon_decay`: Exploration decay rate (default: 0.99995)
- `--target_update_frequency`: Target network update frequency (default: 1000)

### Model Evaluation

Evaluate a trained model:

```bash
python src/evaluate_model.py --model_path ./models/cartpole/cartpole_dqn.pt --num_episodes 10
```

Record gameplay video:

```bash
python src/evaluate_model.py \
    --model_path ./models/pong/ALE_Pong-v5_dqn.pt \
    --record_video \
    --video_episodes 3 \
    --video_path ./results/pong_gameplay.mp4
```

Analyze Q-value distributions:

```bash
python src/evaluate_model.py \
    --model_path ./models/cartpole/cartpole_dqn.pt \
    --analyze_q_values \
    --plot_results
```

### Quick Demo

Run the complete demo script:

```bash
cd demo && chmod +x demo.sh && ./demo.sh
```

## Project Structure

```
value-based-reinforcement-learning/
├── README.md
├── requirements.txt
├── src/
│   ├── dqn_core.py              # Core DQN components and utilities
│   ├── cartpole_dqn.py          # CartPole DQN implementation
│   ├── atari_dqn.py             # Atari DQN implementation
│   └── evaluate_model.py        # Model evaluation and testing
├── demo/
│   └── demo.sh                  # Quick demonstration script
├── models/                      # Saved model checkpoints
└── results/                     # Evaluation results and videos
```

### Key Files Description

- **`dqn_core.py`**: Core DQN components including networks, replay buffers, and utilities
- **`cartpole_dqn.py`**: Complete DQN implementation for CartPole environment
- **`atari_dqn.py`**: DQN implementation for Atari games with CNN and preprocessing
- **`evaluate_model.py`**: Comprehensive evaluation tools for trained models

## Supported Environments

### CartPole

- **Environment**: `CartPole-v1`
- **State Space**: 4-dimensional continuous (position, velocity, angle, angular velocity)
- **Action Space**: 2 discrete actions (left, right)
- **Goal**: Balance pole for 500 timesteps

### Atari Games

- **Pong**: `ALE/Pong-v5`
- **Breakout**: `ALE/Breakout-v5`
- **Space Invaders**: `ALE/SpaceInvaders-v5`
- **State Space**: 210x160x3 RGB images (preprocessed to 84x84 grayscale)
- **Action Space**: Game-specific discrete actions
- **Goal**: Maximize game score

## Experimental Results

### CartPole Performance

- **Target Score**: 475+ (considered solved)
- **Training Episodes**: ~300-500 episodes to convergence
- **Success Rate**: >95% after training

### Atari Pong Performance

- **Target Score**: Positive average reward
- **Training Episodes**: 1000-2000 episodes for basic competency
- **Performance**: Can learn to hit the ball and score points

For comprehensive experimental analysis, training metrics, hyperparameter tuning results, and detailed technical discussion, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Implementation Details

### Network Architectures

**CartPole DQN**:

```
Linear(4, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 2)
```

**Atari DQN**:

```
Conv2d(4, 32, 8, 4) -> ReLU -> Conv2d(32, 64, 4, 2) -> ReLU -> Conv2d(64, 64, 3, 1) -> ReLU
-> Flatten -> Linear(3136, 512) -> ReLU -> Linear(512, num_actions)
```

### Training Configuration

**CartPole**:

- Learning Rate: 1e-3
- Batch Size: 32
- Memory Size: 10,000
- Target Update: Every 100 steps
- Epsilon Decay: 0.995

**Atari**:

- Learning Rate: 1e-4
- Batch Size: 32
- Memory Size: 100,000
- Target Update: Every 1,000 steps
- Epsilon Decay: 0.99995

## Advanced Features

### Prioritized Experience Replay

- Prioritizes transitions with higher TD-errors
- Implements importance sampling for unbiased learning
- Configurable alpha (prioritization exponent) and beta (importance sampling)

### Preprocessing Pipeline

- Frame stacking (4 consecutive frames)
- Grayscale conversion
- Resizing to 84x84 pixels
- Normalization to [0, 1] range

### Evaluation Metrics

- Episode rewards and lengths
- Success rates
- Q-value analysis
- Training loss curves
- Performance visualization

## Extensions and Improvements

This implementation can be extended with:

1. **Double DQN**: Reduce overestimation bias
2. **Dueling DQN**: Separate value and advantage estimation
3. **Noisy Networks**: Parameter space exploration
4. **Rainbow DQN**: Combination of multiple improvements
5. **Multi-step learning**: N-step returns
6. **Distributional RL**: Learn value distributions
