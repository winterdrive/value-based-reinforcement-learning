#!/bin/bash
# Value-Based Reinforcement Learning Demo Script

echo "==== Value-Based Reinforcement Learning Demo ===="

echo "==== 1. Installing Required Packages ===="
pip install -r ../requirements.txt

echo "==== 2. Installing Atari ROMs ===="
python -c "import ale_py.roms as roms; roms.verify_install()"

echo "==== 3. Training CartPole DQN (Quick Demo) ===="
echo "Training CartPole agent for 200 episodes..."
python ../src/cartpole_dqn.py \
    --num_episodes 200 \
    --save_dir "./models/cartpole_demo" \
    --memory_size 5000 \
    --target_update_frequency 50 \
    --epsilon_decay 0.99

echo "==== 4. Evaluating CartPole Agent ===="
CARTPOLE_MODEL=$(find ./models/cartpole_demo -name "*.pt" | head -n 1)
if [ -n "$CARTPOLE_MODEL" ]; then
    echo "Evaluating CartPole model: $CARTPOLE_MODEL"
    python ../src/evaluate_model.py \
        --model_path "$CARTPOLE_MODEL" \
        --num_episodes 10 \
        --plot_results \
        --save_plots "cartpole_evaluation.png" \
        --results_dir "./results"
else
    echo "No CartPole model found for evaluation"
fi

echo "==== 5. Training Atari Pong DQN (Quick Demo) ===="
echo "Training Pong agent for 100 episodes..."
python ../src/atari_dqn.py \
    --env_name "ALE/Pong-v5" \
    --num_episodes 100 \
    --save_dir "./models/pong_demo" \
    --memory_size 10000 \
    --target_update_frequency 200 \
    --epsilon_decay 0.999 \
    --batch_size 16

echo "==== 6. Evaluating Atari Pong Agent ===="
PONG_MODEL=$(find ./models/pong_demo -name "*.pt" | head -n 1)
if [ -n "$PONG_MODEL" ]; then
    echo "Evaluating Pong model: $PONG_MODEL"
    python ../src/evaluate_model.py \
        --model_path "$PONG_MODEL" \
        --num_episodes 5 \
        --plot_results \
        --save_plots "pong_evaluation.png" \
        --record_video \
        --video_episodes 2 \
        --video_path "pong_gameplay.mp4" \
        --results_dir "./results"
else
    echo "No Pong model found for evaluation"
fi

echo "==== 7. Demo Complete! ===="
echo "Results saved in:"
echo "  - Models: ./models/"
echo "  - Evaluation results: ./results/"
echo "  - Videos: ./results/"
echo ""
echo "To train with full settings, use:"
echo "  CartPole: python ../src/cartpole_dqn.py --num_episodes 1000"
echo "  Atari Pong: python ../src/atari_dqn.py --env_name 'ALE/Pong-v5' --num_episodes 2000"
