# CITS3001: Algorithms, Agents and Artificial Intelligence

## Project: Mario AI Agents

This project includes two Python-based AI agents developed for the Super Mario Bros game using the OpenAI Gym environment with the `gym-super-mario-bros` wrapper. These agents were developed as part of the CITS3001: Algorithms, Agents, and Artificial Intelligence course.

### Project Overview

The project consists of two main components:
1. **Rule-Based Agent** (`rule-based.py`): Uses pre-defined rules and image processing techniques to guide Mario through the game.
2. **Reinforcement Learning Agent** (`train_mario1.py`): Employs the PPO algorithm to train Mario to play the game independently.

Each agent demonstrates a different approach to autonomous gameplay, showcasing rule-based programming versus reinforcement learning for a dynamic and interactive game environment.

### Requirements

This project requires the following libraries:
- `gym` and `gym-super-mario-bros` for the environment
- `nes-py` for NES emulation in Python
- `opencv-python` for image processing in the rule-based agent
- `stable-baselines3` for the reinforcement learning model
- `matplotlib` for visualization

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## Rule-Based Agent

The rule-based agent uses OpenCV to process and detect elements within the game environment, such as enemies, pipes, and obstacles. Mario’s behavior is determined by a set of rules based on detected objects and their positions.

### Key Features

- Object Detection: Identifies Mario’s position and the positions of various objects like pipes, enemies, and obstacles using template matching with OpenCV.
- Custom Actions: Decides Mario’s next action based on object locations (e.g., jump over pipes or avoid holes).
- Reward Tracking: Plots a reward chart showing Mario's performance and progress throughout gameplay.
- Heatmap Generation: Displays a heatmap indicating the areas Mario frequently visits.

### Running the Rule-Based Agent

To run the rule-based agent, execute:
```bash
python rule-based.py
```

This script will start the agent and log Mario’s actions and object detections in real-time. It will also display charts and heatmaps summarizing Mario's movements and accumulated rewards.

## Reinforcement Learning Agent

The reinforcement learning agent uses the PPO (Proximal Policy Optimization) algorithm to train Mario to navigate the game autonomously. The training process involves stacking frames to allow Mario to learn from multiple visual perspectives simultaneously.

### Key Features

- GrayScale Observation: Converts the game screen to grayscale to simplify the input and reduce computational complexity.
- Frame Stacking: Stacks four consecutive frames to provide the agent with temporal context.
- Training Checkpoints: Saves progress at regular intervals, allowing Mario’s learning to be resumed from a checkpoint.
- Customizable Training Parameters: Allows setting of training parameters like learning rate, steps, and callbacks.

### Running the Training Script

To train the reinforcement learning agent, execute:
```bash
python train_mario1.py
```

This will start the training process and save checkpoints in the specified directory. After training, the agent can be tested to observe its learned behaviors.

### Loading and Testing the Trained Model

Samples of loading and training from the previous checkpoint can be found in file `train_mario2.py` and `train_mario3.py`

## Directory Structure

- `Images/`: Contains template images for the rule-based agent to detect objects within the game.
- `train/`: Directory for saving training checkpoints for the PPO model.
- `logs/`: Directory for saving training logs.

## Files

-  `rule-based.py`: Script for the rule-based agent.
- `train_mario1.py`: Script for training the reinforcement learning agent.
- `train_mario2.py` and `train_mario3.py `: Script for loading and training from the previous checkpoint
- `requirements.txt`: Dependencies for the project.

## Visualizations

The project includes visualizations such as:

- `Reward Chart`: Plots Mario's cumulative reward over actions.
- `Heatmap`: Displays Mario’s most visited areas during gameplay.

## Acknowledgments

This project was created as part of coursework for the [CITS3001](https://handbooks.uwa.edu.au/year2023/unitdetails?code=CITS3001) Algorithms, Agents, and Artificial Intelligence unit at UWA. Special thanks to the creators of the gym-super-mario-bros and stable-baselines3 libraries for making this project possible.
