#Yasincan Bozkurt
#I used Google colab to run my code.
#I also added utils.py
# Import Google Colab's drive module
from google.colab import drive
# Mount Google Drive to the Colab runtime
drive.mount('/gdrive')
# Define path for data
#I used Google drive
data_dir = '/gdrive/MyDrive/odev'
MODEL_DIR = './train_models/'
LOGGING_DIR = './log_files/'
# Install necessary packages
!pip install wheel==0.38.4
!pip install setuptools==65.5.0
!pip install gym
!pip install gym_super_mario_bros==7.3.0 nes_py
!pip install torch torchvision torchaudio --extra-index-url 
https://download.pytorch.org/whl/cu113
!pip install stable-baselines3[extra]
!pip install ray
# Import necessary modules for game environment and models
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
import numpy as np
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, 
DummyVecEnv
from stable_baselines3 import PPO, DQN

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import RecordVideo
from time import time
from matplotlib import pyplot as plt
# Create game environment
game_instance = gym_super_mario_bros.make('SuperMarioBros-v0')
game_instance = JoypadSpace(game_instance, SIMPLE_MOVEMENT)
# Convert to grayscale to reduce dimensionality
game_instance = GrayScaleObservation(game_instance, keep_dim=True)
# Vectorize the environment for the reinforcement learning agent
game_instance = DummyVecEnv([lambda: game_instance]) 
# Stack frames for temporal information
game_instance = VecFrameStack(game_instance, 4, channels_order='last') 
# Monitor the game instance to keep track of progress
game_instance = VecMonitor(game_instance, MODEL_DIR+"/GameMonitor")
# Callback to save best training rewards
callback = SaveOnBestTrainingRewardCallback(save_freq=100000, check_freq=1000, 
chk_dir=MODEL_DIR)
# Define and configure DQN model
#parameters are given here.
# I changed learning rate#
from stable_baselines3 import DQN
model_object = DQN('CnnPolicy',
game_instance,
batch_size=192,
verbose=1,
learning_starts=10000,
learning_rate=5e-3,
exploration_fraction=0.1,
exploration_initial_eps=1.0,
exploration_final_eps=0.1,
train_freq=8,
buffer_size=10000,
tensorboard_log=LOGGING_DIR
)
# Train model
model_object.learn(total_timesteps=4000000, log_interval=1, callback=callback)
# Create directory for saving model if it does not exist

os.makedirs('train_models', exist_ok=True)
# Save the model
model_save_path = os.path.join('train_models', 'optimal_model')
model_object.save(model_save_path)
# Yasincan Bozkurt