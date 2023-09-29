# Import the game
import os
import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import json

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# 1. Create the base environment
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order="last")


env.reset()


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, "best_model_{}".format(self.n_calls))
            self.model.save(model_path)

        return True


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

TIME_STEPS = 100
model_load_file = "./train/current_best_model.zip"

current_iteration = 0
conf = None
with open("./train/model_statisctics", "r") as f:
    conf = json.load(f)
    current_iteration = conf["iterations"]

# This is the AI model started
model = PPO.load(model_load_file, env=env)

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=10000, callback=callback)

model.save(f"./train/current_best_model")
model.save(f"./train/best_model_{current_iteration+TIME_STEPS}")

with open("./train/model_statisctics", "w") as f:
    conf["iterations"] += TIME_STEPS
    json.dump(conf, f)
