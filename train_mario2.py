import os
import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Define the Super Mario Bros environment and pre-processing steps as before
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

env.reset()


# Custom callback for saving checkpoints
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

model_path = CHECKPOINT_DIR + "best_model_550000.zip"  # Change this to the path of your checkpoint
model = PPO.load(model_path, env)

# Create and set up your custom callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Continue training the model
model.learn(total_timesteps=1000000, callback=callback)
# Save the model after continued training
model.save(CHECKPOINT_DIR + "best_model_continued")
