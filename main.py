import argparse
from copy import deepcopy
import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback


import config
import sys

import numpy as np
from gymnasium import spaces


def add_random_exploration(cls):

  class C(cls):

    def __init__(self, *args, random_exploration=None, **kwargs):
      super().__init__(*args, **kwargs)
      self._random_exploration = random_exploration
    
    def _sample_action(self,
                       learning_starts,
                       action_noise,
                       n_envs):
      
      _sample_action = super()._sample_action
      if self._random_exploration is None:
        return _sample_action(learning_starts,
                              action_noise,
                              n_envs)
      if self._random_exploration >= self._random_exploration:
        return _sample_action(learning_starts,
                              action_noise,
                              n_envs)

      # randomly sample actions from a uniform distribution
      # with a probability self._random_exploration (used in HER + DDPG)      
      unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])

      # from stable-baselines3 v2.0.0
      # see common/off_policy_algorithm.py
      if isinstance(self.action_space, spaces.Box):
          scaled_action = self.policy.scale_action(unscaled_action)

          # Add noise to the action (improve exploration)
          if action_noise is not None:
              scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

          # We store the scaled action in the buffer
          buffer_action = scaled_action
          action = self.policy.unscale_action(scaled_action)
      else:
          # Discrete case, no need to normalize or clip
          buffer_action = unscaled_action
          action = buffer_action

      return action, buffer_action

  return C



def get_parser():

  p = argparse.ArgumentParser()
  p.add_argument("--conf",
    type=str, required=True, help="configuration.")
  p.add_argument("--run",
    type=str, required=True, help="run.")
  return p


if __name__ == "__main__":

  args = sys.argv[1:]
  try:
    index = args.index("--")
    args, extra = args[:index], args[index:]
  except:
    extra = []
  
  parser = get_parser()
  args = parser.parse_args(args)
  conf = config.get(args.conf)
  conf = config.override(conf, extra)
  print(conf)

  run = wandb.init(project="hindsight",
                   config=conf,
                   sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                   monitor_gym=True,  # auto-upload the videos of agents playing the game
                   save_code=True,  # optional
                   name=args.run)
  
  def make_env():
    env = gym.make(conf.env.key, render_mode = "rgb_array")
    env = Monitor(env)  # record stats such as returns
    env = TimeFeatureWrapper(env)
    return env

  env = DummyVecEnv([make_env])
  env = VecNormalize(env, gamma=conf.train.gamma)
  # env = VecVideoRecorder(env, f"videos/{args.run}/train/", record_video_trigger=lambda x: x % 20000 == 0, video_length=200)

  test_env = DummyVecEnv([make_env])
  test_env = VecNormalize(test_env, gamma=conf.train.gamma)
  # test_env = VecVideoRecorder(test_env, f"videos/{args.run}/test/", record_video_trigger=lambda x: x % 20000 == 0, video_length=200)
  test = EvalCallback(test_env,
                      n_eval_episodes=30,
                      log_path=f"logs/{args.run}", eval_freq=5000,
                      deterministic=True, render=False)

  agent = add_random_exploration(conf.agent.type)
  agent = agent(env=env,
                replay_buffer_class=conf.rb.type,
                verbose=1,
                seed=conf.seed,
                device='cuda',
                tensorboard_log=f"runs/{args.run}/",
                policy=conf.policy.type,
                buffer_size=conf.rb.capacity,
                # ent_coef=conf.train.alpha,
                batch_size=conf.train.batch_size,
                gamma=conf.train.gamma,
                learning_rate=conf.train.learning_rate,
                tau=conf.train.tau,
                replay_buffer_kwargs=conf.rb.kwargs,
                policy_kwargs=conf.policy.kwargs,
                random_exploration=conf.train.random_action_probability)

  agent.learn(
    total_timesteps=conf.train.steps,
    callback=[test,
              WandbCallback(model_save_freq=200000,
                            model_save_path=f"models/{args.run}",
                            verbose=2,)])
  run.finish()