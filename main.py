import argparse
from copy import deepcopy
import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import wandb
from wandb.integration.sb3 import WandbCallback
 

import config
import sys

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
  env = VecVideoRecorder(env, f"videos/{args.run}/", record_video_trigger=lambda x: x % 20000 == 0, video_length=200)

  agent = conf.agent.type
  agent = agent(env=env,
                replay_buffer_class=conf.rb.type,
                verbose=1,
                seed=conf.seed,
                device='cuda',
                tensorboard_log=f"runs/{args.run}/",
                policy=conf.policy.type,
                buffer_size=conf.rb.capacity,
                ent_coef=conf.train.alpha,
                batch_size=conf.train.batch_size,
                gamma=conf.train.gamma,
                learning_rate=conf.train.learning_rate,
                tau=conf.train.tau,
                replay_buffer_kwargs=conf.rb.kwargs,
                policy_kwargs=conf.policy.kwargs)

  agent.learn(
    total_timesteps=conf.train.steps,
    callback=WandbCallback(
      model_save_freq=200000,
      model_save_path=f"models/{args.run}",
      verbose=2,
    ),
  )
  run.finish()