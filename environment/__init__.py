

import gymnasium as gym
from gymnasium.envs.registration import register


def make_env(key, *args, **kwargs):
  return gym.make(key, *args, **kwargs, render_mode="rgb_array")


register("environment/fetchpush",
         "environment.fetch.push:PushEnv",
         kwargs={"reward_type": "sparse"},
         max_episode_steps=50)

register("environment/fetchpushleftright",
         "environment.fetch.push:PushEnvOOD",
         kwargs={"reward_type": "sparse",
                 "object_arena": "left",
                 "target_arena": "right"},
         max_episode_steps=50)

register("environment/fetchpushrightleft",
         "environment.fetch.push:PushEnvOOD",
         kwargs={"reward_type": "sparse",
                 "object_arena": "right",
                 "target_arena": "left"},
         max_episode_steps=50)

register("environment/fetchpushbeforeafter",
         "environment.fetch.push:PushEnvOOD",
         kwargs={"reward_type": "sparse",
                 "object_arena": "before",
                 "target_arena": "after"},
         max_episode_steps=50)

register("environment/fetchpushafterbefore",
         "environment.fetch.push:PushEnvOOD",
         kwargs={"reward_type": "sparse",
                 "object_arena": "after",
                 "target_arena": "before"},
         max_episode_steps=50)