
from copy import deepcopy
from config.conf import Conf
from stable_baselines3 import HerReplayBuffer
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from sb3_contrib import TQC


registry = {}
def register(conf, key): registry[key] = conf
def get(key): return deepcopy(registry[key])


u"""
TQC w/ sparse rewards
- FetchPush-v2 for training and test environments.
"""

conf = Conf()

conf.seed = 1

conf.env = {}
conf.env.key = "FetchPush-v2"

conf.test_env = {}
conf.test_env.key = "FetchPush-v2"

conf.agent = {}
conf.agent.type = TQC

conf.train = {}
conf.train.steps = 1005e3
conf.train.batch_size = 2048
conf.train.alpha = 'auto'
conf.train.gamma = 0.95
conf.train.learning_rate = 1e-3
conf.train.tau = 0.05

conf.policy = {}
conf.policy.type = 'MultiInputPolicy'
conf.policy.kwargs = dict(
  net_arch=[512, 512, 512], n_critics=2)

conf.rb = {}
conf.rb.capacity = 1000000
conf.rb.type = DictReplayBuffer
conf.rb.kwargs = {}

register(conf, "tqc-fetchpush-v2")

u"""
TQC+HER w/ sparse rewards
- FetchPush-v2 for training and test environments.
- HER uses 'future' strategy.
"""

conf = get("tqc-fetchpush-v2")
conf.rb.type = HerReplayBuffer
conf.rb.kwargs = dict(
  goal_selection_strategy='future', n_sampled_goal=4,)

register(conf, "tqc-fetchpush-v2-her-future")

u"""
TQC+HER w/ sparse rewards
- FetchPush-v2 for training and test environments.
- HER uses 'final' strategy.
"""

conf = get("tqc-fetchpush-v2-her-future")
conf.rb.kwargs.goal_selection_strategy = "final"

register(conf, "tqc-fetchpush-v2-her-final")

u"""
TQC w/ dense rewards
- FetchPushDense-v2 for training and test environments.
"""

keys = ["tqc-fetchpush-v2",
        "tqc-fetchpush-v2-her-future",
        "tqc-fetchpush-v2-her-final"]
for key in keys:
  conf = get(key)
  conf.env.key = "FetchPushDense-v2"
  conf.test_env.key = "FetchPushDense-v2"
  register(conf, key.replace("fetchpush", "fetchpushdense"))

# u"""
# SAC experiments.
# """

# keys = ["tqc-fetchpush-v2",
#         "tqc-fetchpush-v2-her-future",
#         "tqc-fetchpush-v2-her-final",
#         "tqc-fetchpushdense-v2",
#         "tqc-fetchpushdense-v2-her-future",
#         "tqc-fetchpushdense-v2-her-final"]
# for key in keys:
#   conf = get(key)
#   conf.agent.type = SAC
#   register(conf, key.replace("tqc", "sac"))





