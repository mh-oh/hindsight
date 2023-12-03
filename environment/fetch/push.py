
import numpy as np
from dataclasses import dataclass
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv


class PushEnv(MujocoFetchPushEnv):

  def __init__(self, reward_type, **kwargs):
    super().__init__(reward_type, **kwargs)

  def _sample_goal(self):
    return super()._sample_goal()
  
  def _reset_sim(self):
    return super()._reset_sim()


@dataclass
class Coord:

  x : float
  y : float

  @classmethod
  def from_numpy(cls, a): return cls(a[0], a[1])


@dataclass
class Arena:

  xmin = 1.30 - 0.25
  xmax = 1.30 + 0.25
  ymin = 0.75 - 0.35
  ymax = 0.75 + 0.35

  @property
  def center(self):
    return Coord((self.xmin + self.xmax) / 2,
                 (self.ymin + self.ymax) / 2)
  
  def shrink(self, x, y):
    self.xmin += x
    self.xmax -= x
    self.ymin += y
    self.ymax -= y
  
  def sample(self, rng):
    return Coord(rng.uniform(self.xmin, self.xmax),
                 rng.uniform(self.ymin, self.ymax))


def make_arena(base, option, margin=(0.0, 0.0)):
  
  if option not in {"left", "right", "before", "after"}:
    raise ValueError()

  arena = Arena()
  if option == "left":
    arena.ymin = base.y
  if option == "right":
    arena.ymax = base.y
  if option == "before":
    arena.xmax = base.x
  if option == "after":
    arena.xmin = base.x

  arena.shrink(*margin)
  return arena


class PushEnvOOD(PushEnv):
  
  def __init__(self, 
               reward_type, 
               object_arena=None, target_arena=None, 
               **kwargs):
    super().__init__(reward_type, **kwargs)

    self._arena_options = {None, 
                           "left", "right", 
                           "before", "after", 
                           "near", "far"}

    self._object_arena = object_arena
    self._target_arena = target_arena
    self._margin = (0.1, 0.1)

    if (object_arena not in self._arena_options):
      raise ValueError()
    if (target_arena not in self._arena_options):
      raise ValueError()

  def _sample(self, option):
    agent = Coord.from_numpy(self.initial_gripper_xpos)
    arena = make_arena(agent, option, margin=self._margin)
    return arena.sample(self.np_random)

  def _sample_goal(self):

    if self._target_arena is None:
      return super()._sample_goal()

    target = self.initial_gripper_xpos[:3].copy()
    if self._target_arena in {"left", "right", "before", "after"}:
      target_xy = self._sample(self._target_arena)
      target[0] = target_xy.x
      target[1] = target_xy.y
      target += self.target_offset
      target[2] = self.height_offset
      return target
  
  def _reset_sim(self):

    if self._object_arena is None:
      return super()._reset_sim()
    
    self.data.time = self.initial_time
    self.data.qpos[:] = np.copy(self.initial_qpos)
    self.data.qvel[:] = np.copy(self.initial_qvel)
    if self.model.na != 0:
      self.data.act[:] = None

    object_xpos = self.initial_gripper_xpos[:2].copy()    
    if self._object_arena in {"left", "right", "before", "after"}:
      object_xy = self._sample(self._object_arena)
      object_xpos[0] = object_xy.x
      object_xpos[1] = object_xy.y
    
    object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
    assert object_qpos.shape == (7,)
    object_qpos[:2] = object_xpos
    self._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)

    self._mujoco.mj_forward(self.model, self.data)
    return True

