


from config.registry import get, register
from copy import deepcopy

def override(conf, args):
  conf = deepcopy(conf)
  for a in args[1:]:
    key, x = a.split("=")
    if "." in key:
      raise NotImplementedError(f"ongoing...")
    conf[key] = type(conf[key])(x)
  return conf