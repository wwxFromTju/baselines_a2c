"""
Helpers for scripts like run_atari.py.
"""
import os
import gym
from gym.wrappers import FlattenDictWrapper
from my_a2c import logger
from my_a2c.bench import Monitor
from my_a2c.common import set_global_seeds
from my_a2c.common.atari_wrappers import make_atari, wrap_deepmind
from my_a2c.common.vec_env.subproc_vec_env import SubprocVecEnv

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


