"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import gym
from gym import wrappers

from utils.seed import set_global_seeds
from utils.atari_wrapper import wrap_deepmind, wrap_deepmind_ram, wrap_deepmind_nomaxclip

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = 'tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind(env)

    return env

def get_env_by_id(env_id, seed, skip_f = 4, clip_max = False):
    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = 'tmp/gym-results'
    env = wrap_deepmind_nomaxclip(env,skip_f)

    return env


def get_ram_env(env, seed):
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind_ram(env)

    return env

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
