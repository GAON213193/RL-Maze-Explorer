from environment.gym_env import MazeGymEnv
from environment.gym_v2 import ExplorationEnv, PROFILE

def make_env(config, profile):
    if profile == PROFILE: return ExplorationEnv(config)
    if profile == 'legacy-v1': return MazeGymEnv(config)
    raise ValueError('Unknown training profile: '+str(profile))
