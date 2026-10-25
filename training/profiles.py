from environment.gym_env import MazeGymEnv
from environment.gym_v2 import ExplorationEnv, PROFILE
from environment.gym_v3 import RangeChargeEnv, PROFILE as PROFILE_V3

def make_env(config, profile, lesson_probability=None):
    if profile == PROFILE_V3:
        return RangeChargeEnv(config, 0.3 if lesson_probability is None else lesson_probability)
    if profile == PROFILE: return ExplorationEnv(config)
    if profile == 'legacy-v1': return MazeGymEnv(config)
    raise ValueError('Unknown training profile: '+str(profile))
