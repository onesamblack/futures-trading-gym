from gym.envs.registration import register

register(
    id='futures-v0',
    entry_point='gym_futures.envs:FuturesEnv',
)
