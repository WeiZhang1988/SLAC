from gym.envs.registration import register

register(
    id='CarlaEnv-v0',
    entry_point='carla_env.carla_env:CarlaEnv',
)

