from gymnasium.envs.registration import register

register(
    id="catanatron-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv",
)

# from gymnasium.envs.registration import register

# register(
#     id="catanatronp3-v1",
#     entry_point="catanatron_gym.envs:CatanatronEnv3",
# )
# register(
#     id="catanatronp2-v1",
#     entry_point="catanatron_gym.envs:CatanatronEnv2",
# )
# register(
#     id="catanatronp1-v1",
#     entry_point="catanatron_gym.envs:CatanatronEnv",
# )
# register(
#     id="catanatronp4-v1",
#     entry_point="catanatron_gym.envs:CatanatronEnv4",
# )
