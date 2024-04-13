from gymnasium.envs.registration import register

# 4 v 1, 1st
register(
    id="catanatron-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv",
)
# 4 v 1, 3rd
register(
    id="catanatronp3-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv3",
)
# 4 v 1, 2nd
register(
    id="catanatronp2-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv2",
)
# 4 v 1, 4th
register(
    id="catanatronp4-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv4",
)
