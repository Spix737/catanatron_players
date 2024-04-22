from gymnasium.envs.registration import register

# 4p x 1, random
register(
    id="catanatron-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv",
)
# 4p x 1, random, amended reward function
register(
    id="catanatronReward-v1",
    entry_point="catanatron_gym.envs:CatanatronEnvReward",
)
# 4p x 1, 3rd
register(
    id="catanatronp3-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv3",
)
# 4p x 1, 2nd
register(
    id="catanatronp2-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv2",
)
# 4p x 1, 1st
register(
    id="catanatronp1-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv1",
)
# 4p x 1, 4th
register(
    id="catanatronp4-v1",
    entry_point="catanatron_gym.envs:CatanatronEnv4",
)
