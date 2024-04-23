import pdb
import random
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import CITY, RESOURCES, ROAD, SETTLEMENT, VICTORY_POINT, Action, ActionType, FastResource
from catanatron.models.board import get_edges
from catanatron.players.tracker import CardCounting
from catanatron.state_functions import calculate_resource_probabilities, get_dev_cards_in_hand, get_largest_army, get_longest_road_color, get_player_buildings, player_key
from catanatron_gym.features import (
    create_sample,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)

from catanatron.state_functions import calculate_resource_probabilities, get_dev_cards_in_hand, get_largest_army, get_longest_road_color, get_player_buildings, player_key


BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]
ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    # TODO: One for each tile (and abuse 1v1 setting).
    *[(ActionType.MOVE_ROBBER, tile) for tile in TILE_COORDINATES],
    (ActionType.DISCARD, None),
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    (ActionType.PLAY_KNIGHT_CARD, None),
    *[
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCES[j]))
        for i, first_card in enumerate(RESOURCES)
        for j in range(i, len(RESOURCES))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCES],
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
    # 4:1 with bank
    *[
        (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 3:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 2:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    (ActionType.END_TURN, None),
]
ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)
ACTION_TYPES = [i for i in ActionType]


def to_action_type_space(action):
    return ACTION_TYPES.index(action.action_type)


def normalize_action(action):
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)
    return normalized


def to_action_space(action):
    """maps action to space_action equivalent integer"""
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


def from_action_space(action_int, playable_actions):
    """maps action_int to catantron.models.actions.Action"""
    # Get "catan_action" based on space action.
    # i.e. Take first action in playable that matches ACTIONS_ARRAY blueprint
    (action_type, value) = ACTIONS_ARRAY[action_int]
    catan_action = None
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            catan_action = action
            break  # return the first one
    assert catan_action is not None
    return catan_action


FEATURES = get_feature_ordering(num_players=4)
NUM_FEATURES = len(FEATURES)

# Highest features is NUM_RESOURCES_IN_HAND which in theory is all resource cards
HIGH = 19 * 5


def simple_reward(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return 0
    else:
        return -1
    
def weighted_reward(game, p0_color):
    key = player_key(p0_color)

    end_points = game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    cities = len(get_player_buildings(game.state, p0_color, CITY))
    settlements = len(get_player_buildings(game.state, p0_color, SETTLEMENT))
    road = len(get_player_buildings(game.state, p0_color, ROAD))
    longest = get_longest_road_color(game.state) == p0_color
    largest = get_largest_army(game.state)[0] == p0_color
    devvps = get_dev_cards_in_hand(game.state, p0_color, VICTORY_POINT)
    probabilities = calculate_resource_probabilities(game.state)
    resource_production = { resource: probabilities[p0_color][resource] for resource in FastResource }
    total_resources_gained = game.my_card_counter.total_resources_gained[p0_color]
    amount_of_resources_used = game.my_card_counter.total_resources_used[p0_color]
    seven_robbers_moved = game.my_card_counter.total_robbers_moved[p0_color] - game.state.player_state[f"{key}_PLAYED_KNIGHT"]
    knights_and_robbers_moved = game.my_card_counter.total_robbers_moved[p0_color]
    total_robber_gain = game.my_card_counter.total_robber_gain[p0_color]
    total_resources_lost = game.my_card_counter.total_resources_lost[p0_color]
    total_resources_discarded = game.my_card_counter.total_resources_discarded[p0_color]

def game_stat_reward_OLD(game, key, previous_points, p0_color):

    current_points = game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    points_list = [game.state.player_state[f"{player_key(game.state, p.color)}_ACTUAL_VICTORY_POINTS"] for p in game.state.players]
    
    max_points = max(points_list)
    second_max_points = max(points_list, key=lambda x: (points_list.count(max_points) > 1, x != max_points))

    if p0_color == game.winning_color():
        leading_points = second_max_points  # If agent is winning, compare to second best
    else:
        leading_points = max_points
    
    # Calculate change in points for this step
    point_change = current_points - (previous_points or 0)

    # Cap the current points at 10 for the reward calculation to avoid extra reward for 11!
    adjusted_points = min(current_points, 10)

    # Reward based on closeness to winning the game, increasing exponentially as points approach 10
    proximity_to_goal_reward = (2 ** (adjusted_points - 1)) * 10  # Exponential scaling

    # Encourage keeping up with or staying ahead of the best player
    relative_performance_reward = (current_points - leading_points) * 10

    # Calculate overall relative performance against all players
    average_points = sum(points_list) / len(points_list)
    performance_against_average = (current_points - average_points) * 5

    total_reward = proximity_to_goal_reward + relative_performance_reward + performance_against_average + point_change * 10

    return total_reward

def game_stat_reward(game, key, previous_points, p0_color):
    current_points = game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    # points_list = [game.state.player_state[f"{player_key(game.state, p.color)}_ACTUAL_VICTORY_POINTS"] for p in game.state.players]

    # max_points = max(points_list)
    # points_list_sorted = sorted(points_list, reverse=True)

    # if p0_color == game.winning_color():
    #     leading_points = points_list_sorted[1] if len(points_list) > 1 else current_points  # Compare to second best if it exists
    # else:
    #     leading_points = max_points

    # Calculate change in points for this step
    point_change = current_points - (previous_points or 0)

    # Adjust points calculation
    adjusted_points = min(current_points, 10)

    # else -10000 * (max_points - current_points) if max_points >= 10 else 0
    # Exponential reward based on closeness to winning
    proximity_to_goal_reward = (2 ** (adjusted_points - 1))  # More emphasis on exponential scaling

    # Simplify the relative performance reward to only consider comparison with the closest competitor
    # relative_performance_reward = max(0, current_points - leading_points) * 100 if p0_color != game.winning_color() else 0

    # Performance penalty for not advancing from initial points
    no_progress_penalty = -3 if current_points == 2 else 0

    total_reward = proximity_to_goal_reward + no_progress_penalty + point_change * 10 # + relative_performance_reward
    # Major reward for winning the game
    # win_reward = 50000 
    if current_points > 9:
        total_reward += 100000

    return total_reward


def game_stat_rewardSimple(game, key, previous_points, p0_color):
    current_points = game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    # points_list = [game.state.player_state[f"{player_key(game.state, p.color)}_ACTUAL_VICTORY_POINTS"] for p in game.state.players]

    # max_points = max(points_list)
    # points_list_sorted = sorted(points_list, reverse=True)

    # if p0_color == game.winning_color():
    #     leading_points = points_list_sorted[1] if len(points_list) > 1 else current_points  # Compare to second best if it exists
    # else:
    #     leading_points = max_points

    # Calculate change in points for this step
    # point_change = current_points - (previous_points or 0)

    # Adjust points calculation
    adjusted_points = min(current_points, 10)

    # else -10000 * (max_points - current_points) if max_points >= 10 else 0
    # Exponential reward based on closeness to winning
    proximity_to_goal_reward = ((adjusted_points - 1) ** 2) ** 2  # More emphasis on exponential scaling

    # Simplify the relative performance reward to only consider comparison with the closest competitor
    # relative_performance_reward = max(0, current_points - leading_points) * 100 if p0_color != game.winning_color() else 0

    # Performance penalty for not advancing from initial points
    no_progress_penalty = - 5 if current_points == 2 else 0

    total_reward = proximity_to_goal_reward + no_progress_penalty # + point_change * 10 # + relative_performance_reward
    # Major reward for winning the game
    # win_reward = 50000 
    if current_points > 9:
        total_reward += 100000

    return total_reward

def game_stat_reward_comparisons(game, key, previous_points, p0_color):
    current_points = game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    points_list = [game.state.player_state[f"{player_key(game.state, p.color)}_ACTUAL_VICTORY_POINTS"] for p in game.state.players]

    max_points = max(points_list)
    points_list_sorted = sorted(points_list, reverse=True)

    if p0_color == game.winning_color():
        leading_points = points_list_sorted[1] if len(points_list) > 1 else current_points  # Compare to second best if it exists
    else:
        leading_points = max_points

    # Calculate change in points for this step
    # point_change = current_points - (previous_points or 0)

    # Adjust points calculation
    adjusted_points = min(current_points, 10)

    # else -10000 * (max_points - current_points) if max_points >= 10 else 0
    # Exponential reward based on closeness to winning
    proximity_to_goal_reward = ((adjusted_points - 1) ** 2) ** 2  # More emphasis on exponential scaling

    # Simplify the relative performance reward to only consider comparison with the closest competitor
    relative_performance_reward = max(0, current_points - leading_points) * 100 + 5000 # if p0_color != game.winning_color() else 0

    # Performance penalty for not advancing from initial points
    no_progress_penalty = - 5 if current_points == 2 else 0


    total_reward = proximity_to_goal_reward + no_progress_penalty + relative_performance_reward
    # Major reward for winning the game
    # win_reward = 50000 
    if current_points > 9:
        total_reward += 1000000

    return total_reward

# def game_stat_reward_comparisons(game, key, previous_points, p0_color):



    # return 0

def point_exponentiation_reward(game, key, previous_points, p0_color):
    current_points = game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    turn_count = game.state.num_turns + 1

    adjusted_points = min(current_points, 10)
    # if current_points >= 10:
    #     total = (pow(current_points, current_points))
    # else:
    total = (pow(adjusted_points, adjusted_points) / turn_count)

    return total



class CatanatronEnvReward(gym.Env):
    metadata = {"render_modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    reward_range = (-1, 1)

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", point_exponentiation_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        random.shuffle(self.players)
        # print('Players: ', self.players)
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space smaller if possible (per map_type)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        # self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        if self.game.state.current_color() == self.p0.color:
            key = player_key(self.game.state, self.p0.color)
            previous_points = self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        try:
            catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(valid_actions=self.get_valid_actions())
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, key, previous_points, self.p0.color)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.my_card_counter = CardCounting(players=self.players, color=self.p0.color)
        # for enemy in self.enemies:
        #   self.opponent_card_counter = CardCounting(players=self.players, color=enemy.color)

        catan_map = build_map(self.map_type)

        for player in self.players:
            player.reset_state()
        random.shuffle(self.players)
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
            trackers=[self.my_card_counter],
        )
        self.invalid_actions_count = 0

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot


class CatanatronEnv(gym.Env):
    metadata = {"render_modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    reward_range = (-1, 1)

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        random.shuffle(self.players)
        # print('Players: ', self.players)
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space smaller if possible (per map_type)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        # self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        try:
            catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(valid_actions=self.get_valid_actions())
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, self.p0.color)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.my_card_counter = CardCounting(players=self.players, color=self.p0.color)
        # for enemy in self.enemies:
        #   self.opponent_card_counter = CardCounting(players=self.players, color=enemy.color)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        random.shuffle(self.players)
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
            trackers=[self.my_card_counter],
        )
        self.invalid_actions_count = 0

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot


CatanatronEnv.__doc__ = f"""
1v3 environment against a random player, alpha-beta player and same-turn alpha-beta player.

Attributes:
    reward_range: -1 if player lost, 1 if player won, 0 otherwise.
    action_space: Integers from the [0, 289] interval. 
        See Action Space table below.
    observation_space: Numeric Feature Vector. See Observation Space table 
        below for quantities. They appear in vector in alphabetical order,
        from the perspective of "current" player (hiding/showing information
        accordingly). P0 is "current" player. P1 is next in line.
        
        We use the following nomenclature for Tile ids and Node ids.
        Edge ids are self-describing (node-id, node-id) tuples. We also
        use Cube coordinates for tiles (see 
        https://www.redblobgames.com/grids/hexagons/#coordinates)

.. image:: _static/tile-ids.png
  :width: 300
  :alt: Tile Ids
.. image:: _static/node-ids.png
  :width: 300
  :alt: Node Ids

.. list-table:: Action Space
   :widths: 10 100
   :header-rows: 1

   * - Integer
     - Catanatron Action
"""
for i, v in enumerate(ACTIONS_ARRAY):
    CatanatronEnv.__doc__ += f"   * - {i}\n     - {v}\n"

CatanatronEnv.__doc__ += """

.. list-table:: Observation Space (Raw)
   :widths: 10 50 10 10
   :header-rows: 1

   * - Feature Name
     - Description
     - Number of Features (N=number of players)
     - Type

   * - BANK_<resource>
     - Number of cards of that `resource` in bank
     - 5
     - Integer
   * - BANK_DEV_CARDS
     - Number of development cards in bank
     - 1
     - Integer
    
   * - EDGE<i>_P<j>_ROAD
     - Whether edge `i` is owned by player `j`
     - 72 * N
     - Boolean
   * - NODE<i>_P<j>_SETTLEMENT
     - Whether player `j` has a city in node `i`
     - 54 * N
     - Boolean
   * - NODE<i>_P<j>_CITY
     - Whether player `j` has a city in node `i`
     - 54 * N
     - Boolean
   * - PORT<i>_IS_<resource>
     - Whether node `i` is port of `resource` (or THREE_TO_ONE).
     - 9 * 6
     - Boolean
   * - TILE<i>_HAS_ROBBER
     - Whether robber is on tile `i`.
     - 19
     - Boolean
   * - TILE<i>_IS_<resource>
     - Whether tile `i` yields `resource` (or DESERT).
     - 19 * 6
     - Boolean
   * - TILE<i>_PROBA
     - Tile `i`'s probability of being rolled.
     - 19
     - Float

   * - IS_DISCARDING
     - Whether current player must discard. For now, there is only 1 
       discarding action (at random), since otherwise action space
       would explode in size.
     - 1
     - Boolean
   * - IS_MOVING_ROBBER
     - Whether current player must move robber (because played knight
       or because rolled a 7).
     - 1
     - Boolean
   * - P<i>_HAS_ROLLED
     - Whether player `i` already rolled dice.
     - N
     - Boolean
   * - P0_HAS_PLAYED _DEVELOPMENT_CARD _IN_TURN
     - Whether current player already played a development card
     - 1
     - Boolean

   * - P0_ACTUAL_VPS
     - Total Victory Points (including Victory Point Development Cards)
     - 1
     - Integer
   * - P0_<resource>_IN_HAND
     - Number of `resource` cards in hand
     - 5
     - Integer
   * - P0_<dev-card>_IN_HAND
     - Number of `dev-card` cards in hand
     - 5
     - Integer
   * - P<i>_NUM_DEVS_IN_HAND
     - Number of hidden development cards player `i` has
     - N
     - Integer
   * - P<i>_NUM_RESOURCES _IN_HAND
     - Number of hidden resource cards player `i` has
     - N
     - Integer

   * - P<i>_HAS_ARMY
     - Whether player <i> has Largest Army
     - N
     - Boolean
   * - P<i>_HAS_ROAD
     - Whether player <i> has Longest Road
     - N
     - Boolean
   * - P<i>_ROADS_LEFT
     - Number of roads pieces player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_SETTLEMENTS_LEFT
     - Number of settlements player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_CITIES_LEFT
     - Number of cities player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_LONGEST_ROAD _LENGTH
     - Length of longest road by player `i`
     - N
     - Integer
   * - P<i>_PUBLIC_VPS
     - Amount of visible victory points for player `i` (i.e.
       doesn't include hidden victory point cards; only army,
       road and settlements/cities).
     - N
     - Integer
   * - P<i>_<dev-card>_PLAYED
     - Amount of `dev-card` cards player `i` has played in game
       (VICTORY_POINT not included).
     - 4 * N
     - Integer
   * - 
     - 
     - 194 * N + 226
     - 
"""






class CatanatronEnv3(gym.Env):
    metadata = {"render_modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    reward_range = (-1, 1)

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        myplayers = self.enemies
        random.shuffle(myplayers)
        myplayers.insert(2, self.p0)
        self.players = myplayers
        print('ma playaz: ',self.players)
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10
        self.initial_observation = None
        self.initial_info = None

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        # self.reset()
        # self.initial_observation, self.initial_info = self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        try:
            catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(valid_actions=self.get_valid_actions())
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, self.p0.color)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)
        self.my_card_counter = CardCounting(players=self.players, color=self.p0.color)
        # for enemy in self.enemies:
        #   self.opponent_card_counter = CardCounting(players=self.players, color=enemy.color)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
            trackers=[self.my_card_counter],
        )
        self.invalid_actions_count = 0

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot






class CatanatronEnv2(gym.Env):
    metadata = {"render_modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    reward_range = (-1, 1)

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        myplayers = self.enemies
        random.shuffle(myplayers)
        myplayers.insert(1, self.p0)
        self.players = myplayers
        print('ma playaz: ',self.players)
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space smaller if possible (per map_type)
        # self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        # self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        try:
            catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(valid_actions=self.get_valid_actions())
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, self.p0.color)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.my_card_counter = CardCounting(players=self.players, color=self.p0.color)
        # for enemy in self.enemies:
        #   self.opponent_card_counter = CardCounting(players=self.players, color=enemy.color)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
            trackers=[self.my_card_counter],
        )
        self.invalid_actions_count = 0

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot



class CatanatronEnv1(gym.Env):
    metadata = {"render_modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    reward_range = (-1, 1)

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        random.shuffle(self.enemies)
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        print('ma playaz: ',self.players)
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space smaller if possible (per map_type)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        # self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        try:
            catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(valid_actions=self.get_valid_actions())
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, self.p0.color)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.my_card_counter = CardCounting(players=self.players, color=self.p0.color)
        # for enemy in self.enemies:
        #   self.opponent_card_counter = CardCounting(players=self.players, color=enemy.color)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
            trackers=[self.my_card_counter],
        )
        self.invalid_actions_count = 0

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot




class CatanatronEnv4(gym.Env):
    metadata = {"render_modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    reward_range = (-1, 1)

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        random.shuffle(self.enemies)
        self.p0 = Player(Color.BLUE)
        self.players = self.enemies + [self.p0]  # type: ignore
        print('ma playaz: ',self.players)
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space smaller if possible (per map_type)
        # self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        # self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        result = 0
        print(f"BEFORE: {action}, Current Player: {self.game.state.current_player()}, Turn: {self.game.state.num_turns}")
        try:
            catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(valid_actions=self.get_valid_actions())
            result = 'excepted'
            print(f"After action: {result}, New Current Player: {self.game.state.current_player()}, Turn: {self.game.state.num_turns}")
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, self.p0.color)
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.my_card_counter = CardCounting(players=self.players, color=self.p0.color)
        # for enemy in self.enemies:
        #   self.opponent_card_counter = CardCounting(players=self.players, color=enemy.color)

        self.map_type = self.config.get("map_type", "BASE")
        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        print('so stuffs happenin pre game')
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
            trackers=[self.my_card_counter],
        )
        print('so stuffs happenin post game pre advance')
        self.invalid_actions_count = 0

        self._advance_until_p0_decision()

        print('so stuffs happenin post advance pre ob')
        observation = self._get_observation()
        print('so stuffs happenin post observation')
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot