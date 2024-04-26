from typing import Any, List, Literal, Tuple
import functools
from collections import Counter
from catanatron.models.actions import generate_playable_actions
from catanatron.models.decks import freqdeck_count

import networkx as nx

from catanatron.players.tracker import CardCounting
from catanatron.state_functions import (
    calculate_resource_probabilities,
    get_player_buildings,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.models.board import STATIC_GRAPH, get_edges, get_node_distances
from catanatron.models.map import NUM_TILES, CatanMap, build_map
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import (
    ASSUMED_RESOURCES,
    BRICK,
    DEVELOPMENT_CARDS,
    ORE,
    RESOURCES,
    SETTLEMENT,
    CITY,
    ROAD,
    SHEEP,
    WHEAT,
    WOOD,
    ActionType,
    VICTORY_POINT,
)
from catanatron.game import Game
from catanatron.models.map import number_probability


# ===== Helpers
def is_building(game, node_id, color, building_type):
    building = game.state.board.buildings.get(node_id, None)
    if building is None:
        return False
    else:
        return building[0] == color and building[1] == building_type


def is_road(game, edge, color):
    return game.state.board.get_edge_color(edge) == color


@functools.lru_cache(1024)
def iter_players(colors: Tuple[Color], p0_color: Color):
    """Iterator: for i, player in iter_players(game, p0.color)"""
    start_index = colors.index(p0_color)
    result = []
    for i in range(len(colors)):
        actual_index = (start_index + i) % len(colors)
        result.append((i, colors[actual_index]))
    return result


# ===== Extractors
def player_features(game: Game, p0_color: Color):
    # P0_ACTUAL_VPS
    # P{i}_PUBLIC_VPS, P1_PUBLIC_VPS, ...
    # P{i}_HAS_ARMY, P{i}_HAS_ROAD, P1_HAS_ARMY, ...
    # P{i}_ROADS_LEFT, P{i}_SETTLEMENTS_LEFT, P{i}_CITIES_LEFT, P1_...
    # P{i}_HAS_ROLLED, P{i}_LONGEST_ROAD_LENGTH
    features = dict()
    for i, color in iter_players(game.state.colors, p0_color):
        key = player_key(game.state, color)
        if color == p0_color:
            features["P0_ACTUAL_VPS"] = game.state.player_state[
                key + "_ACTUAL_VICTORY_POINTS"
            ]

        features[f"P{i}_PUBLIC_VPS"] = game.state.player_state[key + "_VICTORY_POINTS"]
        features[f"P{i}_HAS_ARMY"] = game.state.player_state[key + "_HAS_ARMY"]
        features[f"P{i}_HAS_ROAD"] = game.state.player_state[key + "_HAS_ROAD"]
        features[f"P{i}_ROADS_LEFT"] = game.state.player_state[key + "_ROADS_AVAILABLE"]
        features[f"P{i}_SETTLEMENTS_LEFT"] = game.state.player_state[
            key + "_SETTLEMENTS_AVAILABLE"
        ]
        features[f"P{i}_CITIES_LEFT"] = game.state.player_state[
            key + "_CITIES_AVAILABLE"
        ]
        features[f"P{i}_HAS_ROLLED"] = game.state.player_state[key + "_HAS_ROLLED"]
        features[f"P{i}_LONGEST_ROAD_LENGTH"] = game.state.player_state[
            key + "_LONGEST_ROAD_LENGTH"
        ]

    return features


def tracked_features(game: Game, p0_color: Color):
    features = dict()
    try:
        for resource_tracker in game.trackers:
            color = resource_tracker.color
            if resource_tracker.color == p0_color:
                for i, color in iter_players(game.state.colors, color):
                    for resource in ASSUMED_RESOURCES:
                        features[
                            f"P0_ASSUMED_{i}_{resource}_IN_HAND"
                        ] = resource_tracker.assumed_resources[color][resource]
    except Exception as e:
        print(e)
        print("No trackers found, thus no features to extract.")
    return features


ACTIONS_ARRAY = [
    (Color.BLUE, ActionType.ROLL, None),
    (Color.BLUE, ActionType.MOVE_ROBBER, (0, 0, 0)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (1, -1, 0)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (0, -1, 1)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (-1, 0, 1)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (-1, 1, 0)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (0, 1, -1)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (1, 0, -1)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (2, -2, 0)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (1, -2, 1)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (0, -2, 2)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (-1, -1, 2)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (-2, 0, 2)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (-2, 1, 1)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (-2, 2, 0)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (-1, 2, -1)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (0, 2, -2)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (1, 1, -2)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (2, 0, -2)),
    (Color.BLUE, ActionType.MOVE_ROBBER, (2, -1, -1)),
    (Color.BLUE, ActionType.DISCARD, None),
    (Color.BLUE, ActionType.BUILD_ROAD, (0, 5)),
    (Color.BLUE, ActionType.BUILD_ROAD, (0, 1)),
    (Color.BLUE, ActionType.BUILD_ROAD, (0, 20)),
    (Color.BLUE, ActionType.BUILD_ROAD, (1, 2)),
    (Color.BLUE, ActionType.BUILD_ROAD, (1, 6)),
    (Color.BLUE, ActionType.BUILD_ROAD, (2, 3)),
    (Color.BLUE, ActionType.BUILD_ROAD, (2, 9)),
    (Color.BLUE, ActionType.BUILD_ROAD, (3, 4)),
    (Color.BLUE, ActionType.BUILD_ROAD, (3, 12)),
    (Color.BLUE, ActionType.BUILD_ROAD, (4, 5)),
    (Color.BLUE, ActionType.BUILD_ROAD, (4, 15)),
    (Color.BLUE, ActionType.BUILD_ROAD, (5, 16)),
    (Color.BLUE, ActionType.BUILD_ROAD, (6, 7)),
    (Color.BLUE, ActionType.BUILD_ROAD, (6, 23)),
    (Color.BLUE, ActionType.BUILD_ROAD, (7, 8)),
    (Color.BLUE, ActionType.BUILD_ROAD, (7, 24)),
    (Color.BLUE, ActionType.BUILD_ROAD, (8, 9)),
    (Color.BLUE, ActionType.BUILD_ROAD, (8, 27)),
    (Color.BLUE, ActionType.BUILD_ROAD, (9, 10)),
    (Color.BLUE, ActionType.BUILD_ROAD, (10, 11)),
    (Color.BLUE, ActionType.BUILD_ROAD, (10, 29)),
    (Color.BLUE, ActionType.BUILD_ROAD, (11, 12)),
    (Color.BLUE, ActionType.BUILD_ROAD, (11, 32)),
    (Color.BLUE, ActionType.BUILD_ROAD, (12, 13)),
    (Color.BLUE, ActionType.BUILD_ROAD, (13, 14)),
    (Color.BLUE, ActionType.BUILD_ROAD, (13, 34)),
    (Color.BLUE, ActionType.BUILD_ROAD, (14, 15)),
    (Color.BLUE, ActionType.BUILD_ROAD, (14, 37)),
    (Color.BLUE, ActionType.BUILD_ROAD, (15, 17)),
    (Color.BLUE, ActionType.BUILD_ROAD, (16, 18)),
    (Color.BLUE, ActionType.BUILD_ROAD, (16, 21)),
    (Color.BLUE, ActionType.BUILD_ROAD, (17, 18)),
    (Color.BLUE, ActionType.BUILD_ROAD, (17, 39)),
    (Color.BLUE, ActionType.BUILD_ROAD, (18, 40)),
    (Color.BLUE, ActionType.BUILD_ROAD, (19, 21)),
    (Color.BLUE, ActionType.BUILD_ROAD, (19, 20)),
    (Color.BLUE, ActionType.BUILD_ROAD, (19, 46)),
    (Color.BLUE, ActionType.BUILD_ROAD, (20, 22)),
    (Color.BLUE, ActionType.BUILD_ROAD, (21, 43)),
    (Color.BLUE, ActionType.BUILD_ROAD, (22, 23)),
    (Color.BLUE, ActionType.BUILD_ROAD, (22, 49)),
    (Color.BLUE, ActionType.BUILD_ROAD, (23, 52)),
    (Color.BLUE, ActionType.BUILD_ROAD, (24, 25)),
    (Color.BLUE, ActionType.BUILD_ROAD, (24, 53)),
    (Color.BLUE, ActionType.BUILD_ROAD, (25, 26)),
    (Color.BLUE, ActionType.BUILD_ROAD, (26, 27)),
    (Color.BLUE, ActionType.BUILD_ROAD, (27, 28)),
    (Color.BLUE, ActionType.BUILD_ROAD, (28, 29)),
    (Color.BLUE, ActionType.BUILD_ROAD, (29, 30)),
    (Color.BLUE, ActionType.BUILD_ROAD, (30, 31)),
    (Color.BLUE, ActionType.BUILD_ROAD, (31, 32)),
    (Color.BLUE, ActionType.BUILD_ROAD, (32, 33)),
    (Color.BLUE, ActionType.BUILD_ROAD, (33, 34)),
    (Color.BLUE, ActionType.BUILD_ROAD, (34, 35)),
    (Color.BLUE, ActionType.BUILD_ROAD, (35, 36)),
    (Color.BLUE, ActionType.BUILD_ROAD, (36, 37)),
    (Color.BLUE, ActionType.BUILD_ROAD, (37, 38)),
    (Color.BLUE, ActionType.BUILD_ROAD, (38, 39)),
    (Color.BLUE, ActionType.BUILD_ROAD, (39, 41)),
    (Color.BLUE, ActionType.BUILD_ROAD, (40, 42)),
    (Color.BLUE, ActionType.BUILD_ROAD, (40, 44)),
    (Color.BLUE, ActionType.BUILD_ROAD, (41, 42)),
    (Color.BLUE, ActionType.BUILD_ROAD, (43, 44)),
    (Color.BLUE, ActionType.BUILD_ROAD, (43, 47)),
    (Color.BLUE, ActionType.BUILD_ROAD, (45, 47)),
    (Color.BLUE, ActionType.BUILD_ROAD, (45, 46)),
    (Color.BLUE, ActionType.BUILD_ROAD, (46, 48)),
    (Color.BLUE, ActionType.BUILD_ROAD, (48, 49)),
    (Color.BLUE, ActionType.BUILD_ROAD, (49, 50)),
    (Color.BLUE, ActionType.BUILD_ROAD, (50, 51)),
    (Color.BLUE, ActionType.BUILD_ROAD, (51, 52)),
    (Color.BLUE, ActionType.BUILD_ROAD, (52, 53)),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 0),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 1),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 2),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 3),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 4),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 6),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 7),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 8),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 9),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 10),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 11),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 12),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 13),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 14),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 15),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 16),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 17),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 18),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 19),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 20),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 21),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 22),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 23),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 24),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 25),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 26),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 27),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 28),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 29),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 30),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 31),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 32),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 33),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 34),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 35),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 36),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 37),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 38),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 39),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 40),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 41),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 42),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 43),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 44),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 45),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 46),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 47),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 48),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 49),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 50),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 51),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 52),
    (Color.BLUE, ActionType.BUILD_SETTLEMENT, 53),
    (Color.BLUE, ActionType.BUILD_CITY, 0),
    (Color.BLUE, ActionType.BUILD_CITY, 1),
    (Color.BLUE, ActionType.BUILD_CITY, 2),
    (Color.BLUE, ActionType.BUILD_CITY, 3),
    (Color.BLUE, ActionType.BUILD_CITY, 4),
    (Color.BLUE, ActionType.BUILD_CITY, 5),
    (Color.BLUE, ActionType.BUILD_CITY, 6),
    (Color.BLUE, ActionType.BUILD_CITY, 7),
    (Color.BLUE, ActionType.BUILD_CITY, 8),
    (Color.BLUE, ActionType.BUILD_CITY, 9),
    (Color.BLUE, ActionType.BUILD_CITY, 10),
    (Color.BLUE, ActionType.BUILD_CITY, 11),
    (Color.BLUE, ActionType.BUILD_CITY, 12),
    (Color.BLUE, ActionType.BUILD_CITY, 13),
    (Color.BLUE, ActionType.BUILD_CITY, 14),
    (Color.BLUE, ActionType.BUILD_CITY, 15),
    (Color.BLUE, ActionType.BUILD_CITY, 16),
    (Color.BLUE, ActionType.BUILD_CITY, 17),
    (Color.BLUE, ActionType.BUILD_CITY, 18),
    (Color.BLUE, ActionType.BUILD_CITY, 19),
    (Color.BLUE, ActionType.BUILD_CITY, 20),
    (Color.BLUE, ActionType.BUILD_CITY, 21),
    (Color.BLUE, ActionType.BUILD_CITY, 22),
    (Color.BLUE, ActionType.BUILD_CITY, 23),
    (Color.BLUE, ActionType.BUILD_CITY, 24),
    (Color.BLUE, ActionType.BUILD_CITY, 25),
    (Color.BLUE, ActionType.BUILD_CITY, 26),
    (Color.BLUE, ActionType.BUILD_CITY, 27),
    (Color.BLUE, ActionType.BUILD_CITY, 28),
    (Color.BLUE, ActionType.BUILD_CITY, 29),
    (Color.BLUE, ActionType.BUILD_CITY, 30),
    (Color.BLUE, ActionType.BUILD_CITY, 31),
    (Color.BLUE, ActionType.BUILD_CITY, 32),
    (Color.BLUE, ActionType.BUILD_CITY, 33),
    (Color.BLUE, ActionType.BUILD_CITY, 34),
    (Color.BLUE, ActionType.BUILD_CITY, 35),
    (Color.BLUE, ActionType.BUILD_CITY, 36),
    (Color.BLUE, ActionType.BUILD_CITY, 37),
    (Color.BLUE, ActionType.BUILD_CITY, 38),
    (Color.BLUE, ActionType.BUILD_CITY, 39),
    (Color.BLUE, ActionType.BUILD_CITY, 40),
    (Color.BLUE, ActionType.BUILD_CITY, 41),
    (Color.BLUE, ActionType.BUILD_CITY, 42),
    (Color.BLUE, ActionType.BUILD_CITY, 43),
    (Color.BLUE, ActionType.BUILD_CITY, 44),
    (Color.BLUE, ActionType.BUILD_CITY, 45),
    (Color.BLUE, ActionType.BUILD_CITY, 46),
    (Color.BLUE, ActionType.BUILD_CITY, 47),
    (Color.BLUE, ActionType.BUILD_CITY, 48),
    (Color.BLUE, ActionType.BUILD_CITY, 49),
    (Color.BLUE, ActionType.BUILD_CITY, 50),
    (Color.BLUE, ActionType.BUILD_CITY, 51),
    (Color.BLUE, ActionType.BUILD_CITY, 52),
    (Color.BLUE, ActionType.BUILD_CITY, 53),
    (Color.BLUE, ActionType.BUY_DEVELOPMENT_CARD, None),
    (Color.BLUE, ActionType.PLAY_KNIGHT_CARD, None),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WOOD, WOOD)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WOOD, BRICK)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WOOD, SHEEP)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WOOD, WHEAT)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WOOD, ORE)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (BRICK, BRICK)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (BRICK, SHEEP)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (BRICK, WHEAT)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (BRICK, ORE)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (SHEEP, SHEEP)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (SHEEP, WHEAT)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (SHEEP, ORE)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WHEAT, WHEAT)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WHEAT, ORE)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (ORE, ORE)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WOOD,)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (BRICK,)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (SHEEP,)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (WHEAT,)),
    (Color.BLUE, ActionType.PLAY_YEAR_OF_PLENTY, (ORE,)),
    (Color.BLUE, ActionType.PLAY_ROAD_BUILDING, None),
    (Color.BLUE, ActionType.PLAY_MONOPOLY, WOOD),
    (Color.BLUE, ActionType.PLAY_MONOPOLY, BRICK),
    (Color.BLUE, ActionType.PLAY_MONOPOLY, SHEEP),
    (Color.BLUE, ActionType.PLAY_MONOPOLY, WHEAT),
    (Color.BLUE, ActionType.PLAY_MONOPOLY, ORE),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, WOOD, WOOD, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, WOOD, WOOD, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, WOOD, WOOD, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, WOOD, WOOD, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, BRICK, BRICK, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, BRICK, BRICK, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, BRICK, BRICK, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, BRICK, BRICK, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, SHEEP, SHEEP, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, SHEEP, SHEEP, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, SHEEP, SHEEP, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, SHEEP, SHEEP, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, WHEAT, WHEAT, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, WHEAT, WHEAT, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, WHEAT, WHEAT, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, WHEAT, WHEAT, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, ORE, ORE, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, ORE, ORE, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, ORE, ORE, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, ORE, ORE, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, WOOD, None, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, WOOD, None, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, WOOD, None, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, WOOD, None, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, BRICK, None, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, BRICK, None, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, BRICK, None, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, BRICK, None, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, SHEEP, None, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, SHEEP, None, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, SHEEP, None, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, SHEEP, None, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, WHEAT, None, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, WHEAT, None, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, WHEAT, None, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, WHEAT, None, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, ORE, None, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, ORE, None, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, ORE, None, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, ORE, None, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, None, None, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, None, None, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, None, None, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WOOD, WOOD, None, None, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, None, None, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, None, None, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, None, None, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (BRICK, BRICK, None, None, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, None, None, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, None, None, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, None, None, WHEAT)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (SHEEP, SHEEP, None, None, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, None, None, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, None, None, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, None, None, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (WHEAT, WHEAT, None, None, ORE)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, None, None, WOOD)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, None, None, BRICK)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, None, None, SHEEP)),
    (Color.BLUE, ActionType.MARITIME_TRADE, (ORE, ORE, None, None, WHEAT)),
    (Color.BLUE, ActionType.END_TURN, None),
]


def available_actions_features(game: Game, p0_color: Color):
    features = {}
    key = player_key(game.state, p0_color)

    # Generate the list of playable actions for the current state
    playable_actions = set([a for a in game.state.playable_actions])

    # If there are no actions or the first action is not for our player, it's not our turn
    for i, color in iter_players(game.state.colors, p0_color):
        if color == p0_color:
            if not playable_actions or not any(
                action.color == p0_color for action in playable_actions
            ):
                # Set all actions to 0 since it's not our turn
                for action in ACTIONS_ARRAY:
                    action_type = action[1]
                    action_value = action[2]
                    simple_action_type = str(action_type).replace("ActionType.", "")
                    features[f"P0_can_{simple_action_type}_{action_value}"] = 0
            else:
                # Check each predefined action and mark it as playable (1) or not playable (0)
                for action in ACTIONS_ARRAY:
                    action_type = action[1]
                    action_value = action[2]
                    simple_action_type = str(action_type).replace("ActionType.", "")
                    features[f"P0_can_{simple_action_type}_{action_value}"] = (
                        1 if action in playable_actions else 0
                    )

    return features


def resource_hand_features(game: Game, p0_color: Color):
    # P0_WHEATS_IN_HAND, P0_WOODS_IN_HAND, ...
    # P0_ROAD_BUILDINGS_IN_HAND, P0_KNIGHT_IN_HAND, ..., P0_VPS_IN_HAND
    # P0_ROAD_BUILDINGS_PLAYABLE, P0_KNIGHT_PLAYABLE, ...
    # P0_ROAD_BUILDINGS_PLAYED, P0_KNIGHT_PLAYED, ...

    # P1_ROAD_BUILDINGS_PLAYED, P1_KNIGHT_PLAYED, ...
    # TODO: P1_WHEATS_INFERENCE, P1_WOODS_INFERENCE, ...
    # TODO: P1_ROAD_BUILDINGS_INFERENCE, P1_KNIGHT_INFERENCE, ...

    state = game.state
    player_state = state.player_state

    features = {}
    for i, color in iter_players(game.state.colors, p0_color):
        key = player_key(game.state, color)

        if color == p0_color:
            for resource in RESOURCES:
                features[f"P0_{resource}_IN_HAND"] = player_state[
                    key + f"_{resource}_IN_HAND"
                ]
            for card in DEVELOPMENT_CARDS:
                features[f"P0_{card}_IN_HAND"] = player_state[key + f"_{card}_IN_HAND"]
            features[f"P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = player_state[
                key + "_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"
            ]

        for card in DEVELOPMENT_CARDS:
            if card == VICTORY_POINT:
                continue  # cant play VPs
            features[f"P{i}_{card}_PLAYED"] = player_state[key + f"_PLAYED_{card}"]

        features[f"P{i}_NUM_RESOURCES_IN_HAND"] = player_num_resource_cards(
            state, color
        )
        features[f"P{i}_NUM_DEVS_IN_HAND"] = player_num_dev_cards(state, color)

        probablilities = calculate_resource_probabilities(state)
        for resource in RESOURCES:
            features[f"P{i}_{resource}_PRODUCTION"] = probablilities[color][resource]

    return features


@functools.lru_cache(NUM_TILES * 2)  # one for each robber, and acount for Minimap
def map_tile_features(catan_map: CatanMap, robber_coordinate):
    # Returns list of functions that take a game and output a feature.
    # build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
    features = {}

    for tile_id, tile in catan_map.tiles_by_id.items():
        for resource in RESOURCES:
            features[f"TILE{tile_id}_IS_{resource}"] = tile.resource == resource
        features[f"TILE{tile_id}_IS_DESERT"] = tile.resource == None
        features[f"TILE{tile_id}_PROBA"] = (
            0 if tile.resource is None else number_probability(tile.number)
        )
        features[f"TILE{tile_id}_HAS_ROBBER"] = (
            catan_map.tiles[robber_coordinate] == tile
        )
    return features


def tile_features(game: Game, p0_color: Color):
    # Returns list of functions that take a game and output a feature.
    # build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
    return map_tile_features(game.state.board.map, game.state.board.robber_coordinate)


@functools.lru_cache(1)
def map_port_features(catan_map):
    features = {}
    for port_id, port in catan_map.ports_by_id.items():
        for resource in RESOURCES:
            features[f"PORT{port_id}_IS_{resource}"] = port.resource == resource
        features[f"PORT{port_id}_IS_THREE_TO_ONE"] = port.resource is None
    return features


def port_features(game, p0_color):
    # PORT0_WOOD, PORT0_THREE_TO_ONE, ...
    return map_port_features(game.state.board.map)


@functools.lru_cache(4)
def initialize_graph_features_template(num_players, catan_map: CatanMap):
    features = {}
    for i in range(num_players):
        for node_id in range(len(catan_map.land_nodes)):
            for building in [SETTLEMENT, CITY]:
                features[f"NODE{node_id}_P{i}_{building}"] = False
        for edge in get_edges(catan_map.land_nodes):
            features[f"EDGE{edge}_P{i}_ROAD"] = False
    return features


@functools.lru_cache(1024 * 2 * 2 * 2)
def get_node_hot_encoded(player_index, colors, settlements, cities, roads):
    features = {}

    for node_id in settlements:
        features[f"NODE{node_id}_P{player_index}_SETTLEMENT"] = True
    for node_id in cities:
        features[f"NODE{node_id}_P{player_index}_CITY"] = True
    for edge in roads:
        features[f"EDGE{tuple(sorted(edge))}_P{player_index}_ROAD"] = True

    return features


def graph_features(game: Game, p0_color: Color):
    features = initialize_graph_features_template(
        len(game.state.colors), game.state.board.map
    ).copy()

    for i, color in iter_players(game.state.colors, p0_color):
        settlements = tuple(game.state.buildings_by_color[color][SETTLEMENT])
        cities = tuple(game.state.buildings_by_color[color][CITY])
        roads = tuple(game.state.buildings_by_color[color][ROAD])
        to_update = get_node_hot_encoded(
            i, game.state.colors, settlements, cities, roads
        )
        features.update(to_update)

    return features


def build_production_features(consider_robber):
    prefix = "EFFECTIVE_" if consider_robber else "TOTAL_"

    def production_features(game: Game, p0_color: Color):
        # P0_WHEAT_PRODUCTION, P0_ORE_PRODUCTION, ..., P1_WHEAT_PRODUCTION, ...
        features = {}
        board = game.state.board
        robbed_nodes = set(board.map.tiles[board.robber_coordinate].nodes.values())
        for resource in RESOURCES:
            for i, color in iter_players(game.state.colors, p0_color):
                production = 0
                for node_id in get_player_buildings(game.state, color, SETTLEMENT):
                    if consider_robber and node_id in robbed_nodes:
                        continue
                    production += get_node_production(
                        game.state.board.map, node_id, resource
                    )
                for node_id in get_player_buildings(game.state, color, CITY):
                    if consider_robber and node_id in robbed_nodes:
                        continue
                    production += 2 * get_node_production(
                        game.state.board.map, node_id, resource
                    )
                features[f"{prefix}P{i}_{resource}_PRODUCTION"] = production

        return features

    return production_features


@functools.lru_cache(maxsize=1000)
def get_node_production(catan_map, node_id, resource):
    tiles = catan_map.adjacent_tiles[node_id]
    return sum([number_probability(t.number) for t in tiles if t.resource == resource])


def get_player_expandable_nodes(game: Game, color: Color):
    node_sets = game.state.board.find_connected_components(color)
    enemy_colors = [
        enemy_color for enemy_color in game.state.colors if enemy_color != color
    ]
    enemy_node_ids = set()
    for enemy_color in enemy_colors:
        enemy_node_ids.update(get_player_buildings(game.state, enemy_color, SETTLEMENT))
        enemy_node_ids.update(get_player_buildings(game.state, enemy_color, CITY))

    expandable_node_ids = [
        node_id
        for node_set in node_sets
        for node_id in node_set
        if node_id not in enemy_node_ids  # not plowed
    ]  # not exactly "buildable_node_ids" b.c. we could expand from non-buildable nodes
    return expandable_node_ids


REACHABLE_FEATURES_MAX = 2  # inclusive


def get_zero_nodes(game, color):
    zero_nodes = set()
    for component in game.state.board.connected_components[color]:
        for node_id in component:
            zero_nodes.add(node_id)
    return zero_nodes


@functools.lru_cache(maxsize=2000)
def iter_level_nodes(enemy_nodes, enemy_roads, num_roads, zero_nodes):
    """Iterates over possible expansion paths.

    Args:
        enemy_nodes (frozenset[NodeId]): node_ids owned by enemy colors
        enemy_roads (frozenset[EdgeId]): edge_ids owned by enemy colors
        num_roads (int): Max-depth of BFS (inclusive). e.g. 2 will yield
            possible expansions with up to 2 roads.
        zero_nodes (frozenset[NodeId]): Nodes reachable per board.connected_components

    Yields:
        Tuple[int, Set[NodeId], Dict[NodeId, List[EdgeId]]:
            First element is level (roads needed to get there).
            Second element is set of node_ids reachable at this level.
            Third is mapping of NodeId to the list of edges
            that leads to shortest path to that NodeId.
    """
    last_layer_nodes = zero_nodes
    paths = {i: [] for i in zero_nodes}
    results = []
    for level in range(1, num_roads + 1):
        level_nodes = set(last_layer_nodes)
        for node_id in last_layer_nodes:
            if node_id in enemy_nodes:
                continue  # not expandable.

            # here we can assume node is empty or owned
            expandable = []
            for neighbor_id in STATIC_GRAPH.neighbors(node_id):
                edge = (node_id, neighbor_id)
                can_follow_edge = edge not in enemy_roads
                if can_follow_edge:
                    expandable.append(neighbor_id)
                    if neighbor_id not in paths:
                        paths[neighbor_id] = paths[node_id] + [(node_id, neighbor_id)]

            level_nodes.update(expandable)

        results.append((level, level_nodes, paths))

        last_layer_nodes = level_nodes

    return results


def get_owned_or_buildable(game, color, board_buildable):
    return frozenset(
        get_player_buildings(game.state, color, SETTLEMENT)
        + get_player_buildings(game.state, color, CITY)
        + board_buildable
    )


def reachability_features(game: Game, p0_color: Color, levels=REACHABLE_FEATURES_MAX):
    features = {}

    board_buildable = game.state.board.buildable_node_ids(p0_color, True)
    for i, color in iter_players(game.state.colors, p0_color):
        owned_or_buildable = get_owned_or_buildable(game, color, board_buildable)

        # do layer 0
        zero_nodes = get_zero_nodes(game, color)
        production = count_production(
            frozenset(owned_or_buildable.intersection(zero_nodes)),
            game.state.board.map,
        )
        for resource in RESOURCES:
            features[f"P{i}_0_ROAD_REACHABLE_{resource}"] = production[resource]

        # do rest of layers
        enemy_nodes = frozenset(
            k
            for k, v in game.state.board.buildings.items()
            if v is not None and v[0] != color
        )
        enemy_roads = frozenset(
            k for k, v in game.state.board.roads.items() if v is not None and v != color
        )
        for level, level_nodes, paths in iter_level_nodes(
            enemy_nodes, enemy_roads, levels, frozenset(zero_nodes)
        ):
            production = count_production(
                frozenset(owned_or_buildable.intersection(level_nodes)),
                game.state.board.map,
            )
            for resource in RESOURCES:
                features[f"P{i}_{level}_ROAD_REACHABLE_{resource}"] = production[
                    resource
                ]

    return features


@functools.lru_cache(maxsize=1000)
def count_production(nodes, catan_map):
    production = Counter()
    for node_id in nodes:
        production += catan_map.node_production[node_id]
    return production


def expansion_features(game: Game, p0_color: Color):
    MAX_EXPANSION_DISTANCE = 3  # exclusive

    features = {}

    # For each connected component node, bfs_edges (skipping enemy edges and nodes nodes)
    empty_edges = set(get_edges(game.state.board.map.land_nodes))
    for i, color in iter_players(game.state.colors, p0_color):
        empty_edges.difference_update(get_player_buildings(game.state, color, ROAD))
    searchable_subgraph = STATIC_GRAPH.edge_subgraph(empty_edges)

    board_buildable_node_ids = game.state.board.buildable_node_ids(
        p0_color, True
    )  # this should be the same for all players. TODO: Can maintain internally (instead of re-compute).

    for i, color in iter_players(game.state.colors, p0_color):
        expandable_node_ids = get_player_expandable_nodes(game, color)

        def skip_blocked_by_enemy(neighbor_ids):
            for node_id in neighbor_ids:
                node_color = game.state.board.get_node_color(node_id)
                if node_color is None or node_color == color:
                    yield node_id  # not owned by enemy, can explore

        # owned_edges = get_player_buildings(state, color, ROAD)
        dis_res_prod = {
            distance: {k: 0 for k in RESOURCES}
            for distance in range(MAX_EXPANSION_DISTANCE)
        }
        for node_id in expandable_node_ids:
            if node_id in board_buildable_node_ids:  # node itself is buildable
                for resource in RESOURCES:
                    production = get_node_production(
                        game.state.board.map, node_id, resource
                    )
                    dis_res_prod[0][resource] = max(
                        production, dis_res_prod[0][resource]
                    )

            if node_id not in searchable_subgraph.nodes():
                continue  # must be internal node, no need to explore

            bfs_iteration = nx.bfs_edges(
                searchable_subgraph,
                node_id,
                depth_limit=MAX_EXPANSION_DISTANCE - 1,
                sort_neighbors=skip_blocked_by_enemy,
            )

            paths = {node_id: []}
            for edge in bfs_iteration:
                a, b = edge
                path_until_now = paths[a]
                distance = len(path_until_now) + 1
                paths[b] = paths[a] + [b]

                if b not in board_buildable_node_ids:
                    continue

                # means we can get to node b, at distance=d, starting from path[0]
                for resource in RESOURCES:
                    production = get_node_production(game.state.board.map, b, resource)
                    dis_res_prod[distance][resource] = max(
                        production, dis_res_prod[distance][resource]
                    )

        for distance, res_prod in dis_res_prod.items():
            for resource, prod in res_prod.items():
                features[f"P{i}_{resource}_AT_DISTANCE_{int(distance)}"] = prod

    return features


def port_distance_features(game: Game, p0_color: Color):
    # P0_HAS_WHEAT_PORT, P0_WHEAT_PORT_DISTANCE, ..., P1_HAS_WHEAT_PORT,
    features = {}
    ports = game.state.board.map.port_nodes
    distances = get_node_distances()
    resources_and_none: List[Any] = RESOURCES.copy()
    resources_and_none += [None]
    for resource_or_none in resources_and_none:
        port_name = resource_or_none or "3:1"
        for i, color in iter_players(game.state.colors, p0_color):
            expandable_node_ids = get_player_expandable_nodes(game, color)
            if len(expandable_node_ids) == 0:
                features[f"P{i}_HAS_{port_name}_PORT"] = False
                features[f"P{i}_{port_name}_PORT_DISTANCE"] = float("inf")
            else:
                min_distance = min(
                    [
                        distances[port_node_id][my_node]
                        for my_node in expandable_node_ids
                        for port_node_id in ports[resource_or_none]
                    ]
                )
                features[f"P{i}_HAS_{port_name}_PORT"] = min_distance == 0
                features[f"P{i}_{port_name}_PORT_DISTANCE"] = min_distance
    return features


def game_features(game: Game, p0_color: Color):
    # BANK_WOODS, BANK_WHEATS, ..., BANK_DEV_CARDS
    possibilities = set([a.action_type for a in game.state.playable_actions])
    features = {
        "BANK_DEV_CARDS": len(game.state.development_listdeck),
        "IS_MOVING_ROBBER": ActionType.MOVE_ROBBER in possibilities,
        "IS_DISCARDING": ActionType.DISCARD in possibilities,
    }
    for resource in RESOURCES:
        features[f"BANK_{resource}"] = freqdeck_count(
            game.state.resource_freqdeck, resource
        )
    return features


feature_extractors = [
    # PLAYER FEATURES =====
    player_features,
    resource_hand_features,
    tracked_features,
    available_actions_features,
    # TRANSFERABLE BOARD FEATURES =====
    # build_production_features(True),
    # build_production_features(False),
    # expansion_features,
    # reachability_features,
    # RAW BASE-MAP FEATURES =====
    tile_features,
    port_features,
    graph_features,
    # GAME FEATURES =====
    game_features,
]


# TODO: Use OrderedDict instead? To minimize mis-aligned features errors.
def create_sample(game, p0_color):
    record = {}
    for extractor in feature_extractors:
        record.update(extractor(game, p0_color))
    return record


def create_sample_vector(game, p0_color, features=None):
    features = features or get_feature_ordering(len(game.state.colors))
    sample_dict = create_sample(game, p0_color)
    return [float(sample_dict[i]) for i in features if i in sample_dict]


@functools.lru_cache(4 * 3)
def get_feature_ordering(
    num_players=4, map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE"
):
    players = [
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.RED),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    players = players[:num_players]
    game = Game(
        players,
        catan_map=build_map(map_type),
        trackers=[CardCounting(players=players, color=players[0].color)],
    )
    sample = create_sample(game=game, p0_color=players[0].color)
    return sorted(sample.keys())
