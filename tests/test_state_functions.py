import pytest

from catanatron.game import Game
from catanatron.models.board import Board
from catanatron.models.map import build_map
from catanatron.state import State, apply_action
from catanatron.state_functions import (
    buy_dev_card,
    get_actual_victory_points,
    get_largest_army,
    play_dev_card,
    player_deck_random_draw,
    player_deck_replenish,
    calculate_resource_probabilities,
    get_dice_roll_odds,
)
from catanatron.models.enums import (
    BRICK,
    KNIGHT,
    ORE,
    SHEEP,
    WHEAT,
    WOOD,
    Action,
    ActionType,
    FastBuildingType,
    FastResource,
)
from catanatron.models.player import Color, SimplePlayer


def test_cant_steal_devcards():
    # Arrange: Have RED buy 1 dev card (and have no resource cards)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    player_deck_replenish(state, Color.RED, WHEAT)
    player_deck_replenish(state, Color.RED, ORE)
    player_deck_replenish(state, Color.RED, SHEEP)
    buy_dev_card(state, Color.RED, KNIGHT)

    # Act: Attempt to steal a resource
    with pytest.raises(IndexError):  # no resource cards in hand
        player_deck_random_draw(state, Color.RED)


def test_defeating_your_own_largest_army_doesnt_give_more_vps():
    # Arrange: Buy all dev cards
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    player_deck_replenish(state, players[0].color, SHEEP, 26)
    player_deck_replenish(state, players[0].color, WHEAT, 26)
    player_deck_replenish(state, players[0].color, ORE, 26)
    for i in range(25):
        apply_action(
            state, Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None)
        )
    assert get_largest_army(state) == (None, None)
    assert get_actual_victory_points(state, Color.RED) == 5

    # Act - Assert
    play_dev_card(state, Color.RED, KNIGHT)
    play_dev_card(state, Color.RED, KNIGHT)
    play_dev_card(state, Color.RED, KNIGHT)
    assert get_largest_army(state) == (Color.RED, 3)
    assert get_actual_victory_points(state, Color.RED) == 7

    # Act - Assert
    play_dev_card(state, Color.RED, KNIGHT)
    assert get_largest_army(state) == (Color.RED, 4)
    assert get_actual_victory_points(state, Color.RED) == 7



def test_calculating_player_resource_probabilities():
    # map setup
    players = [SimplePlayer(color) for color in [Color.RED, Color.BLUE, Color.WHITE]]
    game = Game(players=players)
    
    # Amend the map for the test's custom values
    for tile in game.state.board.map.land_tiles.values():
        if tile.id == 1:
            tile.resource = WHEAT
            tile.number = 6
        elif tile.id == 6:
            tile.resource = SHEEP
            tile.number = 3
        elif tile.id == 7:
            tile.resource = SHEEP
            tile.number = 3
        elif tile.id == 8:
            tile.resource = SHEEP
            tile.number = 11
        elif tile.id == 15:
            tile.resource = SHEEP
            tile.number = 5
        elif tile.id == 16:
            tile.resource = ORE
            tile.number = 9
        elif tile.id == 18:
            tile.resource = ORE
            tile.number = 2
        

    p0_color = game.state.colors[0]
    p1_color = game.state.colors[1]
    p2_color = game.state.colors[2]

    # Apply actions to simulate game setup
    game.execute(Action(p0_color, ActionType.BUILD_SETTLEMENT, 45))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (45, 47)))

    game.execute(Action(p1_color, ActionType.BUILD_SETTLEMENT, 8))
    game.execute(Action(p1_color, ActionType.BUILD_ROAD, (7, 8)))

    game.execute(Action(p2_color, ActionType.BUILD_SETTLEMENT, 24))
    game.execute(Action(p2_color, ActionType.BUILD_ROAD, (24, 25)))

    game.execute(Action(p2_color, ActionType.BUILD_SETTLEMENT, 26))
    game.execute(Action(p2_color, ActionType.BUILD_ROAD, (26, 27)))

    game.execute(Action(p1_color, ActionType.BUILD_SETTLEMENT, 6))
    game.execute(Action(p1_color, ActionType.BUILD_ROAD, (6, 7)))

    game.execute(Action(p0_color, ActionType.BUILD_SETTLEMENT, 48))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (48, 49)))

    action = Action(players[0].color, ActionType.MOVE_ROBBER, ((0, 2, -2), None, None))
    apply_action(game.state, action)

    probabilities = calculate_resource_probabilities(game.state)
    
    expected_probabilities = {
        p0_color: {WOOD: 0, BRICK: 0, SHEEP: 0, WHEAT: 0, ORE: 4/36},
        p1_color: {WOOD: 0, BRICK: 0, SHEEP: 6/36, WHEAT: 10/36, ORE: 1/36},
        p2_color: {WOOD: 0, BRICK: 0, SHEEP: 4/36, WHEAT: 0, ORE: 1/36},
    }

    assert probabilities == expected_probabilities, f"Expected {expected_probabilities},\n got {probabilities}"