from catanatron.game import Game
from catanatron.models.decks import freqdeck_from_listdeck
from catanatron.models.enums import BRICK, ORE, SHEEP, UNKNOWN, WHEAT, WOOD, Action, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.players.tracker import CardCounting
from catanatron.state import State, apply_action
from catanatron.state_functions import player_freqdeck_add, player_freqdeck_subtract, player_num_resource_cards


def test_buying_road_transaction_is_tracked_using_assumed():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    CardCounting_Blue = CardCounting(game, Color.BLUE)
    # 1 wood, 2 brick, 1 unknown
    CardCounting_Blue.assumed_resources[Color.RED]['WOOD'] = 1
    CardCounting_Blue.assumed_resources[Color.RED]['BRICK'] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['UNKNOWN'] = 1
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [SHEEP]
    
    action = Action(Color.RED, ActionType.BUILD_ROAD, (3, 4))
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [SHEEP]


def test_buying_road_transaction_is_tracked_using_assumed_against_state():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    CardCounting_Blue = CardCounting(game=None, color=Color.BLUE, state=state)

    state.is_initial_build_phase = False
    state.board.build_settlement(players[0].color, 3, True)
    action = Action(players[0].color, ActionType.BUILD_ROAD, (3, 4))
    # 1 wood, 2 brick, 1 Sheep
    player_freqdeck_add(
        state,
        players[0].color,
        freqdeck_from_listdeck([WOOD, BRICK, BRICK, SHEEP]),
    )
    # 1 wood, 2 brick, 1 Sheep
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 2
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 1
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [SHEEP]
    
    apply_action(state, action)
    CardCounting_Blue.update_opponent_resources(state, action)
    
    assert player_num_resource_cards(state, players[0].color, WOOD) == CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert player_num_resource_cards(state, players[0].color, BRICK) == CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert player_num_resource_cards(state, players[0].color, SHEEP) == CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [SHEEP]
    

def test_buying_road_transaction_is_tracked_using_unknown():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    CardCounting_Blue = CardCounting(game=None, color=Color.BLUE, state=state)

    # 0 wood, 2 brick, 2 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 0
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 2
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [WOOD, ORE]
    
    action = Action(players[0].color, ActionType.BUILD_ROAD, (3, 4))
    CardCounting_Blue.update_opponent_resources(state, action)
    
    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [ORE]


def test_buying_road_transaction_is_tracked_using_unknown_against_state():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    CardCounting_Blue = CardCounting(game=None, color=Color.BLUE, state=state)

    state.is_initial_build_phase = False
    state.board.build_settlement(players[0].color, 3, True)
    action = Action(players[0].color, ActionType.BUILD_ROAD, (3, 4))
    # 1 wood, 2 brick, 1 ore
    player_freqdeck_add(
        state,
        players[0].color,
        freqdeck_from_listdeck([WOOD, BRICK, BRICK, ORE]),
    )
    # 0 wood, 2 brick, 2 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 0
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 2
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [WOOD, ORE]
    
    apply_action(state, action)
    CardCounting_Blue.update_opponent_resources(state, action)
    
    assert player_num_resource_cards(state, players[0].color, WOOD) == CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert player_num_resource_cards(state, players[0].color, BRICK) == CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert player_num_resource_cards(state, players[0].color, ORE) == CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [ORE]

    
def test_buying_settlement_transaction_is_tracked_using_assumed():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    CardCounting_Blue = CardCounting(game, Color.BLUE)
    # mimic initial phase having occurred
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 1)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    # 1 wood, 2 brick, 1 wheat, 1 sheep, 0 ore, 1 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 2
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 1
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 1
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 0
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 1
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [ORE]
    
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 5)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == []


def test_buying_settlement_transaction_is_tracked_using_assumed_against_state():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    p0_color = game.state.colors[0]
    p1_color = game.state.colors[1]
    CardCounting_p0 = CardCounting(game, p0_color)

    # mimic initial phase having occurred
    action = Action(p1_color, ActionType.BUILD_SETTLEMENT, 1)
    CardCounting_p0.update_opponent_resources(game.state, action)
    action = Action(p1_color, ActionType.BUILD_SETTLEMENT, 3)
    CardCounting_p0.update_opponent_resources(game.state, action)

    # as assumed is overwritten, the previous 2nd settlement resources are not tracked
    player_freqdeck_add(game.state, p1_color, [1, 2, 1, 1, 1])
    CardCounting_p0.assumed_resources[p1_color][WOOD] = 1
    CardCounting_p0.assumed_resources[p1_color][BRICK] = 2
    CardCounting_p0.assumed_resources[p1_color][SHEEP] = 1
    CardCounting_p0.assumed_resources[p1_color][WHEAT] = 1
    CardCounting_p0.assumed_resources[p1_color][ORE] = 0
    CardCounting_p0.assumed_resources[p1_color][UNKNOWN] = 1
    CardCounting_p0.assumed_resources[p1_color]['unknown_list'] = [ORE]

    # mimics reducing count of resources, just manually; removing the settlement cost anyway
    player_freqdeck_subtract(game.state, p1_color, [1, 1, 1, 1, 0])

    action = Action(p1_color, ActionType.BUILD_SETTLEMENT, 5)
    CardCounting_p0.update_opponent_resources(game.state, action)

    assert CardCounting_p0.assumed_resources[p1_color][WOOD] == player_num_resource_cards(game.state, p1_color, WOOD) == 0
    assert CardCounting_p0.assumed_resources[p1_color][BRICK] == player_num_resource_cards(game.state, p1_color, BRICK) == 1
    assert CardCounting_p0.assumed_resources[p1_color][SHEEP] == player_num_resource_cards(game.state, p1_color, SHEEP) == 0
    assert CardCounting_p0.assumed_resources[p1_color][WHEAT] == player_num_resource_cards(game.state, p1_color, WHEAT) == 0
    assert CardCounting_p0.assumed_resources[p1_color][ORE] == 0
    assert CardCounting_p0.assumed_resources[p1_color][UNKNOWN] == player_num_resource_cards(game.state, p1_color, ORE) == 1
    assert CardCounting_p0.assumed_resources[p1_color]['unknown_list'] == [ORE]


def test_buying_settlement_transaction_is_tracked_using_unknown():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    CardCounting_Blue = CardCounting(game, Color.BLUE)

    # mimic initial phase having occurred
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 1)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    
    # 1 wood, 2 brick, 0 wheat, 0 sheep, 1 ore, 2 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 2
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 0
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 0
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 1
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [SHEEP, WHEAT]

    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 5)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == []


def test_buying_settlement_transaction_is_tracked_using_unknown_against_state():
    # this test is an actual mock of the game and works
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.WHITE)]
    game = Game(players)
    p2_color = game.state.colors[0]
    p1_color = game.state.colors[1]
    p0_color = game.state.colors[2]
    # board setup
    for tile in game.state.board.map.land_tiles.values():
        if tile.id == 0:
            tile.resource = WOOD
            tile.number = 8
        elif tile.id == 1:
            tile.resource = SHEEP
            tile.number = 5
        elif tile.id == 6:
            tile.resource = BRICK
            tile.number = 8
        elif tile.id == 8:
            tile.resource = WHEAT
            tile.number = 4

    CardCounting_Blue = CardCounting(game, p0_color)

    game.state.is_initial_build_phase = True
    build_action = Action(p2_color, ActionType.BUILD_SETTLEMENT, 9)
    apply_action(game.state, action=build_action)
    CardCounting_Blue.update_opponent_resources(game.state, build_action)
    build_action = Action(p2_color, ActionType.BUILD_ROAD, (9, 10))
    apply_action(game.state, action=build_action)
    build_action = Action(p1_color, ActionType.BUILD_SETTLEMENT, 26)
    CardCounting_Blue.update_opponent_resources(game.state, build_action)
    apply_action(game.state, action=build_action)
    build_action = Action(p1_color, ActionType.BUILD_ROAD, (26, 27))
    apply_action(game.state, action=build_action)
    build_action = Action(p0_color, ActionType.BUILD_SETTLEMENT, 33)
    CardCounting_Blue.update_opponent_resources(game.state, build_action)
    apply_action(game.state, action=build_action)
    build_action = Action(p0_color, ActionType.BUILD_ROAD, (33, 34))
    apply_action(game.state, action=build_action)
    build_action = Action(p0_color, ActionType.BUILD_SETTLEMENT, 31)
    CardCounting_Blue.update_opponent_resources(game.state, build_action)
    apply_action(game.state, action=build_action)
    build_action = Action(p0_color, ActionType.BUILD_ROAD, (31, 32))
    apply_action(game.state, action=build_action)
    build_action = Action(p1_color, ActionType.BUILD_SETTLEMENT, 28)
    CardCounting_Blue.update_opponent_resources(game.state, build_action)
    apply_action(game.state, action=build_action)
    build_action = Action(p1_color, ActionType.BUILD_ROAD, (27, 28))
    apply_action(game.state, action=build_action)
    build_action = Action(p2_color, ActionType.BUILD_SETTLEMENT, 1)
    CardCounting_Blue.update_opponent_resources(game.state, build_action)
    apply_action(game.state, action=build_action)
    build_action = Action(p2_color, ActionType.BUILD_ROAD, (0, 1))
    apply_action(game.state, action=build_action)
    # red = sheep, brick, wood
    # White = wheat

    assert player_num_resource_cards(game.state, p2_color, WOOD) == 1
    assert player_num_resource_cards(game.state, p2_color, BRICK) == 1
    assert player_num_resource_cards(game.state, p2_color, SHEEP) == 1
    assert player_num_resource_cards(game.state, p1_color, WHEAT) == 1
    # cheat and add road resource
    player_freqdeck_add(game.state, p2_color, [1, 1, 0, 0, 0])

    # 1 wood, 1 brick, 1 sheep, 1 wheat, 0 ore, 0 unknown
    CardCounting_Blue.assumed_resources[p2_color][WOOD] = 1
    CardCounting_Blue.assumed_resources[p2_color][BRICK] = 1
    CardCounting_Blue.assumed_resources[p2_color][SHEEP] = 1
    CardCounting_Blue.assumed_resources[p2_color][WHEAT] = 0
    CardCounting_Blue.assumed_resources[p2_color][ORE] = 0
    CardCounting_Blue.assumed_resources[p2_color][UNKNOWN] = 0
    CardCounting_Blue.assumed_resources[p2_color]['unknown_list'] = []
    CardCounting_Blue.assumed_resources[p1_color][WHEAT] = 1

    action_roll = Action(p2_color, ActionType.ROLL, (6, 1))
    apply_action(game.state, action=action_roll)
    CardCounting_Blue.update_opponent_resources(game.state, action_roll)

    action_rob = Action(p2_color, ActionType.MOVE_ROBBER, ((2, -2, 0), p1_color, WHEAT))
    apply_action(game.state, action=action_rob)
    CardCounting_Blue.update_opponent_resources(game.state, action_rob)
    assert player_num_resource_cards(game.state, p2_color, WHEAT) == CardCounting_Blue.assumed_resources[p2_color][UNKNOWN] == 1
    assert player_num_resource_cards(game.state, p1_color, WHEAT) == 0
    assert CardCounting_Blue.assumed_resources[p2_color]['unknown_list'] == [WHEAT]

    action = Action(p2_color, ActionType.BUILD_ROAD, (0, 5))
    apply_action(game.state, action=action)
    action = Action(p2_color, ActionType.BUILD_SETTLEMENT, 5)
    apply_action(game.state, action=action)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert player_num_resource_cards(game.state, p2_color, WOOD) == CardCounting_Blue.assumed_resources[p2_color][WOOD] == 0
    assert player_num_resource_cards(game.state, p2_color, BRICK) == CardCounting_Blue.assumed_resources[p2_color][BRICK] == 0
    assert player_num_resource_cards(game.state, p2_color, SHEEP) == CardCounting_Blue.assumed_resources[p2_color][SHEEP] == 0
    assert player_num_resource_cards(game.state, p2_color, WHEAT) == CardCounting_Blue.assumed_resources[p2_color][WHEAT] == 0
    assert player_num_resource_cards(game.state, p2_color, ORE) == CardCounting_Blue.assumed_resources[p2_color][ORE] == 0
    assert CardCounting_Blue.assumed_resources[p2_color][UNKNOWN] == 0
    assert CardCounting_Blue.assumed_resources[p2_color]['unknown_list'] == []


def test_buying_city_transaction_is_tracked_using_assumed_against_state():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    CardCounting_Blue = CardCounting(game, Color.BLUE)

    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 1)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    
    # 1 wood, 1 brick, 2 wheat, 1 sheep, 3 ore
    player_freqdeck_add(game.state, Color.RED, [1, 1, 1, 2, 3])
    # 0 wood, 1 brick, 2 wheat, 0 sheep, 3 ore, 2 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 0
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 1
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 0
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 2
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 3
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [WOOD, SHEEP]

    action = Action(Color.RED, ActionType.BUILD_CITY, 3)
    player_freqdeck_subtract(game.state, Color.RED, [0, 0, 0, 2, 3])
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert player_num_resource_cards(game.state, Color.RED, WOOD) + player_num_resource_cards(game.state, Color.RED, SHEEP) == CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 2
    assert player_num_resource_cards(game.state, Color.RED, BRICK) == CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert player_num_resource_cards(game.state, Color.RED, WHEAT) == CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert player_num_resource_cards(game.state, Color.RED, ORE) == CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [WOOD, SHEEP]


def test_buying_city_transaction_is_tracked_using_assumed():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.state.board.build_settlement(Color.RED, 3, True)
    
    CardCounting_Blue = CardCounting(game, Color.BLUE)
    # 0 wood, 1 brick, 2 wheat, 0 sheep, 3 ore, 2 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 0
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 1
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 0
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 2
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 3
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [WOOD, SHEEP]

    
    action = Action(Color.RED, ActionType.BUILD_CITY, 3)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 2
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [WOOD, SHEEP]


def test_buying_city_transaction_is_tracked_using_unknown_against_state():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    CardCounting_Blue = CardCounting(game, Color.BLUE)

    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 1)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    
    # 1 wood, 1 brick, 2 wheat, 1 sheep, 3 ore
    player_freqdeck_add(game.state, Color.RED, [1, 1, 1, 2, 3])
    # 1 wood, 1 brick, 0 wheat, 0 sheep, 1 ore, 5 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 1
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 0
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 0
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 1
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 5
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [WHEAT, WHEAT, ORE, SHEEP, ORE]

    action = Action(Color.RED, ActionType.BUILD_CITY, 3)
    player_freqdeck_subtract(game.state, Color.RED, [0, 0, 0, 2, 3])
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert player_num_resource_cards(game.state, Color.RED, WOOD) == CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 1
    assert player_num_resource_cards(game.state, Color.RED, BRICK) == CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert player_num_resource_cards(game.state, Color.RED, SHEEP) == CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert player_num_resource_cards(game.state, Color.RED, WHEAT) == CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert player_num_resource_cards(game.state, Color.RED, ORE) == CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [SHEEP]


def test_buying_city_transaction_is_tracked_using_unknown():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.state.board.build_settlement(Color.RED, 3, True)
    
    CardCounting_Blue = CardCounting(game, Color.BLUE)
    # 1 wood, 1 brick, 0 wheat, 0 sheep, 1 ore, 5 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 1
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 0
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 0
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 1
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 5
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [WHEAT, WHEAT, ORE, SHEEP, ORE]

    action = Action(Color.RED, ActionType.BUILD_CITY, 3)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [SHEEP]


def test_buying_dev_card_transaction_is_tracked_using_assumed():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    CardCounting_Blue = CardCounting(game, Color.BLUE)

    # 1 wood, 1 sheep, 2 wheat, 1 ore, 2 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 1
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 2
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 1
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [SHEEP, WOOD]
    
    action = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 2
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [SHEEP, WOOD]



def test_buying_dev_card_transaction_is_tracked_using_assumed_against_state():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    CardCounting_Blue = CardCounting(game, Color.BLUE)

    player_freqdeck_add(game.state, Color.RED, [2, 0, 2, 2, 1])

    # 1 wood, 1 sheep, 2 wheat, 1 ore, 2 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 1
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 2
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 1
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [SHEEP, WOOD]
    
    action = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    player_freqdeck_subtract(game.state, Color.RED, [0, 0, 1, 1, 1])

    assert player_num_resource_cards(game.state, Color.RED, WOOD) == 2
    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 1
    assert player_num_resource_cards(game.state, Color.RED, SHEEP) == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert player_num_resource_cards(game.state, Color.RED, BRICK) == CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 0
    assert player_num_resource_cards(game.state, Color.RED, WHEAT) == CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 1
    assert player_num_resource_cards(game.state, Color.RED, ORE) == CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 2
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [SHEEP, WOOD]


def test_buying_dev_card_transaction_is_tracked_using_unknown():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    CardCounting_Blue = CardCounting(game, Color.BLUE)

    # 1 wood, 0 sheep, 2 wheat, 0 ore, 3 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 0
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 2
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 0
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 3
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [SHEEP, WOOD, ORE]
    
    action = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [WOOD]


def test_buying_dev_card_transaction_is_tracked_using_unknown_against_state():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    CardCounting_Blue = CardCounting(game, Color.BLUE)

    player_freqdeck_add(game.state, Color.RED, [2, 0, 1, 2, 1])

    # 1 wood, 0 sheep, 2 wheat, 0 ore, 3 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 0
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 2
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 0
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 3
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [SHEEP, WOOD, ORE]
    
    action = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)
    CardCounting_Blue.update_opponent_resources(game.state, action)
    player_freqdeck_subtract(game.state, Color.RED, [0, 0, 1, 1, 1])

    assert player_num_resource_cards(game.state, Color.RED, WOOD) == 2
    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 1
    assert player_num_resource_cards(game.state, Color.RED, SHEEP) == CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert player_num_resource_cards(game.state, Color.RED, BRICK) == CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 0
    assert player_num_resource_cards(game.state, Color.RED, WHEAT) == CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 1
    assert player_num_resource_cards(game.state, Color.RED, ORE) == CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [WOOD]





# test_buying_road_transaction_is_tracked_using_unknown_against_state()
# test_buying_road_transaction_is_tracked_using_assumed_against_state()
# test_buying_road_transaction_is_tracked_using_assumed()
# test_buying_road_transaction_is_tracked_using_unknown()

# test_buying_settlement_transaction_is_tracked_using_assumed_against_state()
# test_buying_settlement_transaction_is_tracked_using_assumed()
# test_buying_settlement_transaction_is_tracked_using_unknown_against_state()
# test_buying_settlement_transaction_is_tracked_using_unknown()

# # test_buying_city_transaction_is_tracked_using_unknown_against_state()
# test_buying_city_transaction_is_tracked_using_unknown()
# # test_buying_city_transaction_is_tracked_using_assumed_against_state()
# test_buying_city_transaction_is_tracked_using_assumed()

# Remains of a test that mocks an actual game. For some reason, the board won't be what it's set to. Overkill test?
# def test_buying_settlement_transaction_is_tracked_using_assumed_against_state():
#     players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.WHITE)]
#     game = Game(players)
#     p0_color = game.state.colors[0]
#     p1_color = game.state.colors[1]
#     p2_color = game.state.colors[2]
#     # board setup
#     for tile in game.state.board.map.land_tiles.values():
#         if tile.id == 0:
#             tile.resource = WOOD
#             tile.number = 8
#         elif tile.id == 1:
#             tile.resource = WHEAT
#             tile.number = 5
#         elif tile.id == 2:
#             tile.resource = BRICK
#             tile.number = 6    
#         elif tile.id == 6:
#             tile.resource = BRICK
#             tile.number = 8
#         elif tile.id == 7:
#             tile.resource = ORE
#             tile.number = 3
#         elif tile.id == 8:
#             tile.resource = SHEEP
#             tile.number = 6

#     CardCounting_Blue = CardCounting(game, p0_color)
#     # mimic initial phase having occurred

#     game.state.is_initial_build_phase = True
#     build_action = Action(p0_color, ActionType.BUILD_SETTLEMENT, 33)
#     CardCounting_Blue.update_opponent_resources(game.state, build_action)
#     apply_action(game.state, action=build_action)
#     build_action = Action(p0_color, ActionType.BUILD_ROAD, (33, 34))
#     apply_action(game.state, action=build_action)
#     build_action = Action(p1_color, ActionType.BUILD_SETTLEMENT, 28)
#     CardCounting_Blue.update_opponent_resources(game.state, build_action)
#     apply_action(game.state, action=build_action)
#     build_action = Action(p1_color, ActionType.BUILD_ROAD, (27, 28))
#     apply_action(game.state, action=build_action)
#     build_action = Action(p2_color, ActionType.BUILD_SETTLEMENT, 9)
#     apply_action(game.state, action=build_action)
#     CardCounting_Blue.update_opponent_resources(game.state, build_action)
#     build_action = Action(p2_color, ActionType.BUILD_ROAD, (9, 10))
#     apply_action(game.state, action=build_action)
#     build_action = Action(p2_color, ActionType.BUILD_SETTLEMENT, 1)
#     CardCounting_Blue.update_opponent_resources(game.state, build_action)
#     apply_action(game.state, action=build_action)
#     build_action = Action(p2_color, ActionType.BUILD_ROAD, (1, 2))
#     apply_action(game.state, action=build_action)
#     build_action = Action(p1_color, ActionType.BUILD_SETTLEMENT, 26)
#     CardCounting_Blue.update_opponent_resources(game.state, build_action)
#     apply_action(game.state, action=build_action)
#     build_action = Action(p1_color, ActionType.BUILD_ROAD, (26, 27))
#     apply_action(game.state, action=build_action)
#     build_action = Action(p0_color, ActionType.BUILD_SETTLEMENT, 31)
#     CardCounting_Blue.update_opponent_resources(game.state, build_action)
#     apply_action(game.state, action=build_action)
#     build_action = Action(p0_color, ActionType.BUILD_ROAD, (31, 32))
#     apply_action(game.state, action=build_action)
#     # red = sheep, brick, wood
#     # White = wheat
    
#     print('1: ', player_deck_to_array(game.state, p2_color))
#     print('1: ', CardCounting_Blue.assumed_resources[p2_color])
#     assert player_num_resource_cards(game.state, p2_color, WOOD) == 1
#     assert player_num_resource_cards(game.state, p2_color, WHEAT) == 1
#     assert player_num_resource_cards(game.state, p2_color, BRICK) == 1
#     assert player_num_resource_cards(game.state, p2_color, ORE) == 0
#     assert player_num_resource_cards(game.state, p1_color, ORE) == 1

#     # 1 wood, 1 brick, 1 sheep, 1 wheat, 0 ore, 0 unknown
#     CardCounting_Blue.assumed_resources[p2_color][WOOD] = 1
#     CardCounting_Blue.assumed_resources[p2_color][BRICK] = 1
#     CardCounting_Blue.assumed_resources[p2_color][SHEEP] = 0
#     CardCounting_Blue.assumed_resources[p2_color][WHEAT] = 1
#     CardCounting_Blue.assumed_resources[p2_color][ORE] = 0
#     CardCounting_Blue.assumed_resources[p2_color][UNKNOWN] = 0
#     CardCounting_Blue.assumed_resources[p2_color]['unknown_list'] = []
#     CardCounting_Blue.assumed_resources[p1_color][ORE] = 1

#     # roll to generate resource counts for test's 'build actor (p2)'
#     action_roll = Action(p0_color, ActionType.ROLL, (5, 1))
#     apply_action(game.state, action=action_roll)
#     CardCounting_Blue.update_opponent_resources(game.state, action_roll)
#     print('2: ', player_deck_to_array(game.state, p2_color))
#     print('2: ', CardCounting_Blue.assumed_resources[p2_color])
#     action_end = Action(p0_color, ActionType.END_TURN, None)
#     apply_action(game.state, action=action_end)
#     CardCounting_Blue.update_opponent_resources(game.state, action_end)
#     print('2.5: ', player_deck_to_array(game.state, p2_color))
#     print('2.5: ', CardCounting_Blue.assumed_resources[p2_color])
#     assert player_num_resource_cards(game.state, p2_color, SHEEP) == 1
#     assert player_num_resource_cards(game.state, p2_color, BRICK) == 2

#     # roll to generate resource counts for test's 'build actor (p2)'
#     action_roll = Action(p1_color, ActionType.ROLL, (6, 2))
#     apply_action(game.state, action=action_roll)
#     print('3: ', player_deck_to_array(game.state, p2_color))
#     print('3: ', CardCounting_Blue.assumed_resources[p2_color])
#     CardCounting_Blue.update_opponent_resources(game.state, action_roll)
#     action_end = Action(p1_color, ActionType.END_TURN, None)
#     apply_action(game.state, action=action_end)
#     CardCounting_Blue.update_opponent_resources(game.state, action_end)
#     print('3.5: ', player_deck_to_array(game.state, p2_color))
#     print('3.5: ', CardCounting_Blue.assumed_resources[p2_color])
#     assert player_num_resource_cards(game.state, p2_color, BRICK) == 3
#     assert player_num_resource_cards(game.state, p2_color, WOOD) == 2

#     #robbery to generate unknown naturally for verifying unknown isn't used
#     action_roll = Action(p2_color, ActionType.ROLL, (6, 1))
#     apply_action(game.state, action=action_roll)
#     CardCounting_Blue.update_opponent_resources(game.state, action_roll)
#     print('4: ', player_deck_to_array(game.state, p2_color))
#     print('4: ', CardCounting_Blue.assumed_resources[p2_color])
#     action_rob = Action(p2_color, ActionType.MOVE_ROBBER, ((2, -2, 0), p1_color, ORE))
#     apply_action(game.state, action=action_rob)
#     print('5: ', player_deck_to_array(game.state, p2_color))
#     print('5: ', CardCounting_Blue.assumed_resources[p2_color])
#     assert player_num_resource_cards(game.state, p2_color, ORE) == 1
#     assert player_num_resource_cards(game.state, p1_color, ORE) == 0
#     CardCounting_Blue.update_opponent_resources(game.state, action_rob)
#     assert CardCounting_Blue.assumed_resources[p2_color][UNKNOWN] == 1
#     assert CardCounting_Blue.assumed_resources[p2_color]['unknown_list'] == [ORE]
#     # build actor (p2) builds road (necessary) and settlement (for test)
#     print('6: ', player_deck_to_array(game.state, p2_color))
#     print('6: ', CardCounting_Blue.assumed_resources[p2_color])
#     action = Action(p2_color, ActionType.BUILD_ROAD, (2, 3))
#     apply_action(game.state, action=action)
#     print('7: ', player_deck_to_array(game.state, p2_color))
#     print('7: ', CardCounting_Blue.assumed_resources[p2_color])
#     CardCounting_Blue.update_opponent_resources(game.state, action)
#     print('8: ', player_deck_to_array(game.state, p2_color))
#     print('8: ', CardCounting_Blue.assumed_resources[p2_color])
#     assert player_num_resource_cards(game.state, p2_color, WOOD) == 1
#     assert player_num_resource_cards(game.state, p2_color, BRICK) == 2
#     assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 1
#     assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 2
#     action = Action(p2_color, ActionType.BUILD_SETTLEMENT, 3)
#     apply_action(game.state, action=action)
#     CardCounting_Blue.update_opponent_resources(game.state, action)

#     assert player_num_resource_cards(game.state, p2_color, WOOD) == 0
#     assert player_num_resource_cards(game.state, p2_color, BRICK) == 1
#     assert player_num_resource_cards(game.state, p2_color, SHEEP) == 0
#     assert player_num_resource_cards(game.state, p2_color, WHEAT) == 0
#     assert player_num_resource_cards(game.state, p2_color, ORE) == 1

#     assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
#     assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
#     assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
#     assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
#     assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
#     assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
#     assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [ORE]