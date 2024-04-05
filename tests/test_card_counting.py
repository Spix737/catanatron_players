from catanatron.game import Game
from catanatron.models.decks import freqdeck_from_listdeck
from catanatron.models.enums import BRICK, ORE, RESOURCES, SHEEP, UNKNOWN, WHEAT, WOOD, Action, ActionType, FastResource
from catanatron.models.player import Color, SimplePlayer
from catanatron.players.tracker import CardCounting
from catanatron.state import State, apply_action, yield_resources
from catanatron.state_functions import player_deck_replenish, player_freqdeck_add, player_key, player_num_resource_cards


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
    
    assert player_num_resource_cards(state, players[0].color, WOOD) == 0
    assert player_num_resource_cards(state, players[0].color, BRICK) == 1
    assert player_num_resource_cards(state, players[0].color, SHEEP) == 1

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
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
    
    assert player_num_resource_cards(state, players[0].color, WOOD) == 0
    assert player_num_resource_cards(state, players[0].color, BRICK) == 1
    assert player_num_resource_cards(state, players[0].color, ORE) == 1

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [ORE]


    
def test_buying_settlement_transaction_is_tracked_using_assumed():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    CardCounting_Blue = CardCounting(game, Color.BLUE)
    # 1 wood, 2 brick, 1 wheat, 1 sheep, 0 ore, 1 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 2
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 1
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 1
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 0
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 1
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [ORE]
    
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3)
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
    state = State(players)
    state.is_initial_build_phase = False

    # 1 wood, 2 brick, 1 Sheep
    player_freqdeck_add(
        state,
        players[0].color,
        freqdeck_from_listdeck([WOOD, BRICK, BRICK, SHEEP, WHEAT, ORE]),
    )

    CardCounting_Blue = CardCounting(game, Color.BLUE)
    # 1 wood, 2 brick, 1 wheat, 1 sheep, 0 ore, 1 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 2
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 1
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 1
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 0
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 1
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [ORE]
    
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3)
    apply_action(state, action=action)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert player_num_resource_cards(state, players[0].color, WOOD) == 0
    assert player_num_resource_cards(state, players[0].color, BRICK) == 1
    assert player_num_resource_cards(state, players[0].color, SHEEP) == 0
    assert player_num_resource_cards(state, players[0].color, WHEAT) == 0
    assert player_num_resource_cards(state, players[0].color, ORE) == 1

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [ORE]



def test_buying_settlement_transaction_is_tracked_using_unknown():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    # 1 wood, 2 brick, 1 wheat, 1 sheep, 1 ore
    player_freqdeck_add(game.state, Color.RED, [1, 2, 1, 1, 1])
    
    CardCounting_Blue = CardCounting(game, Color.BLUE)
    # 1 wood, 2 brick, 0 wheat, 0 sheep, 1 ore, 2 unknown
    CardCounting_Blue.assumed_resources[Color.RED][WOOD] = 1
    CardCounting_Blue.assumed_resources[Color.RED][BRICK] = 2
    CardCounting_Blue.assumed_resources[Color.RED][SHEEP] = 0
    CardCounting_Blue.assumed_resources[Color.RED][WHEAT] = 0
    CardCounting_Blue.assumed_resources[Color.RED][ORE] = 1
    CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] = 2
    CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] = [SHEEP, WHEAT]


    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3)
    apply_action(game.state, action=action)

    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert player_num_resource_cards(game.state, Color.RED, WOOD) == 0
    assert player_num_resource_cards(game.state, Color.RED, BRICK) == 1
    assert player_num_resource_cards(game.state, Color.RED, SHEEP) == 0
    assert player_num_resource_cards(game.state, Color.RED, WHEAT) == 0
    assert player_num_resource_cards(game.state, Color.RED, ORE) == 1

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == []

def test_buying_settlement_transaction_is_tracked_using_unknown_against_state():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.WHITE)]
    game = Game(players)
    p0_color = game.state.colors[2]
    p1_color = game.state.colors[1]
    p2_color = game.state.colors[0]
    print('p2_color:', p2_color)
    print('p1_color:', p1_color)
    print('p0_color:', p0_color)
    # map setup
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
    assert player_num_resource_cards(game.state, p2_color, WHEAT) == 1
    assert player_num_resource_cards(game.state, p1_color, WHEAT) == 0

    CardCounting_Blue.update_opponent_resources(game.state, action_rob)
    assert CardCounting_Blue.assumed_resources[p2_color][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[p2_color]['unknown_list'] == [WHEAT]

    action = Action(p2_color, ActionType.BUILD_ROAD, (0, 5))
    apply_action(game.state, action=action)
    action = Action(p2_color, ActionType.BUILD_SETTLEMENT, 5)
    apply_action(game.state, action=action)
    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert player_num_resource_cards(game.state, p2_color, WOOD) == 0
    assert player_num_resource_cards(game.state, p2_color, BRICK) == 0
    assert player_num_resource_cards(game.state, p2_color, SHEEP) == 0
    assert player_num_resource_cards(game.state, p2_color, WHEAT) == 0
    assert player_num_resource_cards(game.state, p2_color, ORE) == 0

    assert CardCounting_Blue.assumed_resources[p2_color][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[p2_color][BRICK] == 0
    assert CardCounting_Blue.assumed_resources[p2_color][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[p2_color][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[p2_color][ORE] == 0
    assert CardCounting_Blue.assumed_resources[p2_color][UNKNOWN] == 0
    assert CardCounting_Blue.assumed_resources[p2_color]['unknown_list'] == []

def test_buying_city_transaction_is_tracked_using_assumed_against_state():
    # needs work
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.state.board.build_settlement(Color.RED, 3, True)

    # 1 wood, 1 brick, 2 wheat, 1 sheep, 3 ore
    player_freqdeck_add(game.state, Color.RED, [1, 1, 1, 2, 3])
    
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
    apply_action(game.state, action=action)

    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert player_num_resource_cards(game.state, Color.RED, WOOD) == 1
    assert player_num_resource_cards(game.state, Color.RED, BRICK) == 1
    assert player_num_resource_cards(game.state, Color.RED, SHEEP) == 0
    assert player_num_resource_cards(game.state, Color.RED, WHEAT) == 1
    assert player_num_resource_cards(game.state, Color.RED, ORE) == 0

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 2
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [WOOD, SHEEP]
    pass

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
    # needs 
    # work
    
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    game.state.board.build_settlement(Color.RED, 3, True)

    # 1 wood, 1 brick, 2 wheat, 1 sheep, 3 ore
    player_freqdeck_add(game.state, Color.RED, [1, 1, 1, 2, 3])
    
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
    apply_action(game.state, action=action)

    CardCounting_Blue.update_opponent_resources(game.state, action)

    assert player_num_resource_cards(game.state, Color.RED, WOOD) == 1
    assert player_num_resource_cards(game.state, Color.RED, BRICK) == 1
    assert player_num_resource_cards(game.state, Color.RED, SHEEP) == 0
    assert player_num_resource_cards(game.state, Color.RED, WHEAT) == 1
    assert player_num_resource_cards(game.state, Color.RED, ORE) == 0

    assert CardCounting_Blue.assumed_resources[Color.RED][WOOD] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][BRICK] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED][SHEEP] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][WHEAT] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][ORE] == 0
    assert CardCounting_Blue.assumed_resources[Color.RED][UNKNOWN] == 1
    assert CardCounting_Blue.assumed_resources[Color.RED]['unknown_list'] == [SHEEP]
    pass

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



# test_buying_road_transaction_is_tracked_using_unknown_against_state()
# test_buying_road_transaction_is_tracked_using_assumed_against_state()
# test_buying_road_transaction_is_tracked_using_assumed()
# test_buying_road_transaction_is_tracked_using_unknown()

# test_buying_settlement_transaction_is_tracked_using_assumed_against_state()
# test_buying_settlement_transaction_is_tracked_using_assumed()
test_buying_settlement_transaction_is_tracked_using_unknown_against_state()
# test_buying_settlement_transaction_is_tracked_using_unknown()

# # test_buying_city_transaction_is_tracked_using_unknown_against_state()
# test_buying_city_transaction_is_tracked_using_unknown()
# # test_buying_city_transaction_is_tracked_using_assumed_against_state()
# test_buying_city_transaction_is_tracked_using_assumed()