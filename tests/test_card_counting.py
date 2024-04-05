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
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    # map setup
    for tile in game.state.board.map.land_tiles.values():
        if tile.id == 0:
            tile.resource = WOOD
            tile.number = 8

        elif tile.id == 1:
            tile.resource = WHEAT
            tile.number = 5
        elif tile.id == 2:
            tile.resource = SHEEP
            tile.number = 5
        
        elif tile.id == 6:
            tile.resource = BRICK
            tile.number = 8
        
        elif tile.id == 7:
            tile.resource = ORE
            tile.number = 4
        
        elif tile.id == 8:
            tile.resource = SHEEP
            tile.number = 6
        
        elif tile.id == 9:
            tile.resource = ORE
            tile.number = 6
        else:
            tile.number = 2

    game.state.board.build_settlement(Color.RED, 1, initial_build_phase=True)
    game.state.board.build_road(Color.RED, (1, 2))
    game.state.board.build_settlement(Color.BLUE, 26, initial_build_phase=True)
    game.state.board.build_road(Color.BLUE, (26, 27))
    game.state.board.build_settlement(Color.BLUE, 28, initial_build_phase=True)
    game.state.board.build_road(Color.BLUE, (27, 28))
    game.state.board.build_settlement(Color.RED, 9, initial_build_phase=True)
    game.state.board.build_road(Color.RED, (9, 10))
    # red = sheep, brick, ore
    # blue = BRICK


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


    action_roll = Action(Color.RED, ActionType.ROLL, (6, 6))
    apply_action(game.state, action=action_roll)
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 4)
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
# test_buying_settlement_transaction_is_tracked_using_unknown_against_state()
# test_buying_settlement_transaction_is_tracked_using_unknown()

# # test_buying_city_transaction_is_tracked_using_unknown_against_state()
# test_buying_city_transaction_is_tracked_using_unknown()
# # test_buying_city_transaction_is_tracked_using_assumed_against_state()
# test_buying_city_transaction_is_tracked_using_assumed()