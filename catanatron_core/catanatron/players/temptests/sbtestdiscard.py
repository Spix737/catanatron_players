import random
import pdb
from catanatron.game import Game
from catanatron.models.decks import freqdeck_from_listdeck
from catanatron.models.enums import BRICK, ORE, RESOURCES, SHEEP, UNKNOWN, WHEAT, WOOD, Action, ActionType, FastResource
from catanatron.models.player import Color, SimplePlayer
from catanatron.players.tracker import CardCounting
from catanatron.state import apply_action, yield_resources
from catanatron.state_functions import get_player_freqdeck, player_deck_replenish, player_deck_to_array

ROAD_COST_FREQDECK = [1, 1, 0, 0, 0]
SETTLEMENT_COST_FREQDECK = [1, 1, 1, 1, 0]
CITY_COST_FREQDECK = [0, 0, 0, 2, 3]
DEVELOPMENT_CARD_COST_FREQDECK = [0, 0, 1, 1, 1]
ResourceFreqdeck = [19, 19, 19, 19, 19]

class Test():
    def __init__(self, game: Game, color):
        self.last_action_index = -1  # Track the last action processed
        self.color = color 
        self.opponents = {}
        # Populate the dictionary with opponent colors and initialize resource counts
        for player_color in game.state.colors:
            if player_color != self.color:
                print('color in state:', player_color)
                self.opponents[player_color] = {
                    BRICK: 0,
                    WOOD: 7,
                    WHEAT: 0,
                    ORE: 0,
                    SHEEP: 0,
                    UNKNOWN: 1,
                }
                # print('playercolor: ')
                # print(player_color)
                # print(self.opponents[player_color])
        pass


    def transact(self, state, action):
        # print('action: ', action, ', action color: ', action.color)
        resource_cost_map = {
            ActionType.BUILD_ROAD: ROAD_COST_FREQDECK,
            ActionType.BUILD_SETTLEMENT: SETTLEMENT_COST_FREQDECK,
            ActionType.BUILD_CITY: CITY_COST_FREQDECK,
            ActionType.BUY_DEVELOPMENT_CARD: DEVELOPMENT_CARD_COST_FREQDECK,
        }

        def player_freqdeck_add(color, freqdeck):
            self.opponents[color][WOOD] += freqdeck[0]
            self.opponents[color][BRICK] += freqdeck[1]
            self.opponents[color][SHEEP] += freqdeck[2]
            self.opponents[color][WHEAT] += freqdeck[3]
            self.opponents[color][ORE] += freqdeck[4]

        if action.action_type == ActionType.ROLL:
            payout, _ = yield_resources(state.board, state.resource_freqdeck, action.value)
            for color, resource_freqdeck in payout.items():
                if color != self.color:
                # Atomically add to player's hand
                    player_freqdeck_add(color, resource_freqdeck)

        elif action.action_type == ActionType.DISCARD:
            discard_deck = freqdeck_from_listdeck(action.value)
            for resource_index, quantity in enumerate(discard_deck):
                resource = RESOURCES[resource_index]

                available = self.opponents[action.color][resource]
                self.opponents[action.color][resource] = max(0, available - quantity)

                if available < quantity:
                    self.opponents[action.color][UNKNOWN] -= (quantity - available)


        elif action.action_type in resource_cost_map:
            resource_cost = resource_cost_map[action.action_type]
            for resource_index, quantity in enumerate(resource_cost):
                resource = RESOURCES[resource_index]

                # Ensure resource doesn't go below 0
                available = self.opponents[action.color][resource]
                self.opponents[action.color][resource] = max(0, available - quantity)

                # If any quantity was unaccounted for, subtract from UNKNOWN
                if available < quantity:
                    self.opponents[action.color][UNKNOWN] -= (quantity - available)
            
        else:
            raise ValueError(f"Unsupported ActionType: {action.action_type}")
        


players = [SimplePlayer(color) for color in [Color.RED, Color.WHITE]]
game = Game(players=players)

p0_color = game.state.colors[0]
p1_color = game.state.colors[1]

test1 = Test(game, p0_color)

# get p0_color's resources from state
player_deck_replenish(game.state, p1_color, ORE, 1)
player_deck_replenish(game.state, p1_color, WOOD, 7)


print('PRE-DISCARD')
print('real: ', get_player_freqdeck(game.state, p1_color))
print('assumed: ', test1.opponents[p1_color], '\n')

hand = player_deck_to_array(game.state, p1_color)
num_to_discard = len(hand) // 2
discarded = random.sample(hand, k=num_to_discard)
print('discarded: ', discarded)

game.execute(action=Action(p1_color, ActionType.DISCARD, discarded), validate_action=False)

# test1.transact(game.state, action=Action(color=p1_color, action_type=ActionType.DISCARD, value=discarded))
CardCounting.update_opponent_resources(test1, game.state, action=Action(color=p1_color, action_type=ActionType.DISCARD, value=discarded))

print('\nPOST-DISCARD')
print('real: ', get_player_freqdeck(game.state, p1_color))
print('assumed: ', test1.opponents[p1_color])


# 0 wood
# 1 Wheat
# 2 Sheep
# 6 Brick
# 8 Brick
# 9 Ore
