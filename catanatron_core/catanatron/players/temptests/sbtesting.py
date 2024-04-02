import random
import pdb
from catanatron.game import Game
from catanatron.models.enums import BRICK, ORE, RESOURCES, SHEEP, UNKNOWN, WHEAT, WOOD, Action, ActionType, FastResource
from catanatron.models.player import Color, SimplePlayer
from catanatron.state import apply_action, yield_resources

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
                    WOOD: 0,
                    WHEAT: 0,
                    ORE: 0,
                    SHEEP: 0,
                    UNKNOWN: 0,
                }
                # print('playercolor: ')
                # print(player_color)
                # print(self.opponents[player_color])
        pass

    def transact(self, action):
        print(action)
        print(action.color)
        """
        This function updates opponent resource counts based on the action type.

        Args:
            action: The action object containing information about the action performed.
        """
        resource_cost_map = {
            ActionType.BUILD_ROAD: ROAD_COST_FREQDECK,
            ActionType.BUILD_SETTLEMENT: SETTLEMENT_COST_FREQDECK,
            ActionType.BUILD_CITY: CITY_COST_FREQDECK,
            ActionType.BUY_DEVELOPMENT_CARD: DEVELOPMENT_CARD_COST_FREQDECK,
        }

        if action.action_type in resource_cost_map:
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
        


players = [SimplePlayer(color) for color in [Color.RED, Color.BLUE, Color.WHITE]]
game = Game(players=players)
test1 = Test(game, Color.RED)
test1.transact(action=Action(color=Color.WHITE, action_type=ActionType.BUILD_ROAD, value=None))
print(test1.opponents[Color.WHITE])