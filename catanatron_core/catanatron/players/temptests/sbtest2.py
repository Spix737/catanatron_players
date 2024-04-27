import random
import pdb
from catanatron.game import Game
from catanatron.models.enums import (
    BRICK,
    ORE,
    RESOURCES,
    SHEEP,
    UNKNOWN,
    WHEAT,
    WOOD,
    Action,
    ActionType,
    FastResource,
)
from catanatron.models.player import Color, SimplePlayer
from catanatron.players.tracker import CardCounting
from catanatron.state import apply_action, yield_resources

ROAD_COST_FREQDECK = [1, 1, 0, 0, 0]
SETTLEMENT_COST_FREQDECK = [1, 1, 1, 1, 0]
CITY_COST_FREQDECK = [0, 0, 0, 2, 3]
DEVELOPMENT_CARD_COST_FREQDECK = [0, 0, 1, 1, 1]
ResourceFreqdeck = [19, 19, 19, 19, 19]


class Test:
    def __init__(self, game: Game, color):
        self.last_action_index = -1  # Track the last action processed
        self.color = color
        self.opponents = {}
        # Populate the dictionary with opponent colors and initialize resource counts
        for player_color in game.state.colors:
            if player_color != self.color:
                print("color in state:", player_color)
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

    def transact(self, state, action):
        print("action: ", action, ", action color: ", action.color)
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

        def player_freqdeck_add(color, freqdeck):
            print("freqdeckcolor: ", color)
            self.opponents[color][WOOD] += freqdeck[0]
            self.opponents[color][BRICK] += freqdeck[1]
            self.opponents[color][SHEEP] += freqdeck[2]
            self.opponents[color][WHEAT] += freqdeck[3]
            self.opponents[color][ORE] += freqdeck[4]

        if action.action_type == ActionType.ROLL:
            payout, _ = yield_resources(
                state.board, state.resource_freqdeck, action.value
            )
            print("payout: ", payout)
            for color, resource_freqdeck in payout.items():
                if color != self.color:
                    # Atomically add to player's hand
                    player_freqdeck_add(color, resource_freqdeck)

        elif action.action_type in resource_cost_map:
            resource_cost = resource_cost_map[action.action_type]
            for resource_index, quantity in enumerate(resource_cost):
                resource = RESOURCES[resource_index]

                # Ensure resource doesn't go below 0
                available = self.opponents[action.color][resource]
                self.opponents[action.color][resource] = max(0, available - quantity)

                # If any quantity was unaccounted for, subtract from UNKNOWN
                if available < quantity:
                    self.opponents[action.color][UNKNOWN] -= quantity - available

        else:
            raise ValueError(f"Unsupported ActionType: {action.action_type}")


players = [SimplePlayer(color) for color in [Color.RED, Color.WHITE]]
game = Game(players=players)

p0_color = game.state.colors[0]
p1_color = game.state.colors[1]

test1 = Test(game, p0_color)

for tile in game.state.board.map.land_tiles.values():
    if tile.id == 6:
        tile.resource = SHEEP
        tile.number = 8
    else:
        tile.number = 2

apply_action(
    game.state,
    Action(color=p0_color, action_type=ActionType.BUILD_SETTLEMENT, value=35),
)
apply_action(
    game.state,
    Action(color=p0_color, action_type=ActionType.BUILD_ROAD, value=(35, 36)),
)

apply_action(
    game.state, Action(color=p1_color, action_type=ActionType.BUILD_SETTLEMENT, value=6)
)
apply_action(
    game.state, Action(color=p1_color, action_type=ActionType.BUILD_ROAD, value=(6, 7))
)

apply_action(
    game.state,
    Action(color=p1_color, action_type=ActionType.BUILD_SETTLEMENT, value=33),
)
apply_action(
    game.state,
    Action(color=p1_color, action_type=ActionType.BUILD_ROAD, value=(33, 34)),
)

apply_action(
    game.state,
    Action(color=p0_color, action_type=ActionType.BUILD_SETTLEMENT, value=31),
)
apply_action(
    game.state,
    Action(color=p0_color, action_type=ActionType.BUILD_ROAD, value=(31, 32)),
)

# test1.transact(game.state, action=Action(color=random.choice([p0_color, p1_color]) , action_type=ActionType.ROLL, value=8))
print(test1.opponents[p1_color])
CardCounting.update_opponent_resources(
    test1,
    game.state,
    action=Action(
        color=random.choice([p0_color, p1_color]), action_type=ActionType.ROLL, value=8
    ),
)
print(test1.opponents[p1_color])
