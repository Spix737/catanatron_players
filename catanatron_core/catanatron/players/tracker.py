import random

from catanatron.game import Game
from catanatron.models.decks import freqdeck_from_listdeck
from catanatron.models.enums import BRICK, ORE, RESOURCES, SHEEP, UNKNOWN, WHEAT, WOOD
from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.state import yield_resources


# WEIGHTS_BY_ACTION_TYPE = {
#     ActionType.BUILD_CITY: 10000,
#     ActionType.BUILD_SETTLEMENT: 1000,
#     ActionType.BUY_DEVELOPMENT_CARD: 100,
# }
ROAD_COST_FREQDECK = [1, 1, 0, 0, 0]
SETTLEMENT_COST_FREQDECK = [1, 1, 1, 1, 0]
CITY_COST_FREQDECK = [0, 0, 0, 2, 3]
DEVELOPMENT_CARD_COST_FREQDECK = [0, 0, 1, 1, 1]


class ResourceTrackingPlayer(Player):
    """
    Player that tracks opponent's resources
    Built on top of weighted random player, to be amended as seen fit
    """
    def __init__(self):
        self.card_counting_module = CardCounting(color=self.color)

    def decide(self, game, playable_actions):
        # get latest action(s)
        # pass this to update
        self.card_counting_module.update(game.state.last_action)
        # TODO: Use enhanced_state to make better decisions
        
        pass
        # return random.choice(playable_actions)


        # TODO: Use enhanced_state to make better decisions
        # Old Implementation of WeightedRandomPlayer
        # bloated_actions = []
        # for action in playable_actions:
            # weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            # bloated_actions.extend([action] * weight)

        # return random.choice(bloated_actions)


class CardCounting:
    def __init__(self, game: Game, color):
        """Saves k and color. Creates an internal data structure to keep track of enemies' hands.

        Args:
            color (enum): id_of_player
            last_action_index (int): index of the last action processed
        """
        self.last_action_index = -1  # Track the last action processed
        self.color = color 
        self.assumed_resources = [player for player in game.state.colors]
        for opponent in self.assumed_resources:
            self.assumed_resources[opponent] = {
                BRICK: 0,
                WOOD: 0,
                WHEAT: 0,
                ORE: 0,
                SHEEP: 0,
                UNKNOWN: 0,
                'unknown_list': []
            }
        # for opponent in self.assumed_resources:
        pass


    def update_opponent_resources(self, state, action):
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

        dev_card_map = {
            # ActionType.PLAY_KNIGHT_CARD: 1, 517!!!!!!!!517!!!!!!!!!!
            ActionType.PLAY_YEAR_OF_PLENTY: 1,
            ActionType.PLAY_MONOPOLY: 1,
        }

        def player_assumed_freqdeck_add(color, freqdeck):
            self.assumed_resources[color][WOOD] += freqdeck[0]
            self.assumed_resources[color][BRICK] += freqdeck[1]
            self.assumed_resources[color][SHEEP] += freqdeck[2]
            self.assumed_resources[color][WHEAT] += freqdeck[3]
            self.assumed_resources[color][ORE] += freqdeck[4]



        if action.action_type == ActionType.ROLL:
            if action.value != 7:
                payout, _ = yield_resources(state.board, state.resource_freqdeck, action.value)
                for color, resource_freqdeck in payout.items():
                    # Atomically add to player's assumed hand
                    player_assumed_freqdeck_add(color, resource_freqdeck)



        elif action.action_type == ActionType.DISCARD:
            if action.value is not None:
                discard_deck = freqdeck_from_listdeck(action.value)
                for resource_index, quantity in enumerate(discard_deck):
                    resource = RESOURCES[resource_index]

                    available = self.assumed_resources[action.color][resource]
                    self.assumed_resources[action.color][resource] = max(0, available - quantity)

                    if available < quantity:
                        self.assumed_resources[action.color][UNKNOWN] -= (quantity - available)
                        for i in range(quantity - available):
                            self.assumed_resources[action.color]['unknown_list'].remove(resource)
            else:
                print("No resources passed to discard from assumed\nThis could be problematic.................!")



        elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            for resource in action.value:
                self.assumed_resources[action.color][resource] += 1



        elif action.action_type == ActionType.PLAY_MONOPOLY:
            for victim in self.assumed_resources:
                self.assumed_resources[action.color][action.value] += self.assumed_resources[victim][action.value]
                self.assumed_resources[victim][action.value] = 0
                for rez in self.assumed_resources[victim]['unknown_list']:
                    if rez == action.value:
                        self.assumed_resources[victim]['unknown_list'].remove(rez)
                        self.assumed_resources[victim][UNKNOWN] -= 1
                        self.assumed_resources[action.color][action.value] += 1



        elif action.action_type == ActionType.MOVE_ROBBER:
            victim = action.value[1]
            if action.color or victim == self.color:
                self.assumed_resources[action.color][action.value[2]] += 1
                if self.assumed_resources[victim][action.value[2]] > 0:
                    self.assumed_resources[victim][action.value[2]] -= 1
                else:
                    self.assumed_resources[victim][UNKNOWN] -= 1
                    self.assumed_resources[victim]['unknown_list'].remove(action.value[2])
            else:
                self.assumed_resources[action.color][UNKNOWN] += 1
                self.assumed_resources[action.color]['unknown_list'].append(action.value[2])
                possibly_stolen = []
                for resource in RESOURCES:
                    if self.assumed_resources[victim][resource] > 0:
                        self.assumed_resources[victim][resource] -= 1
                        possibly_stolen.append(resource)

                self.assumed_resources[victim][UNKNOWN] += max(len(possibly_stolen) - 1, 0)
                self.assumed_resources[victim]['unknown_list'].append(possibly_stolen)



        elif action.action_type == ActionType.OFFER_TRADE:
            pass

        elif action.action_type == ActionType.ACCEPT_TRADE:
            pass

        elif action.action_type == ActionType.CONFIRM_TRADE:
            pass

        elif action.action_type == ActionType.MARITIME_TRADE:
            pass



        elif action.action_type in resource_cost_map:
            resource_cost = resource_cost_map[action.action_type]
            for resource_index, quantity in enumerate(resource_cost):
                resource = RESOURCES[resource_index]

                # Ensure resource doesn't go below 0
                available = self.assumed_resources[action.color][resource]
                self.assumed_resources[action.color][resource] = max(0, available - quantity)

                # If any quantity was unaccounted for, subtract from UNKNOWN
                if available < quantity:
                    self.assumed_resources[action.color][UNKNOWN] -= (quantity - available)
                    for i in range(quantity - available):
                        self.assumed_resources[action.color]['unknown_list'].remove(resource)

        else:
            raise ValueError(f"Unsupported ActionType: {action.action_type}")
        



    def update(self, actions):
        """ Updates the internal state based on the last action
        Args:
            actions (_type_): _description_
        """
        for action in actions:
            if action.action_type in [
                ActionType.ROLL,
                ActionType.DISCARD,

                ActionType.MOVE_ROBBER, 
                # ActionType.PLAY_KNIGHT_CARD,
                ActionType.PLAY_YEAR_OF_PLENTY,
                ActionType.PLAY_MONOPOLY,

                ActionType.BUY_DEVELOPMENT_CARD, 
                ActionType.BUILD_CITY,
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_ROAD,

                ActionType.OFFER_TRADE, 
                ActionType.ACCEPT_TRADE, 
                ActionType.CONFIRM_TRADE,
                ActionType.MARITIME_TRADE,
                ]:
                print(action.action_type)


            if action.action_type == ActionType.BUILD_ROAD:
                if self.assumed_resources[action.color][BRICK] == 0:
                    self.assumed_resources[action.color][UNKNOWN] -= 1
                else:
                    self.assumed_resources[action.color][BRICK] -= 1
                if self.assumed_resources[action.color][WOOD] == 0:
                    self.assumed_resources[action.color][UNKNOWN] -= 1
                else:
                    self.assumed_resources[action.color][WOOD] -= 1

            elif action.action_type == ActionType.BUILD_SETTLEMENT:
                # grab player
                if self.assumed_resources[action.color][BRICK] == 0:
                    self.assumed_resources[action.color][UNKNOWN] -= 1
                else:
                    self.assumed_resources[action.color][BRICK] -= 1
                if self.assumed_resources[action.color][WOOD] == 0:
                    self.assumed_resources[action.color][UNKNOWN] -= 1
                else:
                    self.assumed_resources[action.color][WOOD] -= 1
                if self.assumed_resources[action.color][WHEAT] == 0:
                    self.assumed_resources[action.color][UNKNOWN] -= 1
                else:
                    self.assumed_resources[action.color][WHEAT] -= 1
                if self.assumed_resources[action.color][SHEEP] == 0:
                    self.assumed_resources[action.color][UNKNOWN] -= 1
                else:
                    self.assumed_resources[action.color][SHEEP] -= 1

            elif action.action_type == ActionType.BUILD_CITY:
                if self.assumed_resources[action.color][ORE] == 3:
                    if self.assumed_resources[action.color][ORE] == 2:
                        if self.assumed_resources[action.color][ORE] == 1:
                            if self.assumed_resources[action.color][ORE] == 0:
                                self.assumed_resources[action.color][UNKNOWN] -= 3
                            self.assumed_resources[action.color][ORE] -= 1
                            self.assumed_resources[action.color][UNKNOWN] -= 2
                        self.assumed_resources[action.color][ORE] -= 2
                        self.assumed_resources[action.color][UNKNOWN] -= 1
                    self.assumed_resources[action.color][ORE] -= 3

                if self.assumed_resources[action.color][WHEAT] == 2:
                    if self.assumed_resources[action.color][WHEAT] == 1:
                        if self.assumed_resources[action.color][WHEAT] == 0:
                            self.assumed_resources[action.color][UNKNOWN] -= 2
                        self.assumed_resources[action.color][WHEAT] -= 1
                        self.assumed_resources[action.color][UNKNOWN] -= 1
                    self.assumed_resources[action.color][WHEAT] -= 2
        pass

    # ACTOR = PERFORMER
    # VICTIM/TRADE-COLLABORATOR = ASSISTANT

    def transact(self, action, costFreqdeck):
        
        for index in costFreqdeck:
            for i in range(index):
                if self.assumed_resources[action.color][index] == 0:
                    self.assumed_resources[action.color][UNKNOWN] -= 1
                else:
                    self.assumed_resources[action.color][index] -= 1