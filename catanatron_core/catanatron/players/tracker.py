import pdb
from catanatron.models.decks import freqdeck_from_listdeck
from catanatron.models.enums import BRICK, ORE, RESOURCES, SHEEP, UNKNOWN, WHEAT, WOOD
from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.state_functions import get_player_freqdeck


# WEIGHTS_BY_ACTION_TYPE = {
#     ActionType.BUILD_CITY: 10000,
#     ActionType.BUILD_SETTLEMENT: 1000,
#     ActionType.BUY_DEVELOPMENT_CARD: 100,
# }


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
    def __init__(self, color, players=None, game=None, state=None):
        """Saves k and color. Creates an internal data structure to keep track of enemies' hands.

        Args:
            color (enum): id_of_player
            last_action_index (int): index of the last action processed
        """
        self.last_action_index = -1  # Track the last action processed
        self.color = color
        self.assumed_resources = {}
        self.initial_settlement = {}
        self.initial_road = {}
        self.someone_is_road_building = False
        try:
            for player in players:
                self.assumed_resources[player.color] = {
                        WOOD: 0,
                        BRICK: 0,
                        SHEEP: 0,
                        WHEAT: 0,
                        ORE: 0,
                        UNKNOWN: 0,
                        'unknown_list': []
            }
            for player in players:
                self.initial_settlement[player.color] = 0
                self.initial_road[player.color] = 0
        except:
            try:
                for player in game.state.colors:
                    self.assumed_resources[player] = {
                            WOOD: 0,
                            BRICK: 0,
                            SHEEP: 0,
                            WHEAT: 0,
                            ORE: 0,
                            UNKNOWN: 0,
                            'unknown_list': []
                }
                for player in game.state.colors:
                    self.initial_settlement[player] = 0
                    self.initial_road[player] = 0
            except:
                for player in state.colors:
                    self.assumed_resources[player] = {
                            WOOD: 0,
                            BRICK: 0,
                            SHEEP: 0,
                            WHEAT: 0,
                            ORE: 0,
                            UNKNOWN: 0,
                            'unknown_list': []
                }
                for player in state.colors:
                    self.initial_settlement[player] = 0
                    self.initial_road[player] = 0


    def update_opponent_resources(self, state, action):
        """
        This function updates opponent resource counts based on the action type.

        Args:
            action: The action object containing information about the action performed.
        """
        resource_cost_map = {
            ActionType.BUILD_CITY: [0, 0, 0, 2, 3],
            ActionType.BUY_DEVELOPMENT_CARD: [0, 0, 1, 1, 1]
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
            if action.value[0] + action.value[1] != 7:
                payouts = state.last_payout
                for color, freqdeck in payouts.items():
                    player_assumed_freqdeck_add(color, freqdeck)


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
                if victim != action.color:
                    self.assumed_resources[action.color][action.value] += self.assumed_resources[victim][action.value]
                    self.assumed_resources[victim][action.value] = 0
                    for rez in self.assumed_resources[victim]['unknown_list']:
                        if rez == action.value:
                            self.assumed_resources[victim]['unknown_list'].remove(rez)
                            self.assumed_resources[victim][UNKNOWN] -= 1
                            self.assumed_resources[action.color][action.value] += 1


        elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
            self.someone_is_road_building = True


        elif action.action_type == ActionType.MOVE_ROBBER:
            victim = action.value[1]
            robbed_resource = action.value[2]

            # if no one is robbed, no need to update anything
            if victim != None and robbed_resource != None:
                # if either the robber or the victim is the player, there are no unknowns
                if action.color == self.color or victim == self.color:
                    # add the robbed resource to the robber's resources
                    self.assumed_resources[action.color][robbed_resource] += 1
                    # remove the robbed resource from the victim's resources/unknowns
                    if self.assumed_resources[victim][robbed_resource] > 0:
                        self.assumed_resources[victim][robbed_resource] -= 1
                    else:
                        self.assumed_resources[victim][UNKNOWN] -= 1
                        self.assumed_resources[victim]['unknown_list'].remove(robbed_resource)
                # if the player isn't involved, there are unknowns
                else:
                    # add the robbed resource to the robber's resources
                    self.assumed_resources[action.color][UNKNOWN] += 1
                    self.assumed_resources[action.color]['unknown_list'].append(robbed_resource)
                    # remove the robbed resource from the victim's resources/unknowns
                    # check which of the known resources might've been stolen
                    possibly_stolen = []
                    for resource in RESOURCES:
                        if self.assumed_resources[victim][resource] > 0:
                            self.assumed_resources[victim][resource] -= 1
                            possibly_stolen.append(resource)
                    
                    # add those to unknowns
                    self.assumed_resources[victim][UNKNOWN] += len(possibly_stolen)
                    self.assumed_resources[victim]['unknown_list'].extend(possibly_stolen)

                    # unless unknowns are 0, remove the stolen resource from unknowns
                    if self.assumed_resources[victim][UNKNOWN] > 0:
                        if len(possibly_stolen) == 0 and robbed_resource != None:
                            self.assumed_resources[victim][UNKNOWN] -= 1
                            self.assumed_resources[victim]['unknown_list'].remove(robbed_resource)



        elif action.action_type == ActionType.OFFER_TRADE:
            trade_action_value = action.value
            offered = trade_action_value[:5]
            # rather than amend the object, we can create a copy of the object
            requester_assumed_resources = self.assumed_resources[action.color]
            for resource_quantity, resource_index in offered:
                # then, track each resource in the offering;
                # if the resource is available, chillin
                # if the resource is not available:
                if requester_assumed_resources[RESOURCES[resource_index]] < resource_quantity:
                    # take from unknown and unknown list,
                    diff = resource_quantity - requester_assumed_resources[RESOURCES[resource_index]]

                    requester_assumed_resources[UNKNOWN] -= diff
                    for i in range(diff):
                        requester_assumed_resources['unknown_list'].remove(RESOURCES[resource_index])
                    # and add to that resource
                    requester_assumed_resources[RESOURCES[resource_index]] += diff



        elif action.action_type == ActionType.ACCEPT_TRADE:
            trade_action_value = action.value
            asking = trade_action_value[5:]
            requester_assumed_resources = self.assumed_resources[action.color]
            for resource_quantity, resource_index in asking:
                # then, track each resource in the asking;
                # if the resource is available, chillin
                # if the resource is not available:
                if requester_assumed_resources[RESOURCES[resource_index]] < resource_quantity:
                    # take from unknown and unknown list,
                    diff = resource_quantity - requester_assumed_resources[RESOURCES[resource_index]]

                    requester_assumed_resources[UNKNOWN] -= diff
                    for i in range(diff):
                        requester_assumed_resources['unknown_list'].remove(RESOURCES[resource_index])
                    # and add to that resource
                    requester_assumed_resources[RESOURCES[resource_index]] += diff



        elif action.action_type == ActionType.CONFIRM_TRADE:
            trade_action_value = action.value
            
            offered = trade_action_value[:5]
            if self.assumed_resources[action.color][RESOURCES[resource_index]] < resource_quantity:
                    # take from unknown and unknown list,
                    diff = resource_quantity - self.assumed_resources[action.color][RESOURCES[resource_index]]

                    requester_assumed_resources[UNKNOWN] -= diff
                    for i in range(diff):
                        requester_assumed_resources['unknown_list'].remove(RESOURCES[resource_index])
                    # and add to that resource
                    self.assumed_resources[action.color][RESOURCES[resource_index]] += diff

            for resource_quantity, resource_index in offered:
                for i in range(resource_quantity):
                    if self.assumed_resources[action.color][RESOURCES[resource_index]] > 0:
                        self.assumed_resources[action.color][RESOURCES[resource_index]] -= 1
                    else:
                        self.assumed_resources[action.color][UNKNOWN] -= 1
                        self.assumed_resources[action.color]['unknown_list'].remove(RESOURCES[resource_index])
            
            asking = trade_action_value[5:]
            for resource_quantity, resource_index in asking:
                for i in range(resource_quantity):
                    if self.assumed_resources[action.color][RESOURCES[resource_index]] > 0:
                        self.assumed_resources[action.color][RESOURCES[resource_index]] -= 1
                    else:
                        self.assumed_resources[action.color][UNKNOWN] -= 1
                        self.assumed_resources[action.color]['unknown_list'].remove(RESOURCES[resource_index])



        elif action.action_type == ActionType.MARITIME_TRADE:
            # no longer needs to check if trade is legal since moved from execute to end of apply_action
            trade_offer = action.value
            giving = trade_offer[:-1]
            givingcost = 0
            for i in range (len(giving)):
                if giving[i] != None:
                    givingcost += 1

            givingrez = self.assumed_resources[action.color][giving[0]]
            for i in range(self.assumed_resources[action.color][UNKNOWN]):
                if self.assumed_resources[action.color]['unknown_list'][i] == giving[0]:
                    givingrez += 1
            
            if givingrez >= givingcost:
                self.assumed_resources[action.color][trade_offer[-1]] += 1

                for resource in giving:
                    if resource != None:
                        if self.assumed_resources[action.color][resource] > 0:
                            self.assumed_resources[action.color][resource] -= 1
                        else:
                            self.assumed_resources[action.color][UNKNOWN] -= 1
                            self.assumed_resources[action.color]['unknown_list'].remove(resource)
            else:
                print('ha! back!')


        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            resource_cost = [1, 1, 1, 1, 0]
            if self.initial_settlement[action.color] == 2:
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
            elif self.initial_settlement[action.color] == 0:
                self.initial_settlement[action.color] = 1
            else:
                for tile in state.board.map.adjacent_tiles[action.value]:
                    if tile.resource != None:
                        self.assumed_resources[action.color][tile.resource] += 1
                self.initial_settlement[action.color] = 2
            


        elif action.action_type == ActionType.BUILD_ROAD:
            if self.someone_is_road_building == False:
                resource_cost = [1, 1, 0, 0, 0]
                if self.initial_road[action.color] == 2:
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

                elif self.initial_road[action.color] == 0:
                    self.initial_road[action.color] = 1
                elif self.initial_road[action.color] == 1:
                    self.initial_road[action.color] = 2



        elif action.action_type in resource_cost_map:
            resource_cost = resource_cost_map[action.action_type]

            for resource_index, quantity in enumerate(resource_cost):
                resource = RESOURCES[resource_index]

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

