import random

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.models.actions import ActionType


WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}


class ResourceTrackingPlayer(Player):
    """
    Weighted Random Player but also tracks opponent's resources
    """
    def __init__(self):
        self.card_counting_module = CardCounting(color=self.color)


    def decide(self, game, playable_actions):
        self.card_counting_module.update(game.state.actions)
        enhanced_state = self.card_counting_module.get_enhanced_state(game, self.color)
        
        # TODO: Use enhanced_state to make better decisions

        # Old Implementation of WeightedRandomPlayer
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)


class CardCounting:
    def __init__(self, game: Game, color):
        """Saves k and color. Creates an internal data structure to keep track of enemies' hands.

        Args:
            color (_type_): id_of_player
        """
        self.color = color 
        self.opponents = [player for player in game.state.colors if player != self.color]
        for opponent in self.opponents:
            self.opponents[opponent] = {
                'brick': 0,
                'wood': 0,
                'wheat': 0,
                'ore': 0,
                'sheep': 0,
                'unknown': 0
            }
        pass



    def update(self, last_action):
        """ Updates the internal state based on the last action
        Args:
            actions (_type_): _description_
        """
        if last_action.action_type in [ActionType.MOVE_ROBBER, ActionType.BUY_DEVELOPMENT_CARD]:
            print(last_action.action_type)
        pass



    def get_enhanced_state(game, color, card_counting_module):
        """Makes a copy of state and appends/adds data to it with the "assumed state".

        Args:
            game (_type_): _description_
            color (_type_): _description_
            card_counting_module (_type_): _description_

        Returns:
            _type_: _description_
        """
        enhanced_state = game.state.copy()

        del enhanced_state['P1_BRICK_IN_HAND']
        del enhanced_state['P1_BRICK_IN_HAND']
        del enhanced_state['P1_BRICK_IN_HAND']
        del enhanced_state['P1_BRICK_IN_HAND']

        # using card-counting-module
        enhanced_state['P1_ASSUMED_BRICKES'] = card_counting_module.get()
        enhanced_state['P1_ASSUMED_ORES'] = 1

        return enhanced_state