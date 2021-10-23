import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from card_service import is_natural, is_bust, cmp, score, sum_hand, usable_ace, draw_hand, draw_card


class Deck:
    MIN_VALUE = -32
    
    def __init__(self):
        self.cards = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16]
        self.card_values_map = {
            1: 0,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 1,
            7: 1,
            8: 0,
            9: 0,
            10: -2
        }
        self._deck_count = 52
        self._deck_value = 0

    def reset_deck(self):
        self.cards = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16]
        self._deck_count = 52
        self._deck_value = 0

    def draw_card(self):
        card_index = np.random.choice([i for i, count in enumerate(self.cards) if count > 0])
        card = card_index + 1
        self.cards[card_index] -= 1
        self._deck_count -= 1
        self._deck_value -= self.card_values_map[card]
        
        return card

    @property
    def deck_value(self):
        
        return self._deck_value - self.MIN_VALUE

    @property
    def deck_count(self):
        
        return self._deck_count



class BlackjackEnvDoubleDeck(gym.Env):

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(32), 
                spaces.Discrete(11),
                spaces.Discrete(2), 
                spaces.Discrete(65),
                spaces.Discrete(2)
            )
        )
        self.seed()
        self.deck = Deck()

        self.player = []
        self.dealer = []
        self.natural = natural

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
        return [seed]

    def step(self, action):
#         assert self.action_space.contains(action)
        unknown_dealer = True
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(self.deck.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        elif action == 0:  # stick: play out the dealers hand, and score
            unknown_dealer = False
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.deck.draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.0:
                reward = 1.5
        else:
            done = True
            self.player.append(self.deck.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(self.deck.draw_card())
                reward = cmp(score(self.player), score(self.dealer))
                if self.natural and is_natural(self.player) and reward == 1.0:
                    reward = 1.5
                reward *= 2
        
        return self._get_obs(unknown_dealer), reward, done, {}

    def _get_obs(self, unknown_dealer=True):
        known_value = self.deck.deck_value
        if unknown_dealer:
            known_value += self.deck.card_values_map[self.dealer[1]]
        
        return (
            sum_hand(self.player), 
            self.dealer[0], 
            usable_ace(self.player),
            known_value,
            unknown_dealer
        )

    def reset(self):
        self.dealer = []
        self.player = []
        if self.deck.deck_count < 15:
            self.deck.reset_deck()
        for _ in range(2):
            self.dealer.append(self.deck.draw_card())
            self.player.append(self.deck.draw_card())
        
        return self._get_obs()
