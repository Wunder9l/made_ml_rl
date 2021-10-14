from collections import Counter
from enum import Enum

from gym import spaces
from gym.envs.toy_text import BlackjackEnv
from gym.envs.toy_text.blackjack import cmp


class Action(Enum):
    STICK = 0
    HIT = 1
    DOUBLE = 2


class Card(Enum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13


def card_score(card: Card) -> int:
    if card.value > 10:
        return 10
    elif card == Card.ACE:
        return 11
    else:
        return card.value


def card_points(card: Card) -> int:
    if 2 <= card.value <= 6:
        return +1
    elif 7 <= card.value <= 9:
        return 0
    else:
        return -1


class BlackjackWithDouble(BlackjackEnv):
    def __init__(self, natural=False, sab=False):
        super().__init__(natural, sab)
        self.action_space = spaces.Discrete(3)
        self.reward_multiplier = 1.0

    def step(self, action):
        if isinstance(action, Action):
            action = action.value
        if not self.action_space.contains(action):
            print(self.action_space, action)
        assert self.action_space.contains(action)
        if action == Action.DOUBLE.value:
            self.reward_multiplier = 2.0
            s, r, is_done, other = super().step(Action.HIT.value)
            if is_done:
                return s, r, is_done, other
            return super().step(Action.STICK.value)
        else:
            state, reward, done, other = super().step(action)
            return state, reward * self.reward_multiplier, done, other

    def reset(self):
        self.reward_multiplier = 1.0
        return super(BlackjackWithDouble, self).reset()


class Hand:
    def __init__(self):
        self.usable_ace = 0
        self.sum = 0

    def is_bust(self):
        if self.sum > 21 and self.usable_ace > 0:
            print(self.sum, self.usable_ace)
        assert not (self.sum > 21 and self.usable_ace > 0)
        return self.sum > 21

    def add(self, card: Card):
        if card == Card.ACE:
            self.usable_ace += 1
        self.sum += card_score(card)
        while self.sum > 21 and self.usable_ace > 0:
            self.usable_ace -= 1
            self.sum -= 10


class BlackjackOneDeck(BlackjackWithDouble):
    def __init__(self, natural=False, sab=False):
        super().__init__(natural, sab)
        self.shuffle_count = 0
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2), spaces.Discrete(20 + 1 + 20))
        )
        self._shuffle_deck()

    def step(self, action):
        if isinstance(action, Action):
            action = action.value
        if not self.action_space.contains(action):
            print(self.action_space, action)
        assert self.action_space.contains(action)

        if action == Action.DOUBLE.value:
            self.reward_multiplier = 2.0
            s, r, is_done, other = self.step(Action.HIT.value)
            if is_done:
                return s, r, is_done, other
            return self.step(Action.STICK.value)
        elif action == Action.HIT.value:
            return self._hit()
        else:
            return self._stick()

    def _hit(self):
        self.player.add(self._draw_card())
        if self.player.is_bust():
            return self._get_obs(), -1.0 * self.reward_multiplier, True, {}
        return self._get_obs(), 0, False, {}

    def _get_reward(self):
        if self.player.is_bust() and self.dealer.is_bust():
            return 0.0
        elif self.dealer.is_bust():
            return 1.0
        elif self.player.is_bust():
            return -1.0
        else:
            return cmp(self.player.sum, self.dealer.sum)

    def _stick(self):
        while self.dealer.sum < 17:
            self.dealer.add(self._draw_card())
        reward = self._get_reward()
        return self._get_obs(), reward * self.reward_multiplier, True, {}

    def _get_obs(self):
        return self.player.sum, self.dealer.sum, self.player.usable_ace > 0, self.balance

    def _draw_card(self):
        if len(self.deck) <= 15:
            self._shuffle_deck()
        card = self.deck.pop()
        self.balance += card_points(card)
        # print(f'card {card}, balance {self.balance}, left {len(self.deck)}: {Counter(self.deck)}')
        return card

    def reset(self):
        self.reward_multiplier = 1.0
        self.dealer = Hand()
        self.dealer.add(self._draw_card())
        self.player = Hand()
        self.player.add(self._draw_card())
        self.player.add(self._draw_card())
        return self._get_obs()

    def _shuffle_deck(self):
        self.shuffle_count += 1
        self.deck = [card for card in Card] * 4
        self.np_random.shuffle(self.deck)
        self.balance = 0
