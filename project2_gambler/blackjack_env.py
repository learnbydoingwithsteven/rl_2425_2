"""
Simplified Blackjack Environment for Reinforcement Learning
Project 2: The Gambler
"""
import numpy as np
from typing import Tuple, Optional, List
import random


class BlackjackEnv:
    """
    Simplified Blackjack Environment
    
    State: (player_sum, dealer_card, usable_ace, deck_info)
    Actions: 0=Stand, 1=Hit, 2=Peek (if cheating enabled)
    """
    
    def __init__(self, cheating_enabled=False, peek_penalty=-10, peek_success_rate=0.7):
        self.cheating_enabled = cheating_enabled
        self.peek_penalty = peek_penalty
        self.peek_success_rate = peek_success_rate
        
        # Card values: 1-10 (Ace=1, Face cards=10)
        self.card_values = list(range(1, 11))
        self.card_probabilities = [1/13] * 9 + [4/13]  # 4 face cards worth 10
        
        self.reset()
    
    def reset(self) -> Tuple:
        """Reset the environment and return initial state"""
        # Initialize deck (infinite deck assumption for simplicity)
        self.deck = self._create_deck()
        
        # Deal initial cards
        self.player_cards = [self._draw_card(), self._draw_card()]
        self.dealer_cards = [self._draw_card(), self._draw_card()]
        
        # Track if player has peeked
        self.has_peeked = False
        self.peek_caught = False
        self.next_card_known = None
        
        # Game state
        self.done = False
        
        return self._get_state()
    
    def _create_deck(self) -> List[int]:
        """Create a shuffled deck"""
        deck = []
        for _ in range(4):  # 4 suits
            deck.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])  # A-10, J, Q, K
        random.shuffle(deck)
        return deck
    
    def _draw_card(self) -> int:
        """Draw a card from the deck"""
        if len(self.deck) < 10:
            self.deck = self._create_deck()
        return self.deck.pop()
    
    def _get_hand_value(self, cards: List[int]) -> Tuple[int, bool]:
        """
        Calculate hand value and check for usable ace
        Returns: (hand_value, has_usable_ace)
        """
        total = sum(cards)
        has_ace = 1 in cards
        usable_ace = False
        
        # Use ace as 11 if it doesn't bust
        if has_ace and total + 10 <= 21:
            total += 10
            usable_ace = True
        
        return total, usable_ace
    
    def _get_state(self) -> Tuple:
        """Get current state representation"""
        player_sum, usable_ace = self._get_hand_value(self.player_cards)
        dealer_card = self.dealer_cards[0]
        
        # Additional state info for cheating variant
        peek_info = 0
        if self.cheating_enabled:
            if self.next_card_known is not None:
                peek_info = self.next_card_known
        
        return (player_sum, dealer_card, usable_ace, peek_info)
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, dict]:
        """
        Execute action and return (next_state, reward, done, info)
        
        Actions:
        0 = Stand
        1 = Hit
        2 = Peek (if cheating enabled)
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")
        
        reward = 0
        info = {'action': action, 'peek_caught': False}
        
        # Action 2: Peek at next card (cheating)
        if action == 2 and self.cheating_enabled:
            if not self.has_peeked:
                # Check if caught
                if random.random() > self.peek_success_rate:
                    # Caught cheating!
                    self.peek_caught = True
                    self.done = True
                    reward = self.peek_penalty
                    info['peek_caught'] = True
                    return self._get_state(), reward, self.done, info
                else:
                    # Successfully peeked
                    self.next_card_known = self.deck[-1] if self.deck else self._draw_card()
                    self.has_peeked = True
                    info['peeked_card'] = self.next_card_known
                    return self._get_state(), 0, False, info
            else:
                # Already peeked, treat as stand
                action = 0
        
        # Action 1: Hit
        if action == 1:
            card = self._draw_card()
            self.player_cards.append(card)
            self.next_card_known = None  # Reset peek info
            
            player_sum, _ = self._get_hand_value(self.player_cards)
            
            if player_sum > 21:
                # Bust
                self.done = True
                reward = -1
                return self._get_state(), reward, self.done, info
            elif player_sum == 21:
                # Blackjack on hit, auto-stand
                action = 0
        
        # Action 0: Stand
        if action == 0:
            self.done = True
            
            # Dealer plays
            dealer_sum, _ = self._get_hand_value(self.dealer_cards)
            while dealer_sum < 17:
                self.dealer_cards.append(self._draw_card())
                dealer_sum, _ = self._get_hand_value(self.dealer_cards)
            
            # Determine winner
            player_sum, _ = self._get_hand_value(self.player_cards)
            
            if dealer_sum > 21:
                # Dealer busts, player wins
                reward = 1
            elif player_sum > dealer_sum:
                reward = 1
            elif player_sum < dealer_sum:
                reward = -1
            else:
                reward = 0  # Draw
            
            info['dealer_sum'] = dealer_sum
            info['player_sum'] = player_sum
        
        return self._get_state(), reward, self.done, info
    
    def get_action_space_size(self) -> int:
        """Return number of possible actions"""
        return 3 if self.cheating_enabled else 2
    
    def get_state_space_info(self) -> dict:
        """Return information about state space"""
        return {
            'player_sum_range': (4, 31),  # Min 2 cards, max before bust
            'dealer_card_range': (1, 10),
            'usable_ace': (0, 1),
            'peek_info_range': (0, 10) if self.cheating_enabled else (0, 0)
        }


def test_environment():
    """Test the Blackjack environment"""
    print("Testing Blackjack Environment\n")
    
    # Test basic environment
    env = BlackjackEnv(cheating_enabled=False)
    state = env.reset()
    print(f"Initial state: {state}")
    print(f"Player cards: {env.player_cards}")
    print(f"Dealer showing: {env.dealer_cards[0]}")
    
    # Play a simple game
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 10:
        # Simple policy: hit if sum < 17
        player_sum = state[0]
        action = 1 if player_sum < 17 else 0
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"\nStep {steps}: Action={'Hit' if action==1 else 'Stand'}")
        print(f"State: {state}, Reward: {reward}, Done: {done}")
    
    print(f"\nGame over! Total reward: {total_reward}")
    
    # Test cheating variant
    print("\n" + "="*50)
    print("Testing Cheating Variant\n")
    
    env_cheat = BlackjackEnv(cheating_enabled=True, peek_success_rate=1.0)
    state = env_cheat.reset()
    
    # Try peeking
    print(f"Initial state: {state}")
    state, reward, done, info = env_cheat.step(2)  # Peek
    print(f"After peek: State={state}, Info={info}")
    
    # Now hit or stand based on peeked card
    state, reward, done, info = env_cheat.step(1)  # Hit
    print(f"After hit: State={state}, Reward={reward}, Done={done}")
    
    if not done:
        state, reward, done, info = env_cheat.step(0)  # Stand
        print(f"After stand: Reward={reward}, Info={info}")


if __name__ == "__main__":
    test_environment()
