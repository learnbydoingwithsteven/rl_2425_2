"""
Linear Function Approximation (LFA) Agent for Blackjack
Project 2: The Gambler
"""
import numpy as np
from typing import Tuple, Callable, List
import pickle


class FeatureExtractor:
    """Feature extraction for state representation"""
    
    def __init__(self, feature_type='binary'):
        """
        Initialize feature extractor
        
        Args:
            feature_type: 'binary', 'polynomial', or 'combined'
        """
        self.feature_type = feature_type
        self.feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self) -> int:
        """Calculate feature dimension based on feature type"""
        if self.feature_type == 'binary':
            # Binary features: player_sum (32 values), dealer_card (10), usable_ace (2), peek (11)
            return 32 + 10 + 2 + 11  # 55 features
        elif self.feature_type == 'polynomial':
            # Polynomial features: degree 2
            # Base: player_sum, dealer_card, usable_ace, peek (4)
            # Degree 2: 4 + 4*3/2 = 4 + 6 = 10
            return 15  # Including bias
        elif self.feature_type == 'combined':
            # Both binary and polynomial
            return 55 + 15
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def extract(self, state: Tuple) -> np.ndarray:
        """Extract features from state"""
        player_sum, dealer_card, usable_ace, peek_info = state
        
        if self.feature_type == 'binary':
            return self._binary_features(player_sum, dealer_card, usable_ace, peek_info)
        elif self.feature_type == 'polynomial':
            return self._polynomial_features(player_sum, dealer_card, usable_ace, peek_info)
        elif self.feature_type == 'combined':
            binary = self._binary_features(player_sum, dealer_card, usable_ace, peek_info)
            poly = self._polynomial_features(player_sum, dealer_card, usable_ace, peek_info)
            return np.concatenate([binary, poly])
    
    def _binary_features(self, player_sum, dealer_card, usable_ace, peek_info) -> np.ndarray:
        """Binary feature representation"""
        features = np.zeros(55)
        
        # Player sum (one-hot encoding for 0-31)
        if 0 <= player_sum < 32:
            features[player_sum] = 1
        
        # Dealer card (one-hot encoding for 1-10)
        if 1 <= dealer_card <= 10:
            features[32 + dealer_card - 1] = 1
        
        # Usable ace (one-hot encoding)
        features[42 + int(usable_ace)] = 1
        
        # Peek info (one-hot encoding for 0-10)
        if 0 <= peek_info <= 10:
            features[44 + peek_info] = 1
        
        return features
    
    def _polynomial_features(self, player_sum, dealer_card, usable_ace, peek_info) -> np.ndarray:
        """Polynomial feature representation (degree 2)"""
        # Normalize values
        player_norm = player_sum / 31.0
        dealer_norm = dealer_card / 10.0
        ace_norm = float(usable_ace)
        peek_norm = peek_info / 10.0
        
        # Base features
        base = np.array([1.0, player_norm, dealer_norm, ace_norm, peek_norm])
        
        # Polynomial features (degree 2)
        poly = []
        for i in range(len(base)):
            for j in range(i, len(base)):
                poly.append(base[i] * base[j])
        
        return np.array(poly)


class LFAAgent:
    """Linear Function Approximation Agent using Semi-Gradient SARSA"""
    
    def __init__(self, 
                 n_actions: int,
                 feature_type: str = 'binary',
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.01):
        """
        Initialize LFA Agent
        
        Args:
            n_actions: Number of possible actions
            feature_type: Type of features ('binary', 'polynomial', 'combined')
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
        """
        self.n_actions = n_actions
        self.feature_extractor = FeatureExtractor(feature_type)
        self.feature_dim = self.feature_extractor.feature_dim
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Weight matrix: [n_actions x feature_dim]
        self.weights = np.zeros((n_actions, self.feature_dim))
        
        # Statistics
        self.training_stats = {
            'episodes': 0,
            'total_rewards': [],
            'win_rate': [],
            'epsilon_history': [],
            'avg_q_values': []
        }
    
    def get_q_values(self, state: Tuple) -> np.ndarray:
        """Get Q-values for all actions in given state"""
        features = self.feature_extractor.extract(state)
        q_values = self.weights @ features
        return q_values
    
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action
            q_values = self.get_q_values(state)
            return np.argmax(q_values)
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, next_action: int, done: bool):
        """
        Update weights using Semi-Gradient SARSA
        
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        """
        features = self.feature_extractor.extract(state)
        
        # Current Q-value
        q_current = self.weights[action] @ features
        
        # TD target
        if done:
            td_target = reward
        else:
            next_features = self.feature_extractor.extract(next_state)
            q_next = self.weights[next_action] @ next_features
            td_target = reward + self.gamma * q_next
        
        # TD error
        td_error = td_target - q_current
        
        # Update weights
        self.weights[action] += self.alpha * td_error * features
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save agent to file"""
        data = {
            'weights': self.weights,
            'feature_type': self.feature_extractor.feature_type,
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'training_stats': self.training_stats
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load agent from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.weights = data['weights']
        self.n_actions = data['n_actions']
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.training_stats = data['training_stats']
        self.feature_extractor = FeatureExtractor(data['feature_type'])


def train_agent(env, agent, n_episodes: int = 10000, verbose: bool = True) -> dict:
    """
    Train LFA agent on Blackjack environment
    
    Returns:
        Training statistics
    """
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'win_rate_window': [],
        'epsilon_history': []
    }
    
    wins = 0
    window_size = 100
    recent_results = []
    
    for episode in range(n_episodes):
        state = env.reset()
        action = agent.get_action(state, training=True)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if not done:
                next_action = agent.get_action(next_state, training=True)
                agent.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
            else:
                # Terminal state update
                agent.update(state, action, reward, next_state, 0, done)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['epsilon_history'].append(agent.epsilon)
        
        # Track wins
        recent_results.append(1 if episode_reward > 0 else 0)
        if len(recent_results) > window_size:
            recent_results.pop(0)
        
        win_rate = np.mean(recent_results) if recent_results else 0
        stats['win_rate_window'].append(win_rate)
        
        # Verbose output
        if verbose and (episode + 1) % 1000 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-1000:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Win Rate: {win_rate:.3f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    return stats


if __name__ == "__main__":
    from blackjack_env import BlackjackEnv
    
    print("Testing LFA Agent\n")
    
    # Create environment and agent
    env = BlackjackEnv(cheating_enabled=False)
    agent = LFAAgent(n_actions=2, feature_type='binary', alpha=0.01, epsilon=0.1)
    
    print(f"Feature dimension: {agent.feature_dim}")
    print(f"Weight shape: {agent.weights.shape}")
    
    # Test feature extraction
    state = env.reset()
    features = agent.feature_extractor.extract(state)
    print(f"\nState: {state}")
    print(f"Features shape: {features.shape}")
    print(f"Non-zero features: {np.sum(features != 0)}")
    
    # Test Q-value computation
    q_values = agent.get_q_values(state)
    print(f"\nQ-values: {q_values}")
    
    # Test action selection
    action = agent.get_action(state)
    print(f"Selected action: {action}")
    
    # Quick training test
    print("\n" + "="*50)
    print("Quick Training Test (1000 episodes)\n")
    
    stats = train_agent(env, agent, n_episodes=1000, verbose=True)
    
    print(f"\nFinal win rate: {stats['win_rate_window'][-1]:.3f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
