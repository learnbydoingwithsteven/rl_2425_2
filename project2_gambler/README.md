# Project 2: The Gambler 🎰

## Reinforcement Learning with Linear Function Approximation for Blackjack

A comprehensive implementation of RL agents using Linear Function Approximation (LFA) to play simplified Blackjack, with interactive visualizations and real-time training progress.

---

## 📋 Project Overview

### Main Focus
**Linear solutions and feature design** in Reinforcement Learning

### Problem Description
Implement a reinforcement learning agent to play a simplified version of Blackjack. The agent must decide whether to "hit" or "stand" based on the game state, learning optimal strategies through Linear Function Approximation.

### Challenging Variant
**Cheating Gambler**: The agent can peek at the next card in the deck, but risks a penalty if caught. This introduces a risk-reward trade-off in the action space.

---

## 🎯 Key Features

### 1. **Blackjack Environment** (`blackjack_env.py`)
- Simplified Blackjack rules with infinite deck
- State representation: (player_sum, dealer_card, usable_ace, peek_info)
- Actions: Stand (0), Hit (1), Peek (2, if cheating enabled)
- Rewards: +1 (win), -1 (loss), 0 (draw)
- Cheating variant with configurable penalty and success rate

### 2. **LFA Agent** (`lfa_agent.py`)
- **Semi-Gradient SARSA** algorithm
- **Three feature types**:
  - **Binary Features**: One-hot encoding (55 dimensions)
  - **Polynomial Features**: Degree-2 polynomials (15 dimensions)
  - **Combined Features**: Both binary and polynomial (70 dimensions)
- Epsilon-greedy exploration with decay
- Configurable hyperparameters (α, γ, ε)

### 3. **Interactive Streamlit App** (`app.py`)
- **Real-time training visualization**
- **Performance metrics dashboard**
- **Q-value heatmaps**
- **Policy visualization**
- **Feature type comparison**
- **Demo game player**

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Application Features

### Tab 1: Overview
- Project description and objectives
- Game rules explanation
- Learning method details
- Feature type descriptions

### Tab 2: Training
- **Configure environment**: Enable/disable cheating, set penalties
- **Configure agent**: Choose feature type, set hyperparameters
- **Real-time training**: Live progress bars and metrics
- **Dynamic visualizations**: Training curves update during learning
- **Training metrics**:
  - Episode rewards (with rolling average)
  - Win rate over time
  - Epsilon decay curve
  - Episode length distribution

### Tab 3: Analysis
- **Performance summary**: Final win rate, average reward, episode length
- **Q-value heatmaps**: Visualize learned action values
- **Policy visualization**: See optimal actions for each state
- **Feature comparison**: Compare different feature representations
- **Export trained agents**: Save models for later use

### Tab 4: Demo
- **Play demo games**: Watch trained agent play Blackjack
- **Step-by-step visualization**: See decision-making process
- **Game log**: Detailed action history

---

## 🧪 Experiments & Results

### Experiment 1: Feature Type Comparison
Compare learning performance across three feature representations:

**Configuration:**
- Episodes: 10,000
- Learning rate (α): 0.01
- Discount factor (γ): 0.99
- Initial epsilon (ε): 0.1
- Epsilon decay: 0.9995

**Expected Results:**
- **Binary features**: Fast initial learning, good final performance
- **Polynomial features**: Slower learning, better generalization
- **Combined features**: Best overall performance, highest win rate

### Experiment 2: Cheating Variant Analysis
Investigate the risk-reward trade-off of peeking:

**Configuration:**
- Peek penalty: -10
- Peek success rate: 0.7
- Compare with/without cheating enabled

**Questions to explore:**
- Does peeking improve win rate?
- How does penalty severity affect strategy?
- What's the optimal peek success rate threshold?

### Experiment 3: Hyperparameter Sensitivity
Analyze impact of learning rate and epsilon decay:

**Variables to test:**
- Learning rate: [0.001, 0.01, 0.1]
- Epsilon decay: [0.99, 0.995, 0.9995]

---

## 📈 Performance Metrics

### Key Metrics Tracked
1. **Win Rate**: Percentage of games won (rolling 100-episode window)
2. **Average Reward**: Mean reward over recent episodes
3. **Episode Length**: Average number of steps per episode
4. **Epsilon**: Current exploration rate
5. **Q-Value Distribution**: Range and variance of learned values

### Baseline Performance
A random policy achieves approximately **30-35% win rate**.

A well-trained LFA agent should achieve **42-45% win rate** (approaching optimal Blackjack strategy).

---

## 🔬 Technical Details

### State Representation
```python
state = (player_sum, dealer_card, usable_ace, peek_info)
```
- `player_sum`: Sum of player's cards (4-31)
- `dealer_card`: Dealer's visible card (1-10)
- `usable_ace`: Boolean indicating if ace counts as 11
- `peek_info`: Next card value if peeked (0 if not)

### Feature Extraction

#### Binary Features (55 dimensions)
- Player sum one-hot: 32 features
- Dealer card one-hot: 10 features
- Usable ace one-hot: 2 features
- Peek info one-hot: 11 features

#### Polynomial Features (15 dimensions)
- Normalized base features: [1, player_norm, dealer_norm, ace_norm, peek_norm]
- All degree-2 combinations

### Learning Algorithm: Semi-Gradient SARSA
```
Q(s,a) = w_a^T φ(s)

Update rule:
w_a ← w_a + α[r + γQ(s',a') - Q(s,a)]φ(s)
```

---

## 📁 Project Structure

```
project2_gambler/
├── app.py                  # Streamlit application
├── blackjack_env.py        # Blackjack environment
├── lfa_agent.py            # LFA agent implementation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🎓 Learning Objectives

1. **Understand Linear Function Approximation** in RL
2. **Explore feature engineering** and its impact on learning
3. **Implement Semi-Gradient SARSA** algorithm
4. **Analyze exploration-exploitation trade-offs**
5. **Compare different feature representations**
6. **Visualize learned policies and Q-values**

---

## 🔧 Customization

### Add New Feature Types
Extend `FeatureExtractor` class in `lfa_agent.py`:
```python
def _custom_features(self, player_sum, dealer_card, usable_ace, peek_info):
    # Your custom feature extraction logic
    return features
```

### Modify Game Rules
Edit `BlackjackEnv` class in `blackjack_env.py`:
- Change card values
- Adjust reward structure
- Add new actions

### Experiment with Algorithms
Replace SARSA with Q-Learning or Expected SARSA in `lfa_agent.py`

---

## 📊 Visualization Examples

### Training Progress Dashboard
- **Episode Rewards**: Track learning progress over time
- **Win Rate**: Rolling average of successful games
- **Epsilon Decay**: Exploration rate reduction
- **Episode Length**: Convergence indicator

### Q-Value Heatmaps
- Visualize learned action values for all state combinations
- Compare HIT vs STAND Q-values
- Identify optimal decision boundaries

### Policy Visualization
- Color-coded policy map (Red=Stand, Green=Hit)
- Shows learned strategy across state space
- Compare with optimal Blackjack basic strategy

---

## 🐛 Troubleshooting

### Issue: Slow training
**Solution**: Reduce number of episodes or increase update interval

### Issue: Poor convergence
**Solution**: Adjust learning rate (α) or epsilon decay rate

### Issue: Agent always hits/stands
**Solution**: Check feature extraction and weight initialization

---

## 📚 References

1. **Sutton & Barto**: Reinforcement Learning: An Introduction (Chapter 9: On-policy Prediction with Approximation)
2. **Blackjack Optimal Strategy**: Basic strategy charts for comparison
3. **Semi-Gradient Methods**: Convergence properties and limitations

---

## 🎯 Future Enhancements

- [ ] Add Deep Q-Network (DQN) comparison
- [ ] Implement eligibility traces (SARSA(λ))
- [ ] Multi-hand Blackjack support
- [ ] Tournament mode with multiple agents
- [ ] Advanced feature engineering (neural network features)
- [ ] Hyperparameter optimization (grid search, Bayesian optimization)

---

## 👥 Author

**Reinforcement Learning Course 2024-25**
- Prof. Nicolò Cesa-Bianchi
- Prof. Alfio Ferrara
- Course Assistants: Elisabetta Rocchetti, Luigi Foscari

---

## 📄 License

This project is part of the Reinforcement Learning course materials.

---

## 🙏 Acknowledgments

Special thanks to the RL course instructors and assistants for providing this challenging and educational project!

---

**Happy Learning! 🎓🎰**
