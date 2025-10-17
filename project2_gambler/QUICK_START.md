# Quick Start Guide - Project 2: The Gambler

## 🚀 Get Started in 3 Minutes!

---

## Step 1: Install Dependencies (30 seconds)

```bash
cd project2_gambler
pip install -r requirements.txt
```

**Required packages:**
- streamlit
- numpy
- pandas
- plotly
- scipy

---

## Step 2: Choose Your Experience

### Option A: Run Demo Training (Recommended First)

```bash
python demo_training.py
```

**What it does:**
- Trains 3 agents with different feature types (10,000 episodes each)
- Generates comprehensive visualizations
- Saves trained agents
- Shows test game results
- **Duration**: ~3 minutes

**Output files:**
- `training_results.png` - Comprehensive training dashboard
- `learned_policy.png` - Policy heatmap visualization
- `agent_binary.pkl` - Trained agent (binary features)
- `agent_polynomial.pkl` - Trained agent (polynomial features)
- `agent_combined.pkl` - Trained agent (combined features)

### Option B: Launch Interactive App

```bash
streamlit run app.py
```

**What you get:**
- Interactive web interface at `http://localhost:8501`
- Real-time training visualization
- Configure and train your own agents
- Analyze performance with interactive charts
- Play demo games with trained agents

---

## Step 3: Explore the Results

### View Generated Visualizations

1. **training_results.png** - Shows:
   - Episode rewards over time
   - Win rate evolution
   - Epsilon decay
   - Episode length distribution
   - Reward distribution
   - Performance summary table

2. **learned_policy.png** - Shows:
   - Learned policy heatmap (Hit vs Stand)
   - Q-value differences across states
   - Optimal action for each state

### Use the Streamlit App

**Tab 1: Overview**
- Read project description
- Understand the problem

**Tab 2: Training**
- Configure environment settings
- Set agent hyperparameters
- Train agents with live visualization
- Monitor progress in real-time

**Tab 3: Analysis**
- View performance metrics
- Explore Q-value heatmaps
- Visualize learned policies
- Compare feature types
- Download trained agents

**Tab 4: Demo**
- Watch agents play Blackjack
- See decision-making process
- Review game logs

---

## 📊 Expected Results

After running `demo_training.py`, you should see:

### Training Performance
- **Binary Features**: ~45% win rate
- **Polynomial Features**: ~44% win rate
- **Combined Features**: ~44% win rate

### Comparison to Baselines
- Random policy: ~30-35% win rate
- Optimal Blackjack: ~42-43% win rate
- **Our agents approach optimal!**

---

## 🎮 Quick Test

Want to quickly test a trained agent?

```python
from blackjack_env import BlackjackEnv
from lfa_agent import LFAAgent

# Load trained agent
agent = LFAAgent(n_actions=2, feature_type='binary')
agent.load('agent_binary.pkl')

# Play a game
env = BlackjackEnv()
state = env.reset()
done = False

while not done:
    action = agent.get_action(state, training=False)
    state, reward, done, info = env.step(action)
    print(f"Action: {'Hit' if action==1 else 'Stand'}, Reward: {reward}")

print(f"Game result: {'WIN' if reward > 0 else 'LOSS' if reward < 0 else 'DRAW'}")
```

---

## 🔧 Configuration Options

### Environment Settings
```python
env = BlackjackEnv(
    cheating_enabled=False,      # Enable peek action
    peek_penalty=-10,            # Penalty for getting caught
    peek_success_rate=0.7        # Probability of successful peek
)
```

### Agent Settings
```python
agent = LFAAgent(
    n_actions=2,                 # 2 for normal, 3 for cheating
    feature_type='binary',       # 'binary', 'polynomial', 'combined'
    alpha=0.01,                  # Learning rate
    gamma=0.99,                  # Discount factor
    epsilon=0.1,                 # Initial exploration rate
    epsilon_decay=0.9995,        # Epsilon decay rate
    epsilon_min=0.01             # Minimum epsilon
)
```

### Training Settings
```python
stats = train_agent(
    env=env,
    agent=agent,
    n_episodes=10000,            # Number of training episodes
    verbose=True                 # Print progress
)
```

---

## 📁 Project Files

```
project2_gambler/
├── blackjack_env.py          # Environment implementation
├── lfa_agent.py               # Agent with LFA
├── app.py                     # Streamlit application
├── demo_training.py           # Demo script
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── PROJECT_SUMMARY.md         # Project summary
└── QUICK_START.md            # This file
```

---

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Make sure you're in the `project2_gambler` directory and have installed requirements

### Issue: Streamlit won't start
**Solution**: 
```bash
pip install --upgrade streamlit
streamlit run app.py
```

### Issue: Matplotlib display issues
**Solution**: The demo saves images to files, no display needed

### Issue: Training too slow
**Solution**: Reduce `n_episodes` in demo_training.py (line 226)

---

## 💡 Tips

1. **First time?** Run `demo_training.py` first to see everything work
2. **Want to experiment?** Use the Streamlit app for interactive training
3. **Need custom features?** Modify `FeatureExtractor` class in `lfa_agent.py`
4. **Want different rules?** Edit `BlackjackEnv` class in `blackjack_env.py`

---

## 📚 Next Steps

1. ✅ Run demo training
2. ✅ View generated visualizations
3. ✅ Launch Streamlit app
4. ✅ Train agents with different configurations
5. ✅ Compare feature types
6. ✅ Experiment with cheating variant
7. ✅ Analyze learned policies
8. ✅ Read full documentation in README.md

---

## 🎯 Learning Goals

By completing this quick start, you will:
- ✅ Understand Linear Function Approximation in RL
- ✅ See Semi-Gradient SARSA in action
- ✅ Compare different feature representations
- ✅ Visualize learned policies and Q-values
- ✅ Analyze training dynamics

---

## 🆘 Need Help?

1. Check **README.md** for detailed documentation
2. Review **PROJECT_SUMMARY.md** for comprehensive analysis
3. Look at code comments for implementation details
4. Run `python blackjack_env.py` to test environment
5. Run `python lfa_agent.py` to test agent

---

## 🎉 Success Checklist

- [ ] Dependencies installed
- [ ] Demo training completed
- [ ] Visualizations generated
- [ ] Streamlit app launched
- [ ] Trained agents saved
- [ ] Results analyzed

---

**Ready to become a Blackjack master? Let's go! 🎰**

**Estimated time to complete**: 5-10 minutes
**Difficulty**: Beginner-friendly
**Prerequisites**: Basic Python knowledge

---

**Happy Learning! 🚀**
