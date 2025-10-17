"""
Project 2: The Gambler - Interactive Streamlit Application
Reinforcement Learning with Linear Function Approximation for Blackjack
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from blackjack_env import BlackjackEnv
from lfa_agent import LFAAgent, train_agent
import os


# Page configuration
st.set_page_config(
    page_title="Project 2: The Gambler",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'trained_agents' not in st.session_state:
        st.session_state.trained_agents = {}
    if 'training_stats' not in st.session_state:
        st.session_state.training_stats = {}
    if 'current_training' not in st.session_state:
        st.session_state.current_training = False


def create_training_progress_chart(stats, window=100):
    """Create real-time training progress visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Win Rate (Rolling)', 
                       'Epsilon Decay', 'Episode Length'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    episodes = list(range(len(stats['episode_rewards'])))
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=stats['episode_rewards'], 
                  mode='lines', name='Reward',
                  line=dict(color='#667eea', width=1),
                  opacity=0.6),
        row=1, col=1
    )
    
    # Rolling average
    if len(stats['episode_rewards']) >= window:
        rolling_avg = pd.Series(stats['episode_rewards']).rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(x=episodes, y=rolling_avg,
                      mode='lines', name=f'{window}-Episode Avg',
                      line=dict(color='#764ba2', width=3)),
            row=1, col=1
        )
    
    # Win rate
    fig.add_trace(
        go.Scatter(x=episodes, y=stats['win_rate_window'],
                  mode='lines', name='Win Rate',
                  line=dict(color='#10b981', width=2),
                  fill='tozeroy'),
        row=1, col=2
    )
    
    # Epsilon decay
    fig.add_trace(
        go.Scatter(x=episodes, y=stats['epsilon_history'],
                  mode='lines', name='Epsilon',
                  line=dict(color='#f59e0b', width=2)),
        row=2, col=1
    )
    
    # Episode length
    fig.add_trace(
        go.Scatter(x=episodes, y=stats['episode_lengths'],
                  mode='lines', name='Length',
                  line=dict(color='#ef4444', width=1),
                  opacity=0.6),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate", row=1, col=2)
    fig.update_yaxes(title_text="Epsilon", row=2, col=1)
    fig.update_yaxes(title_text="Steps", row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white',
        title_text="Training Progress Dashboard"
    )
    
    return fig


def create_feature_comparison_chart(agents_stats):
    """Compare performance across different feature types"""
    fig = go.Figure()
    
    for feature_type, stats in agents_stats.items():
        if stats:
            episodes = list(range(len(stats['win_rate_window'])))
            fig.add_trace(
                go.Scatter(x=episodes, y=stats['win_rate_window'],
                          mode='lines', name=feature_type,
                          line=dict(width=2))
            )
    
    fig.update_layout(
        title="Feature Type Comparison: Win Rate Over Time",
        xaxis_title="Episode",
        yaxis_title="Win Rate",
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_q_value_heatmap(agent, env):
    """Create heatmap of Q-values for different states"""
    player_sums = range(12, 22)
    dealer_cards = range(1, 11)
    
    q_values_hit = np.zeros((len(player_sums), len(dealer_cards)))
    q_values_stand = np.zeros((len(player_sums), len(dealer_cards)))
    
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            state = (player_sum, dealer_card, False, 0)
            q_vals = agent.get_q_values(state)
            q_values_hit[i, j] = q_vals[1]  # Hit
            q_values_stand[i, j] = q_vals[0]  # Stand
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Q-Values: HIT', 'Q-Values: STAND'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    fig.add_trace(
        go.Heatmap(z=q_values_hit, x=dealer_cards, y=player_sums,
                  colorscale='RdYlGn', zmid=0),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(z=q_values_stand, x=dealer_cards, y=player_sums,
                  colorscale='RdYlGn', zmid=0),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Dealer Card", row=1, col=1)
    fig.update_xaxes(title_text="Dealer Card", row=1, col=2)
    fig.update_yaxes(title_text="Player Sum", row=1, col=1)
    fig.update_yaxes(title_text="Player Sum", row=1, col=2)
    
    fig.update_layout(
        height=400,
        title_text="Learned Q-Values Visualization"
    )
    
    return fig


def create_policy_visualization(agent, env):
    """Visualize the learned policy"""
    player_sums = range(12, 22)
    dealer_cards = range(1, 11)
    
    policy = np.zeros((len(player_sums), len(dealer_cards)))
    
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            state = (player_sum, dealer_card, False, 0)
            action = agent.get_action(state, training=False)
            policy[i, j] = action
    
    fig = go.Figure(data=go.Heatmap(
        z=policy,
        x=dealer_cards,
        y=player_sums,
        colorscale=[[0, '#ef4444'], [1, '#10b981']],
        colorbar=dict(
            title="Action",
            tickvals=[0, 1],
            ticktext=['Stand', 'Hit']
        )
    ))
    
    fig.update_layout(
        title="Learned Policy (Red=Stand, Green=Hit)",
        xaxis_title="Dealer Card",
        yaxis_title="Player Sum",
        height=400
    )
    
    return fig


def play_demo_game(env, agent):
    """Play a demonstration game with visualization"""
    st.markdown('<p class="sub-header">🎮 Demo Game</p>', unsafe_allow_html=True)
    
    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    game_log = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Player's Hand:**")
        player_display = st.empty()
        
    with col2:
        st.markdown("**Dealer's Visible Card:**")
        dealer_display = st.empty()
    
    action_display = st.empty()
    reward_display = st.empty()
    
    while not done and step < 10:
        # Display current state
        player_sum, dealer_card, usable_ace, _ = state
        player_display.markdown(f"### {env.player_cards} = {player_sum} {'(Usable Ace)' if usable_ace else ''}")
        dealer_display.markdown(f"### {dealer_card}")
        
        # Get action
        action = agent.get_action(state, training=False)
        action_name = ['Stand', 'Hit', 'Peek'][action]
        
        action_display.info(f"**Action:** {action_name}")
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        game_log.append({
            'Step': step + 1,
            'Player Sum': player_sum,
            'Dealer Card': dealer_card,
            'Action': action_name,
            'Reward': reward
        })
        
        state = next_state
        step += 1
        
        time.sleep(0.5)
    
    # Final result
    if total_reward > 0:
        reward_display.success(f"🎉 **WIN!** Total Reward: {total_reward}")
    elif total_reward < 0:
        reward_display.error(f"😞 **LOSS!** Total Reward: {total_reward}")
    else:
        reward_display.warning(f"🤝 **DRAW!** Total Reward: {total_reward}")
    
    # Show game log
    st.dataframe(pd.DataFrame(game_log), use_container_width=True)


def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎰 Project 2: The Gambler</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Reinforcement Learning with Linear Function Approximation for Blackjack</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Environment settings
    st.sidebar.markdown("### 🎮 Environment")
    cheating_enabled = st.sidebar.checkbox("Enable Cheating Variant", value=False)
    
    if cheating_enabled:
        peek_penalty = st.sidebar.slider("Peek Penalty", -20, -5, -10)
        peek_success_rate = st.sidebar.slider("Peek Success Rate", 0.5, 1.0, 0.7)
    else:
        peek_penalty = -10
        peek_success_rate = 0.7
    
    # Agent settings
    st.sidebar.markdown("### 🤖 Agent Configuration")
    feature_type = st.sidebar.selectbox(
        "Feature Type",
        ["binary", "polynomial", "combined"],
        help="Type of feature representation"
    )
    
    alpha = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001, format="%.3f")
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.9, 1.0, 0.99, 0.01)
    epsilon = st.sidebar.slider("Initial Epsilon (ε)", 0.0, 1.0, 0.1, 0.05)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.99, 1.0, 0.9995, 0.0001, format="%.4f")
    
    # Training settings
    st.sidebar.markdown("### 🏋️ Training")
    n_episodes = st.sidebar.number_input("Number of Episodes", 1000, 50000, 10000, 1000)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Overview", "🏋️ Training", "📊 Analysis", "🎮 Demo"])
    
    with tab1:
        st.markdown('<p class="sub-header">Project Overview</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Objective</h3>
            <p>Train a reinforcement learning agent to play simplified Blackjack using 
            <strong>Linear Function Approximation (LFA)</strong> with different feature representations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <h3>🎲 Game Rules</h3>
            <ul>
                <li>Player starts with 2 cards</li>
                <li>Dealer shows 1 card</li>
                <li>Actions: <strong>Hit</strong> (draw card) or <strong>Stand</strong> (stop)</li>
                <li>Goal: Get closer to 21 than dealer without busting</li>
                <li>Aces can be 1 or 11</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h3>🧠 Learning Method</h3>
            <p><strong>Semi-Gradient SARSA</strong> with Linear Function Approximation</p>
            <p>Q(s,a) = w<sub>a</sub><sup>T</sup> φ(s)</p>
            <p>Update rule: w ← w + α[r + γQ(s',a') - Q(s,a)]∇Q(s,a)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <h3>🎨 Feature Types</h3>
            <ul>
                <li><strong>Binary:</strong> One-hot encoding of state components</li>
                <li><strong>Polynomial:</strong> Degree-2 polynomial features</li>
                <li><strong>Combined:</strong> Both binary and polynomial</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if cheating_enabled:
            st.warning("⚠️ **Cheating Variant Enabled**: Agent can peek at next card with risk of penalty!")
    
    with tab2:
        st.markdown('<p class="sub-header">Training Dashboard</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Start training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state.current_training = True
            
            # Create environment and agent
            env = BlackjackEnv(
                cheating_enabled=cheating_enabled,
                peek_penalty=peek_penalty,
                peek_success_rate=peek_success_rate
            )
            
            n_actions = env.get_action_space_size()
            agent = LFAAgent(
                n_actions=n_actions,
                feature_type=feature_type,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay
            )
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            
            # Metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            metric_episode = metrics_col1.empty()
            metric_reward = metrics_col2.empty()
            metric_winrate = metrics_col3.empty()
            metric_epsilon = metrics_col4.empty()
            
            # Training loop with visualization
            stats = {
                'episode_rewards': [],
                'episode_lengths': [],
                'win_rate_window': [],
                'epsilon_history': []
            }
            
            recent_results = []
            window_size = 100
            update_interval = max(1, n_episodes // 100)
            
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
                        agent.update(state, action, reward, next_state, 0, done)
                
                agent.decay_epsilon()
                
                # Track statistics
                stats['episode_rewards'].append(episode_reward)
                stats['episode_lengths'].append(episode_length)
                stats['epsilon_history'].append(agent.epsilon)
                
                recent_results.append(1 if episode_reward > 0 else 0)
                if len(recent_results) > window_size:
                    recent_results.pop(0)
                
                win_rate = np.mean(recent_results) if recent_results else 0
                stats['win_rate_window'].append(win_rate)
                
                # Update UI
                if episode % update_interval == 0 or episode == n_episodes - 1:
                    progress = (episode + 1) / n_episodes
                    progress_bar.progress(progress)
                    status_text.text(f"Training: Episode {episode + 1}/{n_episodes}")
                    
                    # Update metrics
                    metric_episode.metric("Episode", f"{episode + 1}/{n_episodes}")
                    metric_reward.metric("Avg Reward", f"{np.mean(stats['episode_rewards'][-100:]):.3f}")
                    metric_winrate.metric("Win Rate", f"{win_rate:.1%}")
                    metric_epsilon.metric("Epsilon", f"{agent.epsilon:.4f}")
                    
                    # Update chart
                    if episode > 100:
                        fig = create_training_progress_chart(stats)
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Save results
            agent_key = f"{feature_type}_cheat{cheating_enabled}"
            st.session_state.trained_agents[agent_key] = agent
            st.session_state.training_stats[agent_key] = stats
            
            st.success(f"✅ Training completed! Final win rate: {stats['win_rate_window'][-1]:.1%}")
            st.session_state.current_training = False
    
    with tab3:
        st.markdown('<p class="sub-header">Performance Analysis</p>', unsafe_allow_html=True)
        
        if st.session_state.trained_agents:
            # Select agent for analysis
            agent_keys = list(st.session_state.trained_agents.keys())
            selected_agent_key = st.selectbox("Select Agent", agent_keys)
            
            agent = st.session_state.trained_agents[selected_agent_key]
            stats = st.session_state.training_stats[selected_agent_key]
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            final_winrate = stats['win_rate_window'][-1]
            avg_reward = np.mean(stats['episode_rewards'][-1000:])
            avg_length = np.mean(stats['episode_lengths'][-1000:])
            final_epsilon = stats['epsilon_history'][-1]
            
            col1.markdown(f'<div class="metric-card"><h3>{final_winrate:.1%}</h3><p>Final Win Rate</p></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-card"><h3>{avg_reward:.3f}</h3><p>Avg Reward</p></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-card"><h3>{avg_length:.1f}</h3><p>Avg Episode Length</p></div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="metric-card"><h3>{final_epsilon:.4f}</h3><p>Final Epsilon</p></div>', unsafe_allow_html=True)
            
            # Visualizations
            st.plotly_chart(create_training_progress_chart(stats), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                env = BlackjackEnv(cheating_enabled='cheat' in selected_agent_key)
                st.plotly_chart(create_q_value_heatmap(agent, env), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_policy_visualization(agent, env), use_container_width=True)
            
            # Feature comparison if multiple agents trained
            if len(st.session_state.trained_agents) > 1:
                st.markdown('<p class="sub-header">Feature Type Comparison</p>', unsafe_allow_html=True)
                st.plotly_chart(create_feature_comparison_chart(st.session_state.training_stats), use_container_width=True)
            
            # Download results
            if st.button("💾 Download Agent"):
                agent.save(f"agent_{selected_agent_key}.pkl")
                st.success(f"Agent saved as agent_{selected_agent_key}.pkl")
        else:
            st.info("👈 Train an agent first to see analysis!")
    
    with tab4:
        st.markdown('<p class="sub-header">Play Demo Games</p>', unsafe_allow_html=True)
        
        if st.session_state.trained_agents:
            agent_keys = list(st.session_state.trained_agents.keys())
            selected_agent_key = st.selectbox("Select Agent for Demo", agent_keys, key="demo_agent")
            
            if st.button("🎲 Play Game", type="primary"):
                agent = st.session_state.trained_agents[selected_agent_key]
                env = BlackjackEnv(cheating_enabled='cheat' in selected_agent_key)
                play_demo_game(env, agent)
        else:
            st.info("👈 Train an agent first to play demo games!")


if __name__ == "__main__":
    main()
