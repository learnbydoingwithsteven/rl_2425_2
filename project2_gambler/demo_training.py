"""
Demo Training Script for Project 2: The Gambler
Demonstrates all features with comprehensive visualizations
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from blackjack_env import BlackjackEnv
from lfa_agent import LFAAgent, train_agent
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def create_comprehensive_visualization(stats_dict, save_path='training_results.png'):
    """Create comprehensive visualization of training results"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = ['#667eea', '#764ba2', '#f59e0b', '#10b981', '#ef4444']
    
    # Plot 1: Episode Rewards Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    for idx, (name, stats) in enumerate(stats_dict.items()):
        episodes = range(len(stats['episode_rewards']))
        # Rolling average
        window = 100
        rolling_avg = np.convolve(stats['episode_rewards'], 
                                  np.ones(window)/window, mode='valid')
        ax1.plot(rolling_avg, label=name, color=colors[idx % len(colors)], linewidth=2)
    
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Reward (100-episode window)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Progress: Episode Rewards', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Win Rate Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    win_rates = [stats['win_rate_window'][-1] for stats in stats_dict.values()]
    bars = ax2.bar(range(len(stats_dict)), win_rates, color=colors[:len(stats_dict)])
    ax2.set_xticks(range(len(stats_dict)))
    ax2.set_xticklabels(stats_dict.keys(), rotation=45, ha='right')
    ax2.set_ylabel('Final Win Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Final Performance', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 0.5])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, win_rates)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Win Rate Over Time
    ax3 = fig.add_subplot(gs[1, :2])
    for idx, (name, stats) in enumerate(stats_dict.items()):
        episodes = range(len(stats['win_rate_window']))
        ax3.plot(episodes, stats['win_rate_window'], 
                label=name, color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
    
    ax3.axhline(y=0.42, color='red', linestyle='--', linewidth=2, 
                label='Optimal Strategy (~42%)', alpha=0.7)
    ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Win Rate Evolution', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Epsilon Decay
    ax4 = fig.add_subplot(gs[1, 2])
    for idx, (name, stats) in enumerate(stats_dict.items()):
        episodes = range(len(stats['epsilon_history']))
        ax4.plot(episodes, stats['epsilon_history'], 
                label=name, color=colors[idx % len(colors)], linewidth=2)
    
    ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Epsilon', fontsize=12, fontweight='bold')
    ax4.set_title('Exploration Rate Decay', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Episode Length Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    for idx, (name, stats) in enumerate(stats_dict.items()):
        ax5.hist(stats['episode_lengths'], bins=20, alpha=0.5, 
                label=name, color=colors[idx % len(colors)])
    
    ax5.set_xlabel('Episode Length', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Reward Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    reward_data = []
    labels = []
    for name, stats in stats_dict.items():
        reward_data.append(stats['episode_rewards'][-1000:])  # Last 1000 episodes
        labels.append(name)
    
    bp = ax6.boxplot(reward_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax6.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax6.set_title('Reward Distribution (Last 1000 Episodes)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 7: Performance Summary Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    table_data = []
    for name, stats in stats_dict.items():
        final_wr = stats['win_rate_window'][-1]
        avg_reward = np.mean(stats['episode_rewards'][-1000:])
        avg_length = np.mean(stats['episode_lengths'][-1000:])
        table_data.append([name, f"{final_wr:.1%}", f"{avg_reward:.3f}", f"{avg_length:.1f}"])
    
    table = ax7.table(cellText=table_data,
                     colLabels=['Agent', 'Win Rate', 'Avg Reward', 'Avg Length'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            table[(i, j)].set_facecolor('#f0f2f6' if i % 2 == 0 else 'white')
    
    ax7.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Project 2: The Gambler - Comprehensive Training Results', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    return fig


def visualize_policy(agent, env, save_path='learned_policy.png'):
    """Visualize the learned policy as a heatmap"""
    player_sums = range(12, 22)
    dealer_cards = range(1, 11)
    
    policy = np.zeros((len(player_sums), len(dealer_cards)))
    q_diff = np.zeros((len(player_sums), len(dealer_cards)))
    
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            state = (player_sum, dealer_card, False, 0)
            q_values = agent.get_q_values(state)
            action = np.argmax(q_values)
            policy[i, j] = action
            q_diff[i, j] = q_values[1] - q_values[0]  # Hit - Stand
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Policy heatmap
    im1 = axes[0].imshow(policy, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(dealer_cards)))
    axes[0].set_xticklabels(dealer_cards)
    axes[0].set_yticks(range(len(player_sums)))
    axes[0].set_yticklabels(player_sums)
    axes[0].set_xlabel('Dealer Card', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Player Sum', fontsize=12, fontweight='bold')
    axes[0].set_title('Learned Policy (Red=Stand, Green=Hit)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(player_sums)):
        for j in range(len(dealer_cards)):
            text = 'H' if policy[i, j] == 1 else 'S'
            color = 'white' if policy[i, j] == 1 else 'black'
            axes[0].text(j, i, text, ha='center', va='center', 
                        color=color, fontweight='bold', fontsize=10)
    
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_ticks([0, 1])
    cbar1.set_ticklabels(['Stand', 'Hit'])
    
    # Q-value difference heatmap
    im2 = axes[1].imshow(q_diff, cmap='RdYlGn', aspect='auto')
    axes[1].set_xticks(range(len(dealer_cards)))
    axes[1].set_xticklabels(dealer_cards)
    axes[1].set_yticks(range(len(player_sums)))
    axes[1].set_yticklabels(player_sums)
    axes[1].set_xlabel('Dealer Card', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Player Sum', fontsize=12, fontweight='bold')
    axes[1].set_title('Q-Value Difference (Q(Hit) - Q(Stand))', fontsize=14, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Q-Value Difference', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Policy visualization saved to: {save_path}")
    
    return fig


def main():
    """Main demo script"""
    print("="*70)
    print("PROJECT 2: THE GAMBLER - COMPREHENSIVE DEMO")
    print("Reinforcement Learning with Linear Function Approximation")
    print("="*70)
    
    # Configuration
    n_episodes = 10000
    feature_types = ['binary', 'polynomial', 'combined']
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Episodes: {n_episodes}")
    print(f"   Feature Types: {', '.join(feature_types)}")
    print(f"   Algorithm: Semi-Gradient SARSA with LFA")
    
    # Train agents with different feature types
    stats_dict = {}
    agents_dict = {}
    
    for feature_type in feature_types:
        print(f"\n{'='*70}")
        print(f"🏋️  Training Agent with {feature_type.upper()} features...")
        print(f"{'='*70}")
        
        # Create environment and agent
        env = BlackjackEnv(cheating_enabled=False)
        agent = LFAAgent(
            n_actions=2,
            feature_type=feature_type,
            alpha=0.01,
            gamma=0.99,
            epsilon=0.1,
            epsilon_decay=0.9995,
            epsilon_min=0.01
        )
        
        print(f"   Feature dimension: {agent.feature_dim}")
        print(f"   Weight matrix shape: {agent.weights.shape}")
        
        # Train
        start_time = time.time()
        stats = train_agent(env, agent, n_episodes=n_episodes, verbose=True)
        training_time = time.time() - start_time
        
        # Store results
        stats_dict[feature_type] = stats
        agents_dict[feature_type] = agent
        
        # Print summary
        final_wr = stats['win_rate_window'][-1]
        avg_reward = np.mean(stats['episode_rewards'][-1000:])
        
        print(f"\n✅ Training Complete!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Final win rate: {final_wr:.1%}")
        print(f"   Average reward (last 1000): {avg_reward:.3f}")
        print(f"   Final epsilon: {agent.epsilon:.4f}")
    
    # Create comprehensive visualizations
    print(f"\n{'='*70}")
    print("📊 Creating Comprehensive Visualizations...")
    print(f"{'='*70}")
    
    create_comprehensive_visualization(stats_dict)
    
    # Visualize best agent's policy
    best_feature_type = max(stats_dict.keys(), 
                           key=lambda k: stats_dict[k]['win_rate_window'][-1])
    best_agent = agents_dict[best_feature_type]
    
    print(f"\n🏆 Best performing agent: {best_feature_type.upper()}")
    print(f"   Win rate: {stats_dict[best_feature_type]['win_rate_window'][-1]:.1%}")
    
    visualize_policy(best_agent, env)
    
    # Test games
    print(f"\n{'='*70}")
    print("🎮 Playing Test Games with Best Agent...")
    print(f"{'='*70}")
    
    test_results = []
    for game in range(10):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = best_agent.get_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        test_results.append(total_reward)
        result_str = "WIN" if total_reward > 0 else ("LOSS" if total_reward < 0 else "DRAW")
        print(f"   Game {game+1}: {result_str} (reward: {total_reward:+d}, steps: {steps})")
    
    test_win_rate = sum(1 for r in test_results if r > 0) / len(test_results)
    print(f"\n   Test win rate: {test_win_rate:.1%}")
    
    # Save agents
    print(f"\n{'='*70}")
    print("💾 Saving Trained Agents...")
    print(f"{'='*70}")
    
    for feature_type, agent in agents_dict.items():
        filename = f"agent_{feature_type}.pkl"
        agent.save(filename)
        print(f"   ✅ Saved: {filename}")
    
    print(f"\n{'='*70}")
    print("🎉 DEMO COMPLETE!")
    print(f"{'='*70}")
    print("\n📁 Generated Files:")
    print("   - training_results.png (comprehensive visualization)")
    print("   - learned_policy.png (policy heatmap)")
    print("   - agent_binary.pkl (trained agent)")
    print("   - agent_polynomial.pkl (trained agent)")
    print("   - agent_combined.pkl (trained agent)")
    print("\n🚀 To run the interactive Streamlit app:")
    print("   streamlit run app.py")
    print("="*70)


if __name__ == "__main__":
    main()
