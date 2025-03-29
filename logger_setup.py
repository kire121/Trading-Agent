import logging
import os
import numpy as np
from datetime import datetime

# Skapa alla nödvändiga kataloger
os.makedirs("statistics", exist_ok=True)
os.makedirs("rewards", exist_ok=True)
os.makedirs("debug", exist_ok=True)

# Timestamp för unika filnamn
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Standard logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading_agent.log"), logging.StreamHandler()]
)
logger = logging.getLogger("trading_agent")

# Lägg till en separat filhanterare för debug-meddelanden
debug_handler = logging.FileHandler(f"debug/trading_agent_debug_{timestamp}.log")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(debug_handler)

# Skapa en separat reward-logger för detaljerad loggning av reward
reward_logger = logging.getLogger("reward_logger")
reward_logger.setLevel(logging.DEBUG)
reward_logger.propagate = False  # Förhindra att meddelanden går till root logger
reward_file_handler = logging.FileHandler(f"rewards/reward_log_{timestamp}.txt")
reward_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
reward_logger.addHandler(reward_file_handler)

# Create a dedicated statistics logger
stats_logger = logging.getLogger("trading_stats")
stats_logger.setLevel(logging.INFO)
stats_logger.propagate = False  # Don't propagate to root logger
stats_file_handler = logging.FileHandler(f"statistics/trading_stats_{timestamp}.txt")
stats_file_handler.setFormatter(logging.Formatter('%(message)s'))  # Simple format, just the message
stats_logger.addHandler(stats_file_handler)

# Nya hjälpfunktioner för felsökning
def log_reward_breakdown(episode, step, reward_components):
    """
    Loggar detaljerad information om reward-komponenter för felsökning.
    
    Args:
        episode: Aktuell episod
        step: Aktuellt steg
        reward_components: Dictionary med reward-komponenter och deras värden
    """
    if not reward_components:
        return
        
    try:
        # Skapa en sträng med alla komponenter
        components_str = ", ".join([f"{k}={v:.4f}" for k, v in reward_components.items()])
        
        # Logga med varningsnivå om belöningen är extrem
        total = reward_components.get('total', 0.0)
        if abs(total) > 5.0:  # Minskat från 10.0 till 5.0 för den nya reward-skalan
            reward_logger.warning(f"EP {episode}, STEP {step}: STOR REWARD {total:.2f} - {components_str}")
            logger.warning(f"Stor reward {total:.2f} i episod {episode}, steg {step}")
        else:
            # Använd debug-nivå för normala rewards (mindre output)
            reward_logger.debug(f"EP {episode}, STEP {step}: Reward {total:.2f} - {components_str}")
            
    except Exception as e:
        logger.error(f"Fel vid loggning av reward breakdown: {e}")

def log_warning_value(name, value, threshold=5.0):  # Minskat threshold från 100.0 till 5.0
    """
    Loggar varning för extrema värden.
    
    Args:
        name: Namn på värdet
        value: Värdet som ska kontrolleras
        threshold: Tröskelvärde för att utlösa varning
    """
    if abs(value) > threshold:
        logger.warning(f"EXTREMT VÄRDE: {name}={value:.2f} överskrider tröskelvärdet {threshold}")
        return True
    return False

def log_episode_stats(episode, env, agent, reward, portfolio_value, sharpe, sortino, additional_metrics=None):
    """
    Log statistics for a single episode.
    
    Args:
        episode: Current episode number
        env: Trading environment
        agent: DQN agent
        reward: Episode reward
        portfolio_value: Final portfolio value
        sharpe: Sharpe ratio
        sortino: Sortino ratio
        additional_metrics: Dictionary of additional metrics to log
    """
    # Check for extreme reward values
    if abs(reward) > 1000:
        logger.warning(f"Extremt reward-värde i episod {episode}: {reward:.2f}")
    
    # Start with basic metrics
    stats_logger.info(f"\n====== EPISODE {episode} STATISTICS ======")
    stats_logger.info(f"Reward: {reward:.2f}")
    stats_logger.info(f"Portfolio Value: {portfolio_value:.2f}")
    stats_logger.info(f"Cash: {env.cash:.2f}")
    
    # Calculate returns
    returns_pct = ((portfolio_value - env.initial_cash) / env.initial_cash) * 100
    stats_logger.info(f"Returns: {returns_pct:.2f}%")
    
    # Buy & Hold comparison
    if len(env.prices) > 1:
        buy_hold_return = ((env.prices[-1] - env.prices[0]) / env.prices[0]) * 100
        stats_logger.info(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        stats_logger.info(f"Alpha: {returns_pct - buy_hold_return:.2f}%")
    
    # Risk metrics
    stats_logger.info(f"Sharpe Ratio: {sharpe:.4f}")
    stats_logger.info(f"Sortino Ratio: {sortino:.4f}")
    
    # Agent metrics
    stats_logger.info(f"Epsilon: {agent.epsilon:.6f}")
    stats_logger.info(f"Loss: {agent.loss_history[-1]:.6f}" if agent.loss_history else "Loss: N/A")
    
    # Kelly metrics if available
    if hasattr(env, 'use_kelly') and env.use_kelly and hasattr(env, 'kelly_sizer'):
        kelly_fraction = env.kelly_sizer.calculate_kelly_fraction()
        win_rate = env.kelly_sizer.win_rate
        win_loss_ratio = env.kelly_sizer.win_loss_ratio
        stats_logger.info(f"Kelly Fraction: {kelly_fraction:.4f}")
        stats_logger.info(f"Win Rate: {win_rate:.4f}")
        stats_logger.info(f"Win/Loss Ratio: {win_loss_ratio:.4f}")
    
    # Trade statistics
    if hasattr(env, 'completed_trades') and env.completed_trades:
        total_trades = len(env.completed_trades)
        profitable_trades = sum(1 for t in env.completed_trades if t.get('profit_loss', 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average profit/loss
        profits = [t.get('profit_loss', 0) * 100 for t in env.completed_trades]
        avg_profit = sum(profits) / len(profits) if profits else 0
        
        # Stop-loss and take-profit statistics
        sl_exits = sum(1 for t in env.completed_trades if t.get('is_stop_loss_exit', False))
        tp_exits = sum(1 for t in env.completed_trades if t.get('is_take_profit_exit', False))
        manual_exits = total_trades - sl_exits - tp_exits
        
        stats_logger.info(f"Total Trades: {total_trades}")
        stats_logger.info(f"Win Rate: {win_rate:.4f} ({profitable_trades}/{total_trades})")
        stats_logger.info(f"Average P/L: {avg_profit:.2f}%")
        stats_logger.info(f"Stop-Loss Exits: {sl_exits}")
        stats_logger.info(f"Take-Profit Exits: {tp_exits}")
        stats_logger.info(f"Manual Exits: {manual_exits}")
    
    # Pattern recognition statistics if available
    if hasattr(env, 'successful_trade_patterns'):
        stats_logger.info(f"Saved Patterns: {len(env.successful_trade_patterns)}")
    
    # Additional metrics if provided
    if additional_metrics:
        stats_logger.info("\nAdditional Metrics:")
        
        # Gruppera reward-komponenter separat om de finns
        reward_components = {}
        other_metrics = {}
        
        for key, value in additional_metrics.items():
            if key in ['direct_return', 'sharpe_change', 'drawdown_penalty', 'regime_reward']:
                reward_components[key] = value
            else:
                other_metrics[key] = value
        
        # Logga vanliga metrics först
        for key, value in other_metrics.items():
            stats_logger.info(f"{key}: {value}")
        
        # Visa reward-komponenter i sin egen sektion om de finns
        if reward_components:
            stats_logger.info("\nReward Components:")
            total_contribution = 0.0
            for key, value in reward_components.items():
                # Beräkna viktad påverkan baserat på nya vikter
                weight = 1.0  # Standard
                if key == 'direct_return':
                    weight = 0.7
                elif key == 'sharpe_change':
                    weight = 5.0
                elif key == 'drawdown_penalty':
                    weight = 0.2
                elif key == 'regime_reward':
                    weight = 1.0
                    
                contribution = value * weight
                total_contribution += contribution
                stats_logger.info(f"{key}: {value:.4f} (viktat: {contribution:.4f})")
                
            stats_logger.info(f"Summa viktade komponenter: {total_contribution:.4f}")
    
    stats_logger.info("=" * 40)

def log_training_summary(episodes, env, agent, reward_history, sharpe_history, sortino_history):
    """
    Log a summary of the entire training session.
    
    Args:
        episodes: Total number of episodes
        env: Trading environment
        agent: DQN agent
        reward_history: List of rewards from all episodes
        sharpe_history: List of Sharpe ratios
        sortino_history: List of Sortino ratios
    """
    stats_logger.info("\n\n")
    stats_logger.info("=" * 50)
    stats_logger.info("       TRAINING SESSION SUMMARY       ")
    stats_logger.info("=" * 50)
    stats_logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats_logger.info(f"Model Type: {agent.model_type}")
    stats_logger.info(f"Episodes: {episodes}")
    
    # Overall performance
    stats_logger.info("\n--- OVERALL PERFORMANCE ---")
    stats_logger.info(f"Final Portfolio Value: {env.portfolio_value():.2f}")
    returns_pct = ((env.portfolio_value() - env.initial_cash) / env.initial_cash) * 100
    stats_logger.info(f"Total Return: {returns_pct:.2f}%")
    
    # Calculate buy & hold return
    if len(env.prices) > 1:
        buy_hold_return = ((env.prices[-1] - env.prices[0]) / env.prices[0]) * 100
        stats_logger.info(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        stats_logger.info(f"Alpha: {returns_pct - buy_hold_return:.2f}%")
    
    # Reward statistics med utökad analys
    stats_logger.info("\n--- REWARD STATISTICS ---")
    stats_logger.info(f"Mean Reward: {np.mean(reward_history):.2f}")
    stats_logger.info(f"Median Reward: {np.median(reward_history):.2f}")
    stats_logger.info(f"Std Dev Reward: {np.std(reward_history):.2f}")
    stats_logger.info(f"Min Reward: {np.min(reward_history):.2f}")
    stats_logger.info(f"Max Reward: {np.max(reward_history):.2f}")
    
    # Ytterligare reward-statistik för att identifiera extremvärden
    rewards_np = np.array(reward_history)
    if len(rewards_np) > 0:
        # Hitta extrema värden (över 3 standardavvikelser)
        mean_reward = np.mean(rewards_np)
        std_reward = np.std(rewards_np)
        threshold = mean_reward + 3 * std_reward
        extreme_values = rewards_np[rewards_np > threshold]
        
        if len(extreme_values) > 0:
            stats_logger.info(f"\nExtremvärden i rewards (>{threshold:.2f}):")
            stats_logger.info(f"Antal extremvärden: {len(extreme_values)}")
            stats_logger.info(f"Proportion av alla rewards: {len(extreme_values)/len(rewards_np)*100:.2f}%")
            stats_logger.info(f"Medel av extremvärden: {np.mean(extreme_values):.2f}")
            stats_logger.info(f"Max extremvärde: {np.max(extreme_values):.2f}")
    
    # Calculate trend in last 20% of episodes
    if len(reward_history) >= 5:
        cutoff = int(len(reward_history) * 0.8)
        early_rewards = reward_history[:cutoff]
        late_rewards = reward_history[cutoff:]
        early_mean = np.mean(early_rewards)
        late_mean = np.mean(late_rewards)
        stats_logger.info(f"Early vs Late Performance Change: {((late_mean - early_mean) / abs(early_mean if early_mean != 0 else 1)) * 100:.2f}%")
    
    # Risk metrics
    stats_logger.info("\n--- RISK METRICS ---")
    stats_logger.info(f"Mean Sharpe: {np.mean(sharpe_history):.4f}")
    stats_logger.info(f"Final Sharpe: {sharpe_history[-1] if sharpe_history else 'N/A':.4f}")
    stats_logger.info(f"Mean Sortino: {np.mean(sortino_history):.4f}")
    stats_logger.info(f"Final Sortino: {sortino_history[-1] if sortino_history else 'N/A':.4f}")
    
    # Calculate maximum drawdown
    if len(reward_history) > 0:
        cumulative_returns = np.cumsum(reward_history)
        max_returns = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - max_returns) / (max_returns + env.initial_cash)
        max_drawdown = abs(min(drawdowns)) * 100  # Convert to percentage
        stats_logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Agent statistics
    stats_logger.info("\n--- AGENT STATISTICS ---")
    stats_logger.info(f"Final Epsilon: {agent.epsilon:.6f}")
    stats_logger.info(f"Gamma: {agent.gamma:.4f}")
    stats_logger.info(f"Learning Rate: {agent.learning_rate:.6f}")
    stats_logger.info(f"Memory Size: {len(agent.memory)}")
    
    # Model parameters
    stats_logger.info(f"Use Kelly: {getattr(agent, 'use_kelly', False)}")
    stats_logger.info(f"Use Risk Management: {getattr(agent, 'use_risk_management', False)}")
    
    # Trade statistics
    if hasattr(env, 'completed_trades') and env.completed_trades:
        stats_logger.info("\n--- TRADE STATISTICS ---")
        total_trades = len(env.completed_trades)
        profitable_trades = sum(1 for t in env.completed_trades if t.get('profit_loss', 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profits and losses
        profits = [t.get('profit_loss', 0) * 100 for t in env.completed_trades if t.get('profit_loss', 0) > 0]
        losses = [t.get('profit_loss', 0) * 100 for t in env.completed_trades if t.get('profit_loss', 0) <= 0]
        
        # Kelly-specific trades
        kelly_trades = [t for t in env.completed_trades if t.get('entry_action', 0) == 3]  # Action 3 is Kelly buy
        kelly_win_rate = sum(1 for t in kelly_trades if t.get('profit_loss', 0) > 0) / len(kelly_trades) if kelly_trades else 0
        
        # Calculate profit factor
        gross_profit = sum(profits) if profits else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Stop-loss and take-profit statistics
        sl_exits = sum(1 for t in env.completed_trades if t.get('is_stop_loss_exit', False))
        tp_exits = sum(1 for t in env.completed_trades if t.get('is_take_profit_exit', False))
        manual_exits = total_trades - sl_exits - tp_exits
        
        # Calculate average P&L by exit type
        sl_pl = [t.get('profit_loss', 0) * 100 for t in env.completed_trades if t.get('is_stop_loss_exit', False)]
        tp_pl = [t.get('profit_loss', 0) * 100 for t in env.completed_trades if t.get('is_take_profit_exit', False)]
        manual_pl = [t.get('profit_loss', 0) * 100 for t in env.completed_trades 
                     if not t.get('is_stop_loss_exit', False) and not t.get('is_take_profit_exit', False)]
        
        # Log trade statistics
        stats_logger.info(f"Total Trades: {total_trades}")
        stats_logger.info(f"Win Rate: {win_rate:.4f} ({profitable_trades}/{total_trades})")
        stats_logger.info(f"Profit Factor: {profit_factor:.4f}")
        stats_logger.info(f"Average Profit: {np.mean(profits) if profits else 0:.2f}%")
        stats_logger.info(f"Average Loss: {np.mean(losses) if losses else 0:.2f}%")
        stats_logger.info(f"Best Trade: {max(profits) if profits else 0:.2f}%")
        stats_logger.info(f"Worst Trade: {min(losses) if losses else 0:.2f}%")
        
        # Kelly-specific statistics
        if kelly_trades:
            stats_logger.info(f"\n--- KELLY TRADE STATISTICS ---")
            stats_logger.info(f"Kelly Trades: {len(kelly_trades)}")
            stats_logger.info(f"Kelly Win Rate: {kelly_win_rate:.4f}")
            kelly_profits = [t.get('profit_loss', 0) * 100 for t in kelly_trades if t.get('profit_loss', 0) > 0]
            kelly_losses = [t.get('profit_loss', 0) * 100 for t in kelly_trades if t.get('profit_loss', 0) <= 0]
            stats_logger.info(f"Kelly Average Profit: {np.mean(kelly_profits) if kelly_profits else 0:.2f}%")
            stats_logger.info(f"Kelly Average Loss: {np.mean(kelly_losses) if kelly_losses else 0:.2f}%")
        
        # Risk management statistics
        stats_logger.info(f"\n--- RISK MANAGEMENT STATISTICS ---")
        stats_logger.info(f"Stop-Loss Exits: {sl_exits} ({sl_exits/total_trades*100 if total_trades > 0 else 0:.2f}%)")
        stats_logger.info(f"Take-Profit Exits: {tp_exits} ({tp_exits/total_trades*100 if total_trades > 0 else 0:.2f}%)")
        stats_logger.info(f"Manual Exits: {manual_exits} ({manual_exits/total_trades*100 if total_trades > 0 else 0:.2f}%)")
        stats_logger.info(f"Avg P/L on Stop-Loss: {np.mean(sl_pl) if sl_pl else 0:.2f}%")
        stats_logger.info(f"Avg P/L on Take-Profit: {np.mean(tp_pl) if tp_pl else 0:.2f}%")
        stats_logger.info(f"Avg P/L on Manual Exits: {np.mean(manual_pl) if manual_pl else 0:.2f}%")
    
    # Pattern recognition statistics
    if hasattr(env, 'successful_trade_patterns') and env.successful_trade_patterns:
        stats_logger.info("\n--- PATTERN RECOGNITION STATISTICS ---")
        stats_logger.info(f"Total Saved Patterns: {len(env.successful_trade_patterns)}")
        
        # Analyze pattern profitability
        pattern_profits = [p.get('profit', 0) * 100 for p in env.successful_trade_patterns]
        stats_logger.info(f"Average Pattern Profit: {np.mean(pattern_profits):.2f}%")
        stats_logger.info(f"Best Pattern Profit: {max(pattern_profits):.2f}%")
        
        # Action distribution in patterns
        action_counts = {}
        for p in env.successful_trade_patterns:
            action = p.get('action', 0)
            action_counts[action] = action_counts.get(action, 0) + 1
        
        stats_logger.info("Pattern Action Distribution:")
        for action, count in action_counts.items():
            action_name = "Buy" if action == 1 else "Kelly Buy" if action == 3 else f"Action {action}"
            stats_logger.info(f"  {action_name}: {count} patterns ({count/len(env.successful_trade_patterns)*100:.2f}%)")
    
    stats_logger.info("\n--- RECOMMENDATIONS ---")
    
    # Simple recommendations based on results
    if hasattr(env, 'completed_trades') and env.completed_trades:
        sl_effectiveness = np.mean(sl_pl) if sl_pl else 0
        tp_effectiveness = np.mean(tp_pl) if tp_pl else 0
        
        if sl_effectiveness < -1.0:
            stats_logger.info("• Consider tightening stop-loss levels (losses are too large)")
        elif sl_exits < total_trades * 0.1 and win_rate < 0.5:
            stats_logger.info("• Consider more aggressive stop-loss strategy to limit losses")
        
        if tp_effectiveness > 2.0 and tp_exits < total_trades * 0.2:
            stats_logger.info("• Take-profit levels may be too conservative, consider looser take-profit targets")
            
        if win_rate < 0.4:
            stats_logger.info("• Overall win rate is low, consider more conservative position sizing")
        
        # Kelly recommendations
        if hasattr(agent, 'use_kelly') and agent.use_kelly and kelly_trades:
            if len(kelly_trades) > 5 and kelly_win_rate > win_rate + 0.1:
                stats_logger.info("• Kelly-sized trades are outperforming regular trades, consider using exclusively Kelly sizing")
            elif len(kelly_trades) > 5 and kelly_win_rate < win_rate - 0.1:
                stats_logger.info("• Kelly-sized trades are underperforming, review Kelly implementation")
    
    # Nya rekommendationer relaterade till reward
    if len(reward_history) > 5:
        reward_std = np.std(reward_history)
        if reward_std > 5:  # Justerat från 100 för ny reward-skala
            stats_logger.info("• Hög reward-variation upptäckt. Överväg att justera reward-funktionen ytterligare.")
            
        # Check if rewards are converging 
        last_10_std = np.std(reward_history[-10:])
        overall_std = np.std(reward_history)
        if last_10_std < overall_std * 0.5:
            stats_logger.info("• Reward-stabilisering pågår. De justerade reward-parametrarna verkar fungera.")
    
    stats_logger.info("\n" + "=" * 50)

def log_reward_metrics(episode, rewards_this_episode):
    """
    Logga detaljerad reward-statistik för en episod
    
    Args:
        episode: Aktuell episod
        rewards_this_episode: Lista med rewards från episoden
    """
    if not rewards_this_episode:
        return
        
    try:
        # Beräkna statistik
        mean_reward = np.mean(rewards_this_episode)
        max_reward = np.max(rewards_this_episode)
        min_reward = np.min(rewards_this_episode)
        std_reward = np.std(rewards_this_episode)
        
        # Logga resultat
        reward_logger.info(f"EP {episode} REWARD STATS: mean={mean_reward:.2f}, min={min_reward:.2f}, "
                           f"max={max_reward:.2f}, std={std_reward:.2f}")
        
        # Varna för extrema värden (anpassat för ny reward-skala)
        if max_reward > 5 or min_reward < -5:  # Justerat från 100 till 5
            logger.warning(f"Extrema reward-värden i episod {episode}: min={min_reward:.2f}, max={max_reward:.2f}")
            
    except Exception as e:
        logger.error(f"Fel vid loggning av reward-statistik för episod {episode}: {e}")