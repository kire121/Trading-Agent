import os
import sys
import pygame
import datetime
import time
import numpy as np
import json

from config import args, risk_management_params, reward_params, pattern_params
from factory import create_training_session, switch_instrument, reset_histories, analyze_risk_management_performance
from ui import UIManager
from logger_setup import logger, log_episode_stats, log_training_summary, log_reward_breakdown, log_reward_metrics, log_warning_value

def load_last_optimization():
    """Ladda senaste sparade optimeringsresultat"""
    import json
    import os
    
    filepath = os.path.join('optimization_results', 'latest_optimization.json')
    
    if not os.path.exists(filepath):
        logger.warning("Inga sparade optimeringsresultat hittades.")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            result = json.load(f)
        
        logger.info(f"Laddade optimeringsresultat från {result.get('timestamp', 'unknown date')}")
        logger.info(f"Best value: {result.get('value', 'N/A')}")
        for key, value in result.get('params', {}).items():
            logger.info(f"  {key}: {value}")
        
        return result.get('params', {})
    except Exception as e:
        logger.error(f"Fel vid laddning av optimeringsresultat: {e}")
        return {}

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('optimization_results', exist_ok=True)
    os.makedirs('risk_management_stats', exist_ok=True)
    os.makedirs('statistics', exist_ok=True)
    os.makedirs('rewards', exist_ok=True)  # Ny katalog för reward-loggning
    os.makedirs('debug', exist_ok=True)    # Ny katalog för debugging

    instruments = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    instrument_index = 0
    
    # Variabler för att spåra rewards inom en episod
    episode_rewards = []
    reward_breakdown_logged = False
    breakdown_count = 0
    max_logged_breakdowns = 20  # Max antal reward-breakdowns att logga per episod

    # Initialize Pygame and create window
    pygame.init()
    infoObject = pygame.display.Info()
    screen_width = infoObject.current_w if infoObject.current_w < 1400 else 1200
    screen_height = infoObject.current_h if infoObject.current_h < 1000 else 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"RL Trading Agent - {args.model_type.upper()}")
    clock = pygame.time.Clock()

    # Create UIManager with pre-run menu
    ui_manager = UIManager(
        screen=screen,
        instruments=instruments,
        initial_instrument_index=instrument_index,
        default_episodes=args.episodes  # Default value for episodes
    )

    # Wait for user to make a choice in pre-run menu
    running = True
    while running and ui_manager.show_pre_run_menu:
        event_status = ui_manager.handle_events()
        if event_status == "quit":
            logger.info("Exiting before training (pre-run menu).")
            running = False
            break
        ui_manager.draw(0, None, None, [])
        clock.tick(30)

    if not running:
        pygame.quit()
        sys.exit(0)

    # Get user choices - this overrides any command-line arguments
    pre_run_choice = ui_manager.pre_run_choice
    n_episodes = ui_manager.n_episodes
    logger.info(f"User selected {n_episodes} episodes and mode '{pre_run_choice}'.")
    
    # Hämta Kelly- och Risk Management-parametrar från UI-menyn
    args.use_kelly = ui_manager.use_kelly
    args.max_kelly_fraction = ui_manager.max_kelly_fraction
    args.use_risk_management = ui_manager.use_risk_management if hasattr(ui_manager, 'use_risk_management') else args.use_risk_management
    
    # Logga konfiguration
    logger.info(f"Kelly-kriteriet: {'aktiverat' if args.use_kelly else 'inaktiverat'}, Max fraktion: {args.max_kelly_fraction}")
    logger.info(f"Risk Management: {'aktiverat' if args.use_risk_management else 'inaktiverat'}")
    
    # Uppdatera risk management parametrar
    if hasattr(ui_manager, 'default_stop_loss_pct'):
        risk_management_params['default_stop_loss_pct'] = ui_manager.default_stop_loss_pct
    if hasattr(ui_manager, 'default_take_profit_pct'):
        risk_management_params['default_take_profit_pct'] = ui_manager.default_take_profit_pct
    
    # Om default stop-loss/take-profit är satta, logga dessa
    if args.use_risk_management:
        logger.info(f"Default stop-loss: {risk_management_params['default_stop_loss_pct']*100:.1f}%, "
                   f"take-profit: {risk_management_params['default_take_profit_pct']*100:.1f}%")

    # If user chose to run optimization
    opt_params = {}
    if pre_run_choice == "optimize":
        from optimization import run_optimization
        logger.info("Starting hyperparameter optimization...")
        # Pass UI parameters to optimization process
        study = run_optimization(n_trials=ui_manager.n_trials, ui_manager=ui_manager, screen=screen)
        best_trial = study.best_trial
        logger.info("Optimization complete. Best trial:")
        logger.info(f"  Mean reward: {best_trial.value:.2f}")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        # If toggle is on, use optimized parameters for training
        if ui_manager.use_optimized:
            opt_params = best_trial.params
            logger.info("Using optimized parameters for training.")
        else:
            logger.info("Ignoring optimized parameters, using defaults.")
        # Continue with training after optimization
        pre_run_choice = "train"
        # Ensure episode count is preserved after optimization
        n_episodes = ui_manager.n_episodes
    # Ladda tidigare optimeringsresultat om toggle är på och ingen optimering gjordes nu
    elif ui_manager.use_optimized:
        logger.info("Laddar senaste sparade optimeringsparametrar")
        opt_params = load_last_optimization()
        if not opt_params:
            logger.warning("Inga optimeringsparametrar hittades. Använder standardvärden.")

    # -------------------------------------------------
    # Create environment and agent using factory
    # -------------------------------------------------
    try:
        current_instrument = instruments[instrument_index]
        
        # Configure agent parameters with optimization results if available
        agent_params = {
            'gamma': opt_params.get("gamma", args.gamma),
            'epsilon': args.epsilon,
            'epsilon_min': args.epsilon_min,
            'epsilon_decay': opt_params.get("epsilon_decay", args.epsilon_decay),
            'learning_rate': opt_params.get("learning_rate", args.learning_rate),
            'per_alpha': opt_params.get("per_alpha", args.per_alpha),
            'per_beta': opt_params.get("per_beta", args.per_beta),
            'per_beta_increment': opt_params.get("per_beta_increment", args.per_beta_increment),
        }
        
        # Lägg till risk management-parametrar om optimerade
        risk_params = {}
        for key in ["risk_learning_rate", "risk_batch_size", "min_stop_loss_pct", "max_stop_loss_pct",
                   "min_take_profit_pct", "max_take_profit_pct", "loss_prevention_weight",
                   "profit_capture_weight", "early_exit_penalty", "stability_bonus"]:
            if key in opt_params:
                risk_params[key] = opt_params[key]
        
        if risk_params:
            agent_params['risk_params'] = risk_params
            logger.info("Using optimized risk management parameters:")
            for key, value in risk_params.items():
                logger.info(f"  {key}: {value}")
        
        # Handle model loading if specified
        if args.load_model:
            logger.info(f"Loading model from: {args.load_model}")
            load_model_path = args.load_model
        else:
            load_model_path = None
        
        # Skapa miljö-specifika parametrar för Kelly och Risk Management
        env_params = {
            'use_kelly': args.use_kelly,
            'max_kelly_fraction': args.max_kelly_fraction,
            'kelly_window': args.kelly_window if hasattr(args, 'kelly_window') else 20,
            'use_risk_management': args.use_risk_management,
            'risk_management_params': risk_management_params,
            'reward_params': reward_params,  # Add reward parameters
            'pattern_params': pattern_params  # Add pattern parameters
        }
        
        env, agent = create_training_session(
            instrument=current_instrument,
            model_type=args.model_type,
            custom_agent_params=agent_params,
            custom_env_params=env_params,
            load_model_path=load_model_path,
            use_kelly=args.use_kelly,
            use_risk_management=args.use_risk_management
        )
        
        # I början av träningen, logga initial information
        logger.info(f"Träningen påbörjad med modell: {args.model_type}, Kelly: {args.use_kelly}, "
                  f"Risk Management: {args.use_risk_management}")
        logger.info(f"Initiala parametrar - Epsilon: {agent.epsilon:.4f}, "
                  f"Gamma: {agent.gamma:.4f}, Learning Rate: {agent.learning_rate:.6f}")
        
        # Log reward parameters
        logger.info(f"Reward parametrar - Direct Return Weight: {args.direct_return_weight}, "
                  f"Sharpe Change Weight: {args.sharpe_change_weight}, "
                  f"Drawdown Penalty Weight: {args.drawdown_penalty_weight}, "
                  f"Regime Reward Weight: {args.regime_reward_weight}")
                  
    except Exception as e:
        logger.error(f"Failed to create training session: {e}")
        pygame.quit()
        sys.exit(1)
    
    # Initialize tracking variables
    reward_history, sharpe_history, sortino_history = reset_histories()
    episode_counter = 0
    
    # Initialize risk management tracking
    risk_management_stats = []  # Lista för att spåra risk management-statistik

    # -------------------------------------------------
    # Start the training loop
    # -------------------------------------------------
    running = True
    ui_timeout_counter = 0
    MAX_UI_TIMEOUT = 1000

    while running:
        event_status = ui_manager.handle_events()
        if event_status == "quit":
            logger.info("User is exiting...")
            running = False
            break
        if event_status == "switch":
            # Switch instrument
            instrument_index = ui_manager.instrument_index
            current_instrument = instruments[instrument_index]
            logger.info(f"Switching to instrument: {current_instrument}")
            env, agent, reward_history, sharpe_history, sortino_history = switch_instrument(
                current_instrument, 
                opt_params if ui_manager.use_optimized else None,
                use_kelly=args.use_kelly,
                use_risk_management=args.use_risk_management
            )
            if env is None or agent is None:
                logger.error("Failed to switch instrument, using previous settings")
                # Try to recover by not changing episode counter and continuing
                continue
            
            # Reset episode counter but preserve the target episode count
            episode_counter = 0
            # Important: Make sure n_episodes is still the value from UI
            n_episodes = ui_manager.n_episodes
            logger.info(f"Continuing with {n_episodes} episodes for new instrument")
            
            # Återställ reward-spårning
            episode_rewards = []
            reward_breakdown_logged = False
            breakdown_count = 0
            continue
            
        if event_status == "save_model":
            model_path = os.path.join(args.save_dir, f"{instruments[instrument_index]}_{args.model_type}_ep{episode_counter}")
            agent.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
        if event_status == "toggle_kelly":
            if hasattr(env, 'use_kelly'):
                # Växla Kelly-status
                env.use_kelly = not env.use_kelly
                if agent and hasattr(agent, 'use_kelly'):
                    agent.use_kelly = env.use_kelly
                args.use_kelly = env.use_kelly  # Uppdatera även global inställning
                logger.info(f"Kelly-kriteriet {'aktiverat' if env.use_kelly else 'inaktiverat'} under körning")
        
        if event_status == "toggle_risk_mgmt":
            if hasattr(env, 'use_risk_management'):
                # Växla Risk Management-status
                env.use_risk_management = not env.use_risk_management
                if agent and hasattr(agent, 'use_risk_management'):
                    agent.use_risk_management = env.use_risk_management
                args.use_risk_management = env.use_risk_management  # Uppdatera global inställning
                logger.info(f"Risk Management {'aktiverat' if env.use_risk_management else 'inaktiverat'} under körning")

        # Draw current UI state
        ui_manager.draw(episode_counter, env, agent, reward_history)

        if ui_manager.paused:
            ui_timeout_counter += 1
            if ui_timeout_counter > MAX_UI_TIMEOUT:
                logger.warning("UI paused for too long, continuing...")
                ui_manager.paused = False
                ui_timeout_counter = 0
            clock.tick(5)
            continue
        ui_timeout_counter = 0

        # Play one episode
        state = env.reset().astype(np.float32)
        total_reward = 0
        episode_sharpe = 0
        episode_sortino = 0
        done = False
        step_count = 0
        
        # Återställ reward-tracking för ny episod
        episode_rewards = []
        reward_breakdown_logged = False
        breakdown_count = 0
        
        # För att spåra risk management-belöningar
        risk_rewards = []
        auto_exits = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)
            
            # LÄGG TILL: Spara reward för loggning
            episode_rewards.append(reward)
            
            # LÄGG TILL: Logga reward breakdown om tillgängligt
            if 'direct_return' in info and not reward_breakdown_logged:
                # Logga bara de första X breakdowns för att undvika för mycket loggdata
                if len(episode_rewards) % 100 == 0 or abs(reward) > 5.0:  # Justerat threshold från 10.0 till 5.0
                    reward_components = {
                        'direct_return': info.get('direct_return', 0.0),
                        'sharpe_change': info.get('sharpe_change', 0.0),
                        'drawdown_penalty': info.get('drawdown_penalty', 0.0),
                        'regime_reward': info.get('regime_reward', 0.0),
                        'total': reward
                    }
                    log_reward_breakdown(episode_counter, step_count, reward_components)
                    
                    # Om vi redan har loggat max antal breakdowns, sätt flaggan
                    breakdown_count += 1
                    if breakdown_count >= max_logged_breakdowns:
                        reward_breakdown_logged = True
                        
            # LÄGG TILL: Varna för extrema värden
            if abs(reward) > 5.0:  # Justerat threshold från 100.0 till 5.0
                log_warning_value("reward", reward, threshold=5.0)
            
            # Spara state och action för risk management-uppdatering
            agent.remember(state, action, reward, next_state, done)
            
            # Uppdatera risk management med info om auto exit
            if agent.use_risk_management and hasattr(agent, 'update_risk_management'):
                agent.update_risk_management(state, reward, next_state, done, info)
                
                # Spåra risk management-belöningar och exits
                if 'risk_management_reward' in info:
                    risk_rewards.append(info['risk_management_reward'])
                if info.get('auto_exit_executed', False):
                    auto_exits += 1
            
            state = next_state
            total_reward += reward
            
            # Track Sharpe/Sortino when available
            if 'sharpe' in info:
                episode_sharpe = info['sharpe']  # Save the latest value
            if 'sortino' in info:
                episode_sortino = info['sortino']  # Save the latest value
            
            # Capture drawdown if available
            drawdown = info.get('drawdown', 0)
            
            step_count += 1

            ui_manager.draw(episode_counter, env, agent, reward_history + [total_reward])
            pygame.time.delay(10)

            sub_event = ui_manager.handle_events()
            if sub_event == "quit":
                running = False
                break
            if sub_event == "switch":
                done = True
                break
            if sub_event == "save_model":
                model_path = os.path.join(args.save_dir, f"{instruments[instrument_index]}_{args.model_type}_ep{episode_counter}")
                agent.save(model_path)
                logger.info(f"Model saved to {model_path}")
            if sub_event == "toggle_kelly":
                if hasattr(env, 'use_kelly'):
                    env.use_kelly = not env.use_kelly
                    if agent and hasattr(agent, 'use_kelly'):
                        agent.use_kelly = env.use_kelly
                    args.use_kelly = env.use_kelly  # Uppdatera även global inställning
                    logger.info(f"Kelly-kriteriet {'aktiverat' if env.use_kelly else 'inaktiverat'} under körning i episod {episode_counter}")
            if sub_event == "toggle_risk_mgmt":
                if hasattr(env, 'use_risk_management'):
                    env.use_risk_management = not env.use_risk_management
                    if agent and hasattr(agent, 'use_risk_management'):
                        agent.use_risk_management = env.use_risk_management
                    args.use_risk_management = env.use_risk_management  # Uppdatera global inställning
                    logger.info(f"Risk Management {'aktiverat' if env.use_risk_management else 'inaktiverat'} under körning i episod {episode_counter}")
            if ui_manager.paused:
                pause_timeout = 0
                while ui_manager.paused and running and pause_timeout < MAX_UI_TIMEOUT:
                    sub_event = ui_manager.handle_events()
                    if sub_event == "quit":
                        running = False
                        break
                    ui_manager.draw(episode_counter, env, agent, reward_history + [total_reward])
                    clock.tick(10)
                    pause_timeout += 1
                if pause_timeout >= MAX_UI_TIMEOUT:
                    logger.warning("Pause mode timeout, continuing...")
                    ui_manager.paused = False

        if not running:
            break
        
        # Efter while not done-loopen, lägg till reward-loggning för hela episoden
        if episode_rewards:
            log_reward_metrics(episode_counter, episode_rewards)

        reward_history.append(total_reward)
        sharpe_history.append(episode_sharpe)
        sortino_history.append(episode_sortino)
        
        # Detektera plötsliga stora förändringar i reward
        if len(reward_history) >= 2:
            current_reward = reward_history[-1]
            prev_reward = reward_history[-2]
            reward_change = abs(current_reward - prev_reward)
            
            # Varna för stora reward-förändringar (anpassat för ny reward-skala)
            if reward_change > 20:  # Ändrat från 1000 till 20
                logger.warning(f"Stor förändring i reward upptäckt i episod {episode_counter}: "
                              f"från {prev_reward:.2f} till {current_reward:.2f} "
                              f"(förändring: {reward_change:.2f})")

        # Beräkna average risk/reward ratio för completed trades
        avg_risk_reward_ratio = 0.0
        if hasattr(env, 'completed_trades') and env.completed_trades:
           risk_reward_ratios = []
           for trade in env.completed_trades:
               stop_loss_pct = trade.get('stop_loss_pct', 0)
               take_profit_pct = trade.get('take_profit_pct', 0)
               # Säkerställ att stop_loss_pct och take_profit_pct inte är None
               stop_loss_pct = 0 if stop_loss_pct is None else stop_loss_pct
               take_profit_pct = 0 if take_profit_pct is None else take_profit_pct
               
               if stop_loss_pct > 0:
                   risk_reward_ratios.append(take_profit_pct / stop_loss_pct)
           
           avg_risk_reward_ratio = np.mean(risk_reward_ratios) if risk_reward_ratios else 0.0

        # Log detailed episode statistics with updated reward components
        log_episode_stats(
            episode=episode_counter,
            env=env,
            agent=agent,
            reward=total_reward,
            portfolio_value=env.portfolio_value(),
            sharpe=episode_sharpe,
            sortino=episode_sortino,
            additional_metrics={
                'drawdown': drawdown * 100 if isinstance(drawdown, (int, float)) else 0,  # Convert to percentage
                'risk_reward_ratio': avg_risk_reward_ratio,
                'auto_exits': auto_exits,
                'memory_utilization': len(agent.memory) / agent.memory_size if hasattr(agent, 'memory_size') and agent.memory_size > 0 else 0,
                # Include new reward components
                'direct_return': info.get('direct_return', 0.0) if info else 0.0,
                'sharpe_change': info.get('sharpe_change', 0.0) if info else 0.0,
                'drawdown_penalty': info.get('drawdown_penalty', 0.0) if info else 0.0,
                'regime_reward': info.get('regime_reward', 0.0) if info else 0.0
            }
        )

        # Använd optimerade batch_size om tillgänglig och ui_manager.use_optimized är satt
        batch_size = opt_params.get("batch_size", args.batch_size) if ui_manager.use_optimized else args.batch_size
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)  # Träna båda nätverken

        # Spara risk management-statistik efter varje episod
        if hasattr(env, 'completed_trades') and env.completed_trades:
            # Analysera risk management prestanda om vi har trades
            risk_stats = analyze_risk_management_performance(env.completed_trades)
            
            # Lägg till episod och belöningsinformation
            risk_stats['episode'] = episode_counter
            risk_stats['total_reward'] = total_reward
            risk_stats['sharpe'] = episode_sharpe
            risk_stats['sortino'] = episode_sortino
            risk_stats['auto_exits'] = auto_exits
            risk_stats['avg_risk_reward'] = np.mean(risk_rewards) if risk_rewards else 0.0
            
            # Lägg till i vår spårningslista
            risk_management_stats.append(risk_stats)

        # Loggning med Kelly- och Risk Management-information
        kelly_info = ""
        if hasattr(env, 'use_kelly') and env.use_kelly and hasattr(env, 'kelly_sizer'):
            kelly_fraction = env.kelly_sizer.calculate_kelly_fraction()
            win_rate = env.kelly_sizer.win_rate
            win_loss_ratio = env.kelly_sizer.win_loss_ratio
            kelly_info = f", Kelly: {kelly_fraction:.2f}, WinRate: {win_rate:.2f}, W/L: {win_loss_ratio:.2f}"
        
        risk_mgmt_info = ""
        if hasattr(env, 'use_risk_management') and env.use_risk_management and hasattr(env, 'completed_trades'):
            total_trades = len(env.completed_trades)
            stop_loss_exits = sum(1 for t in env.completed_trades if t.get('is_stop_loss_exit', False))
            take_profit_exits = sum(1 for t in env.completed_trades if t.get('is_take_profit_exit', False))
            
            if total_trades > 0:
                avg_profit = np.mean([t.get('profit_loss', 0.0) for t in env.completed_trades]) * 100  # Till procent
                risk_mgmt_info = f", Trades: {total_trades}, SL: {stop_loss_exits}, TP: {take_profit_exits}, AvgProfit: {avg_profit:.2f}%"
            else:
                risk_mgmt_info = f", Risk Management: {args.use_risk_management}, Inga trades än"

        # Sharpe change info
        sharpe_info = ""
        if info and 'sharpe_change' in info:
            sharpe_info = f", Sharpe Δ: {info['sharpe_change']:.4f}"

        logger.info(f"Episode: {episode_counter}/{n_episodes}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Cash: {env.cash:.2f}, "
                    f"Portfolio Value: {env.portfolio_value():.2f}, "
                    f"Epsilon: {agent.epsilon:.3f}, "
                    f"Sharpe: {episode_sharpe:.2f}{sharpe_info}, "
                    f"Sortino: {episode_sortino:.2f}"
                    f"{kelly_info}{risk_mgmt_info}")
                    
        episode_counter += 1

        if episode_counter % 10 == 0:
            model_path = os.path.join(args.save_dir, f"{instruments[instrument_index]}_{args.model_type}_ep{episode_counter}")
            agent.save(model_path)
            logger.info(f"Autosaving model at episode {episode_counter}")

            # Analyze performance after every 10th episode
            if len(reward_history) > 0:
                avg_reward = np.mean(reward_history[-10:])
                avg_sharpe = np.mean(sharpe_history[-10:]) if len(sharpe_history) >= 10 else 0
                avg_sortino = np.mean(sortino_history[-10:]) if len(sortino_history) >= 10 else 0
                
                # Kelly-statistik för de senaste 10 episoderna
                if hasattr(env, 'use_kelly') and env.use_kelly and hasattr(env, 'kelly_sizer'):
                    kelly_fraction = env.kelly_sizer.calculate_kelly_fraction()
                    win_rate = env.kelly_sizer.win_rate
                    win_loss_ratio = env.kelly_sizer.win_loss_ratio
                    logger.info(f"Kelly-statistik: Fraktion={kelly_fraction:.2f}, WinRate={win_rate:.2f}, W/L Ratio={win_loss_ratio:.2f}")
                
                # Risk Management-statistik för de senaste 10 episoderna
                if hasattr(env, 'use_risk_management') and env.use_risk_management and risk_management_stats:
                    recent_stats = risk_management_stats[-10:] if len(risk_management_stats) >= 10 else risk_management_stats
                    total_trades = sum(s.get('total_trades', 0) for s in recent_stats)
                    stop_loss_exits = sum(s.get('stop_loss_exits', 0) for s in recent_stats)
                    take_profit_exits = sum(s.get('take_profit_exits', 0) for s in recent_stats)
                    
                    if total_trades > 0:
                        all_profits = []
                        for stat in recent_stats:
                            if 'total_trades' in stat and stat['total_trades'] > 0:
                                all_profits.append(stat.get('total_profit_pct', 0.0))
                        
                        avg_profit = np.mean(all_profits) * 100 if all_profits else 0  # Till procent
                        logger.info(f"Risk Management-statistik (senaste 10): "
                                   f"Trades={total_trades}, "
                                   f"SL-Exit={stop_loss_exits}, "
                                   f"TP-Exit={take_profit_exits}, "
                                   f"AvgProfit={avg_profit:.2f}%")
                
                logger.info(f"Last 10 episodes: "
                            f"Average Reward: {avg_reward:.2f}, "
                            f"Average Sharpe: {avg_sharpe:.2f}, "
                            f"Average Sortino: {avg_sortino:.2f}")
                
            # Spara risk management-statistik till fil
            if risk_management_stats:
                stats_path = os.path.join('risk_management_stats', 
                                         f"{instruments[instrument_index]}_risk_stats_ep{episode_counter}.json")
                try:
                    with open(stats_path, 'w') as f:
                        json.dump(risk_management_stats, f, indent=2)
                    logger.info(f"Risk management statistics saved to {stats_path}")
                except Exception as e:
                    logger.error(f"Failed to save risk management statistics: {e}")

        # Check if we've reached the user-defined episode limit
        if episode_counter >= n_episodes:
            logger.info(f"Maximum episodes ({n_episodes}) reached. Exiting.")
            break

        clock.tick(5)

    if len(reward_history) > 0:
        # Analysera reward-distribution
        reward_mean = np.mean(reward_history)
        reward_median = np.median(reward_history)
        reward_std = np.std(reward_history)
        reward_max = np.max(reward_history)
        reward_min = np.min(reward_history)
        
        logger.info(f"Reward-statistik - Medel: {reward_mean:.2f}, Median: {reward_median:.2f}, "
                  f"Std: {reward_std:.2f}, Min: {reward_min:.2f}, Max: {reward_max:.2f}")
                  
        # Jämför första och sista halvan för att se förbättring
        first_half = reward_history[:len(reward_history)//2]
        second_half = reward_history[len(reward_history)//2:]
        
        first_half_mean = np.mean(first_half) if first_half else 0
        second_half_mean = np.mean(second_half) if second_half else 0
        
        improvement = (second_half_mean - first_half_mean) / (abs(first_half_mean) if first_half_mean != 0 else 1)
        logger.info(f"Reward-förbättring: {improvement*100:.2f}% från första halvan till andra halvan av träningen")

    # Final model save before exit
    if agent is not None and episode_counter > 0:
        final_model_path = os.path.join(args.save_dir, f"{instruments[instrument_index]}_{args.model_type}_final")
        agent.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Spara slutlig risk management-statistik
        if risk_management_stats:
            final_stats_path = os.path.join('risk_management_stats', 
                                           f"{instruments[instrument_index]}_risk_stats_final.json")
            try:
                with open(final_stats_path, 'w') as f:
                    json.dump(risk_management_stats, f, indent=2)
                logger.info(f"Final risk management statistics saved to {final_stats_path}")
            except Exception as e:
                logger.error(f"Failed to save final risk management statistics: {e}")

    # Generate comprehensive training summary
    log_training_summary(
        episodes=episode_counter,
        env=env,
        agent=agent,
        reward_history=reward_history,
        sharpe_history=sharpe_history,
        sortino_history=sortino_history
    )

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()