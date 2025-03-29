import os
import random
import datetime
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union

from data import safe_data_fetcher, calculate_indicators, create_synthetic_data
from environment import TradingEnv
from agent import DQNAgent
from config import args, risk_management_params, reward_params, pattern_params
from logger_setup import logger
from risk_management import RiskManagementNetwork, Position, calculate_dynamic_risk_parameters

def create_environment(
    instrument: str, 
    start_date: datetime.datetime = datetime.datetime.now() - datetime.timedelta(days=365*5),
    end_date: datetime.datetime = datetime.datetime.now() - datetime.timedelta(days=1),
    initial_cash: float = 10000,
    risk_free_rate: float = 0.0,
    risk_window: int = 20,
    synthetic_fallback: bool = True,
    use_kelly: bool = True,  # Parameter for Kelly criterion
    max_kelly_fraction: float = 0.5,  # Parameter for Kelly fraction limit
    kelly_window: int = 20,  # Parameter for Kelly history window
    use_risk_management: bool = True,  # Parameter for risk management
    risk_management_params: Optional[Dict[str, Any]] = None,  # Risk management configuration
    reward_params: Optional[Dict[str, Any]] = None,  # Reward function configuration
    pattern_params: Optional[Dict[str, Any]] = None  # Pattern recognition configuration
) -> TradingEnv:
    """
    Create a trading environment with the specified parameters.
    
    Args:
        instrument: The trading instrument (e.g., 'AAPL')
        start_date: Start date for data fetching
        end_date: End date for data fetching
        initial_cash: Initial cash for the environment
        risk_free_rate: Risk-free rate for Sharpe/Sortino calculations
        risk_window: Window size for risk metrics
        synthetic_fallback: Whether to create synthetic data if real data cannot be fetched
        use_kelly: Whether to use Kelly criterion for position sizing
        max_kelly_fraction: Maximum fraction of capital to allocate (Half-Kelly approach)
        kelly_window: Number of past trades to use for Kelly criterion calculations
        use_risk_management: Whether to use risk management with stop-loss/take-profit
        risk_management_params: Configuration parameters for risk management
        reward_params: Configuration parameters for reward function
        pattern_params: Configuration parameters for pattern recognition
    
    Returns:
        TradingEnv: A configured trading environment
    """
    logger.info(f"Creating environment for {instrument} from {start_date} to {end_date}")
    
    # Attempt to load real data
    env_df = safe_data_fetcher(instrument, start_date, end_date)
    
    # Fall back to synthetic data if needed
    if (env_df is None or len(env_df) < 30) and synthetic_fallback:
        logger.info(f"Using synthetic data for {instrument}")
        env_df = create_synthetic_data(instrument, with_enhanced_features=True)
    elif env_df is not None:
        env_df = env_df.sort_index()
        # Calculate all technical indicators and enhanced features
        env_df = calculate_indicators(env_df)
    else:
        raise ValueError(f"Could not load data for {instrument} and synthetic fallback is disabled")
    
    # Use default params if not provided
    if reward_params is None:
        reward_params = {
            'direct_return_weight': args.direct_return_weight,
            'sharpe_change_weight': args.sharpe_change_weight,
            'drawdown_penalty_weight': args.drawdown_penalty_weight,
            'regime_reward_weight': args.regime_reward_weight,
            'volatility_threshold': args.volatility_threshold,
            'trend_threshold': args.trend_threshold,
            'range_threshold': args.range_threshold
        }
    
    if pattern_params is None:
        pattern_params = {
            'use_pattern_features': args.use_pattern_features,
            'similarity_threshold': args.pattern_similarity_threshold,
            'max_pattern_count': args.max_pattern_count,
            'min_profit_for_pattern': args.min_profit_for_pattern
        }
    
    # Ensure risk_management_params is properly configured
    if risk_management_params is None:
        risk_management_params = {
            'default_stop_loss_pct': args.default_stop_loss_pct,
            'default_take_profit_pct': args.default_take_profit_pct,
            'default_trailing_pct': args.default_trailing_pct,
            'min_stop_loss_pct': args.min_stop_loss_pct,
            'max_stop_loss_pct': args.max_stop_loss_pct,
            'min_take_profit_pct': args.min_take_profit_pct,
            'max_take_profit_pct': args.max_take_profit_pct,
            'stop_loss_adjustment_threshold': args.stop_loss_adjustment_threshold,
            'reward_params': {
                'loss_prevention_weight': args.loss_prevention_weight,
                'profit_capture_weight': args.profit_capture_weight,
                'early_exit_penalty': args.early_exit_penalty,
                'stability_bonus': args.stability_bonus
            }
        }
    
    # Create the environment with the loaded data and parameters
    env = TradingEnv(
        env_df, 
        initial_cash=initial_cash, 
        risk_free_rate=risk_free_rate,
        window_size=risk_window,
        use_kelly=use_kelly,
        max_kelly_fraction=max_kelly_fraction,
        use_risk_management=use_risk_management,
        risk_management_params=risk_management_params,
        reward_params=reward_params,
        pattern_params=pattern_params
    )
    
    # Log Kelly configuration
    if use_kelly:
        logger.info(f"Kelly criterion enabled with max_fraction={max_kelly_fraction}, window={kelly_window}")
    else:
        logger.info("Kelly criterion disabled")
    
    # Log risk management configuration
    if use_risk_management:
        logger.info("Risk management enabled with stop-loss/take-profit functionality")
        if risk_management_params:
            stop_loss = risk_management_params.get('default_stop_loss_pct', 0.05)
            take_profit = risk_management_params.get('default_take_profit_pct', 0.10)
            trailing = risk_management_params.get('default_trailing_pct', 0.03)
            logger.info(f"Default stop-loss: {stop_loss*100:.1f}%, take-profit: {take_profit*100:.1f}%, trailing: {trailing*100:.1f}%")
    else:
        logger.info("Risk management disabled")
    
    # Log reward configuration
    logger.info(f"Reward function: Direct Return Weight={reward_params.get('direct_return_weight', 0.7)}, "
               f"Sharpe Change Weight={reward_params.get('sharpe_change_weight', 5.0)}, "
               f"Drawdown Penalty Weight={reward_params.get('drawdown_penalty_weight', 0.2)}, "
               f"Regime Reward Weight={reward_params.get('regime_reward_weight', 1.0)}")
    
    # Log pattern recognition configuration
    pattern_status = "enabled" if pattern_params.get('use_pattern_features', True) else "disabled"
    logger.info(f"Pattern recognition: {pattern_status}, "
               f"Max patterns={pattern_params.get('max_pattern_count', 50)}, "
               f"Similarity threshold={pattern_params.get('similarity_threshold', 0.7)}")
    
    return env

def create_agent(
    env: TradingEnv,
    model_type: str = 'dqn',
    custom_params: Optional[Dict[str, Any]] = None,
    load_model_path: Optional[str] = None,
    use_kelly: bool = True,  # Parameter for Kelly criterion
    use_risk_management: bool = True,  # Parameter for risk management
    risk_params: Optional[Dict[str, Any]] = None  # Specific risk management network parameters
) -> DQNAgent:
    """
    Create a DQN agent for the specified environment.
    
    Args:
        env: The trading environment
        model_type: Type of DQN model ('dqn', 'double_dqn', 'dueling_dqn', 'rainbow_dqn')
        custom_params: Optional dictionary of custom hyperparameters to override defaults
        load_model_path: Optional path to load a pre-trained model
        use_kelly: Whether to use Kelly criterion for position sizing
        use_risk_management: Whether to use risk management with stop-loss/take-profit
        risk_params: Optional specific parameters for risk management network
    
    Returns:
        DQNAgent: A configured DQN agent
    """
    # Start with default parameters from args
    params = {
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'learning_rate': args.learning_rate,
        'memory_size': args.memory_size,
        'per_alpha': args.per_alpha,
        'per_beta': args.per_beta,
        'per_beta_increment': args.per_beta_increment
    }
    
    # Override with custom parameters if provided
    if custom_params:
        params.update(custom_params)
        logger.info("Using custom agent parameters:")
        for key, value in custom_params.items():
            logger.info(f"  {key}: {value}")
    
    # Add risk management parameters
    if risk_params:
        params['risk_params'] = risk_params
        logger.info("Using custom risk management parameters:")
        for key, value in risk_params.items():
            if isinstance(value, dict):
                logger.info(f"  {key}: {len(value)} parameters")
            else:
                logger.info(f"  {key}: {value}")
    
    # Create the agent with the specified parameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    logger.info(f"Creating {model_type} agent with state_size={state_size}, action_size={action_size}, "
                f"gamma={params['gamma']}, lr={params['learning_rate']}, "
                f"epsilon_decay={params['epsilon_decay']}, Kelly={use_kelly}, "
                f"Risk Management={use_risk_management}")
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        model_type=model_type,
        use_kelly=use_kelly,
        use_risk_management=use_risk_management,
        **params
    )
    
    # Load pre-trained model if specified
    if load_model_path:
        try:
            logger.info(f"Loading agent model from {load_model_path}")
            agent.load(load_model_path)
            logger.info(f"Successfully loaded model with epsilon {agent.epsilon}")
            
            # Respect loaded settings or use specified ones
            if hasattr(agent, 'use_kelly'):
                logger.info(f"Loaded model has use_kelly={agent.use_kelly}")
            else:
                agent.use_kelly = use_kelly
                logger.info(f"Setting use_kelly={use_kelly} for loaded model")
            
            if hasattr(agent, 'use_risk_management'):
                logger.info(f"Loaded model has use_risk_management={agent.use_risk_management}")
            else:
                agent.use_risk_management = use_risk_management
                logger.info(f"Setting use_risk_management={use_risk_management} for loaded model")
                
            # After loading a model with the old reward function, we should adjust for the new reward scale
            logger.info("Adjusting agent for new reward function...")
            if hasattr(agent, 'adjust_for_reward_scale'):
                agent.adjust_for_reward_scale(new_scale=10.0)  # New reward scale is -10 to 10
                logger.info("Agent adjusted for new reward scale")
            else:
                # If method doesn't exist, perform basic adjustments
                agent.epsilon = max(agent.epsilon, 0.3)  # Increase exploration to adapt to new rewards
                agent.loss_history = []  # Reset loss history
                logger.info(f"Reset loss history and set epsilon to {agent.epsilon} for adaptation")
                
        except Exception as e:
            logger.error(f"Failed to load model from {load_model_path}: {e}")
            logger.info("Continuing with freshly initialized model")
    
    return agent

def create_training_session(
    instrument: str,
    model_type: str = 'dqn',
    custom_env_params: Optional[Dict[str, Any]] = None,
    custom_agent_params: Optional[Dict[str, Any]] = None,
    load_model_path: Optional[str] = None,
    use_kelly: bool = True,  # Parameter for Kelly criterion
    use_risk_management: bool = True  # Parameter for risk management
) -> Tuple[TradingEnv, DQNAgent]:
    """
    Create a complete training session with environment and agent.
    
    Args:
        instrument: The trading instrument
        model_type: Type of DQN model
        custom_env_params: Custom parameters for the environment
        custom_agent_params: Custom parameters for the agent
        load_model_path: Optional path to load a pre-trained model
        use_kelly: Whether to use Kelly criterion for position sizing
        use_risk_management: Whether to use risk management with stop-loss/take-profit
    
    Returns:
        Tuple[TradingEnv, DQNAgent]: The configured environment and agent
    """
    # Default environment parameters
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    five_years_ago = yesterday - datetime.timedelta(days=365*5)
    
    env_params = {
        'start_date': five_years_ago,
        'end_date': yesterday,
        'initial_cash': 10000,
        'risk_free_rate': getattr(args, 'risk_free_rate', 0.0),
        'risk_window': getattr(args, 'risk_window', 20),
        'synthetic_fallback': True,
        'use_kelly': use_kelly,
        'max_kelly_fraction': getattr(args, 'max_kelly_fraction', 0.5),
        'kelly_window': getattr(args, 'kelly_window', 20),
        'use_risk_management': use_risk_management,
        'risk_management_params': risk_management_params,
        'reward_params': reward_params,
        'pattern_params': pattern_params
    }
    
    # Override with custom parameters if provided
    if custom_env_params:
        env_params.update(custom_env_params)
    
    # Create environment
    env = create_environment(instrument, **env_params)
    
    # Extract risk_params from agent_params if available
    agent_risk_params = None
    if custom_agent_params and 'risk_params' in custom_agent_params:
        agent_risk_params = custom_agent_params['risk_params']
    elif risk_management_params:
        # Create a copy to avoid modifying the original
        agent_risk_params = risk_management_params.copy()
        # Add learning rate if not set
        if 'learning_rate' not in agent_risk_params:
            agent_risk_params['learning_rate'] = getattr(args, 'risk_learning_rate', 0.001)
    
    # Create agent
    agent = create_agent(
        env=env, 
        model_type=model_type, 
        custom_params=custom_agent_params,
        load_model_path=load_model_path, 
        use_kelly=use_kelly,
        use_risk_management=use_risk_management,
        risk_params=agent_risk_params
    )
    
    return env, agent

def reset_histories() -> Tuple[List[float], List[float], List[float]]:
    """
    Create or reset the history tracking lists
    
    Returns:
        Tuple of empty histories (reward_history, sharpe_history, sortino_history)
    """
    return [], [], []

def switch_instrument(
    instrument: str, 
    opt_params: Optional[Dict[str, Any]] = None,
    load_model_path: Optional[str] = None,
    use_kelly: Optional[bool] = None,  # Parameter for Kelly setting when switching
    use_risk_management: Optional[bool] = None  # Parameter for risk management when switching
) -> Tuple[TradingEnv, DQNAgent, List[float], List[float], List[float]]:
    """
    Switch to a different trading instrument.
    
    Args:
        instrument: The new instrument to use
        opt_params: Optional optimization parameters
        load_model_path: Optional path to load a pre-trained model
        use_kelly: Whether to use Kelly criterion (if None, uses args.use_kelly)
        use_risk_management: Whether to use risk management (if None, uses args.use_risk_management)
    
    Returns:
        Tuple containing the new environment, agent, and empty history lists
    """
    try:
        # Use settings from args if none specified
        kelly_enabled = use_kelly if use_kelly is not None else getattr(args, 'use_kelly', True)
        risk_mgmt_enabled = use_risk_management if use_risk_management is not None else getattr(args, 'use_risk_management', True)
        
        # Calculate dates for yesterday and 5 years back
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        five_years_ago = yesterday - datetime.timedelta(days=365*5)
        
        # Configure agent parameters
        agent_params = {}
        if opt_params:
            logger.info(f"Using optimized parameters when switching to {instrument}")
            agent_params = {
                'gamma': opt_params.get("gamma", args.gamma),
                'epsilon': args.epsilon,  # Keep current epsilon
                'epsilon_min': args.epsilon_min,
                'epsilon_decay': opt_params.get("epsilon_decay", args.epsilon_decay),
                'learning_rate': opt_params.get("learning_rate", args.learning_rate),
                'per_alpha': opt_params.get("per_alpha", args.per_alpha),
                'per_beta': opt_params.get("per_beta", args.per_beta),
                'per_beta_increment': opt_params.get("per_beta_increment", args.per_beta_increment),
                'memory_size': opt_params.get("memory_size", args.memory_size),
            }
            
            # Add optimized risk management parameters if available
            risk_params = {}
            for risk_key in ["risk_learning_rate", "loss_prevention_weight", "profit_capture_weight", 
                           "early_exit_penalty", "stability_bonus", "min_stop_loss_pct", 
                           "max_stop_loss_pct", "min_take_profit_pct", "max_take_profit_pct"]:
                if risk_key in opt_params:
                    # Convert parameter name format
                    clean_key = risk_key.replace("risk_", "")
                    risk_params[clean_key] = opt_params[risk_key]
            
            if risk_params:
                agent_params['risk_params'] = risk_params
            
            # Add optimized reward parameters if available
            custom_reward_params = None
            if any(key in opt_params for key in ["direct_return_weight", "sharpe_change_weight", 
                                                "drawdown_penalty_weight", "regime_reward_weight"]):
                custom_reward_params = reward_params.copy()  # Start with defaults
                # Override with optimized values
                for key in ["direct_return_weight", "sharpe_change_weight", 
                          "drawdown_penalty_weight", "regime_reward_weight"]:
                    if key in opt_params:
                        custom_reward_params[key] = opt_params[key]
            
            # Log optimized parameters
            logger.info("Applied optimized parameters:")
            for key, value in agent_params.items():
                if key in opt_params:
                    logger.info(f"  {key}: {value} (optimized)")
        else:
            logger.info(f"Using default parameters when switching to {instrument}")
        
        # Add custom environment parameters
        env_params = {
            'start_date': five_years_ago,
            'end_date': yesterday,
            'use_kelly': kelly_enabled,
            'max_kelly_fraction': getattr(args, 'max_kelly_fraction', 0.5),
            'kelly_window': getattr(args, 'kelly_window', 20),
            'use_risk_management': risk_mgmt_enabled,
            'risk_management_params': risk_management_params,
            'reward_params': custom_reward_params if 'custom_reward_params' in locals() else reward_params,
            'pattern_params': pattern_params
        }
        
        # Create new environment and agent
        env, agent = create_training_session(
            instrument=instrument,
            model_type=args.model_type,
            custom_agent_params=agent_params,
            custom_env_params=env_params,
            load_model_path=load_model_path,
            use_kelly=kelly_enabled,
            use_risk_management=risk_mgmt_enabled
        )
        
        # Reset history
        reward_history, sharpe_history, sortino_history = reset_histories()
        
        return env, agent, reward_history, sharpe_history, sortino_history
    
    except Exception as e:
        logger.error(f"Failed to switch to instrument {instrument}: {e}")
        # Return None values which will be checked by the caller
        return None, None, [], [], []

def analyze_risk_management_performance(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the performance of risk management strategies based on completed trades.
    
    Args:
        trades: List of completed trades
    
    Returns:
        Dict[str, Any]: A dictionary with risk management statistics
    """
    if not trades:
        return {
            'total_trades': 0,
            'stop_loss_exits': 0,
            'take_profit_exits': 0,
            'stop_loss_pct': 0.0,
            'take_profit_pct': 0.0,
            'avg_profit_pct': 0.0
        }
    
    # Count total trades
    total_trades = len(trades)
    
    # Count stop-loss and take-profit exits
    stop_loss_exits = sum(1 for t in trades if t.get('is_stop_loss_exit', False))
    take_profit_exits = sum(1 for t in trades if t.get('is_take_profit_exit', False))
    
    # Calculate average stop-loss and take-profit percentages
    stop_loss_pcts = [t.get('stop_loss_pct', 0.0) for t in trades if t.get('stop_loss_pct') is not None]
    take_profit_pcts = [t.get('take_profit_pct', 0.0) for t in trades if t.get('take_profit_pct') is not None]
    
    # Calculate average profit/loss for different exit types
    stop_loss_profits = [t.get('profit_loss', 0.0) for t in trades if t.get('is_stop_loss_exit', False)]
    take_profit_profits = [t.get('profit_loss', 0.0) for t in trades if t.get('is_take_profit_exit', False)]
    manual_exit_profits = [t.get('profit_loss', 0.0) for t in trades if 
                          not t.get('is_stop_loss_exit', False) and not t.get('is_take_profit_exit', False)]
    
    # Calculate total profit/loss
    total_profit_pct = sum(t.get('profit_loss', 0.0) for t in trades if t.get('profit_loss') is not None)
    
    # Calculate win/loss ratio
    wins = sum(1 for t in trades if t.get('profit_loss', 0.0) > 0)
    losses = sum(1 for t in trades if t.get('profit_loss', 0.0) <= 0)
    win_loss_ratio = wins / losses if losses > 0 else float('inf')
    
    # Calculate average profit per trade
    avg_profit = total_profit_pct / total_trades if total_trades > 0 else 0.0
    
    # Analyze exit strategies
    exit_strategies = {}
    for strategy in ['standard', 'trailing', 'partial']:
        strategy_trades = [t for t in trades if t.get('exit_strategy') == strategy]
        if strategy_trades:
            win_rate = sum(1 for t in strategy_trades if t.get('profit_loss', 0.0) > 0) / len(strategy_trades)
            avg_profit = np.mean([t.get('profit_loss', 0.0) for t in strategy_trades])
            exit_strategies[strategy] = {
                'count': len(strategy_trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit
            }
    
    # Analyze partial exits
    partial_exit_count = sum(1 for t in trades if t.get('closed_parts', []))
    partial_exit_trades = [t for t in trades if t.get('closed_parts', [])]
    
    partial_exit_stats = None
    if partial_exit_trades:
        total_parts = sum(len(t.get('closed_parts', [])) for t in partial_exit_trades)
        avg_parts_per_trade = total_parts / len(partial_exit_trades)
        partial_exit_stats = {
            'count': partial_exit_count,
            'avg_parts_per_trade': avg_parts_per_trade
        }
    
    # Calculate stats by exit type
    exit_type_stats = {
        'stop_loss': {
            'count': stop_loss_exits,
            'avg_profit': np.mean(stop_loss_profits) if stop_loss_profits else 0.0
        },
        'take_profit': {
            'count': take_profit_exits,
            'avg_profit': np.mean(take_profit_profits) if take_profit_profits else 0.0
        },
        'manual': {
            'count': total_trades - stop_loss_exits - take_profit_exits,
            'avg_profit': np.mean(manual_exit_profits) if manual_exit_profits else 0.0
        }
    }
    
    # Advanced risk metrics
    risk_metrics = {}
    if stop_loss_pcts and take_profit_pcts:
        avg_risk_reward = np.mean([tp/sl for tp, sl in zip(take_profit_pcts, stop_loss_pcts) if sl > 0])
        risk_metrics['avg_risk_reward_ratio'] = avg_risk_reward
    
    # Calculate max drawdowns and runups
    if any('max_drawdown' in t for t in trades):
        max_drawdowns = [t.get('max_drawdown', 0.0) for t in trades if 'max_drawdown' in t]
        avg_max_drawdown = np.mean(max_drawdowns) if max_drawdowns else 0.0
        risk_metrics['avg_max_drawdown'] = avg_max_drawdown
    
    if any('max_runup' in t for t in trades):
        max_runups = [t.get('max_runup', 0.0) for t in trades if 'max_runup' in t]
        avg_max_runup = np.mean(max_runups) if max_runups else 0.0
        risk_metrics['avg_max_runup'] = avg_max_runup
    
    # Create and return statistics
    stats = {
        'total_trades': total_trades,
        'stop_loss_exits': stop_loss_exits,
        'take_profit_exits': take_profit_exits,
        'manual_exits': total_trades - stop_loss_exits - take_profit_exits,
        'stop_loss_pct': np.mean(stop_loss_pcts) if stop_loss_pcts else 0.0,
        'take_profit_pct': np.mean(take_profit_pcts) if take_profit_pcts else 0.0,
        'avg_stop_loss_profit': np.mean(stop_loss_profits) if stop_loss_profits else 0.0,
        'avg_take_profit_profit': np.mean(take_profit_profits) if take_profit_profits else 0.0,
        'avg_manual_exit_profit': np.mean(manual_exit_profits) if manual_exit_profits else 0.0,
        'total_profit_pct': total_profit_pct,
        'avg_profit_pct': avg_profit,
        'win_rate': wins / total_trades if total_trades > 0 else 0.0,
        'win_loss_ratio': win_loss_ratio,
        'wins': wins,
        'losses': losses,
        'exit_strategies': exit_strategies,
        'exit_type_stats': exit_type_stats,
        'partial_exit_stats': partial_exit_stats,
        'risk_metrics': risk_metrics
    }
    
    return stats