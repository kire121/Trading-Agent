import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import deque

from data import calculate_indicators
from kelly import KellyPositionSizer
from risk_management import (
    Position, 
    calculate_risk_management_reward, 
    calculate_dynamic_risk_parameters,
    analyze_position_risk
)
from logger_setup import logger

class TradingEnv(gym.Env):
    """
    Trading environment with enhanced risk management features.
    Supports dynamic stop-loss/take-profit levels, partial position exits,
    and multiple exit strategies.
    """
    def __init__(self, df, initial_cash=10000, risk_free_rate=0.0, window_size=20, 
                use_kelly=True, max_kelly_fraction=0.5, 
                use_risk_management=True, risk_management_params=None,
                reward_params=None, pattern_params=None):
        # Do not call super().__init__() for older gym versions
        # Gym.Env is an abstract base class without its own __init__
        
        self.df = df
        self.prices = df['Close'].values
        
        # Save dates if available for time-based features
        self.dates = df.index if hasattr(df.index, 'date') else None
        
        self.n_steps = len(self.prices)
        self.indicators = {}
        self.indicator_values = {}
        for indicator in df.columns:
            if indicator in df.columns:
                self.indicators[indicator] = df[indicator].values
                self.indicator_values[indicator] = df[indicator].values
        self.scaler = StandardScaler()
        indicators_array = np.array([self.indicators[ind] for ind in self.indicators]).T
        if indicators_array.shape[0] > 0:
            self.scaled_indicators = self.scaler.fit_transform(indicators_array)
        else:
            self.scaled_indicators = np.array([])
        self.current_step = 0
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.owned = 0.0  # For backward compatibility, replaced by self.position
        self.buy_price = 0  # For backward compatibility
        
        # For Kelly implementation
        self.use_kelly = use_kelly
        self.kelly_sizer = KellyPositionSizer(
            window_size=window_size,
            max_kelly_fraction=max_kelly_fraction
        )
        
        # For risk management
        self.use_risk_management = use_risk_management
        self.risk_management_params = risk_management_params or {}
        self.position = None  # Active position from risk_management.Position
        self.completed_trades = []  # Save completed trades for analysis
        
        # Expanded trade_events format for compatibility
        # Format: (step, price, action, position_size, entry_price, stop_loss_level, take_profit_level, exit_strategy)
        self.trade_events = []
        
        # Track active trades for Kelly calculations (backward compatibility)
        self.active_trade = None
        
        # For risk-adjusted rewards
        self.portfolio_values = [initial_cash]  # Track portfolio values
        self.returns = []  # Track returns
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
        self.window_size = window_size  # Window for risk calculations
        
        # For time-based features
        self.has_time_features = self.dates is not None
        self.time_features = {}
        
        if self.has_time_features:
            # Extract time-based features
            self.time_features['day_of_week'] = np.array([d.dayofweek for d in self.dates])
            self.time_features['day_of_month'] = np.array([d.day for d in self.dates])
            self.time_features['month'] = np.array([d.month for d in self.dates])
            self.time_features['quarter'] = np.array([d.quarter if hasattr(d, 'quarter') 
                                                    else ((d.month-1)//3)+1 for d in self.dates])
            
            # If hours are available (not just dates)
            if hasattr(self.dates[0], 'hour'):
                self.time_features['hour'] = np.array([d.hour for d in self.dates])
            
            # Scale time features to [0, 1]
            for feature in self.time_features:
                if feature == 'day_of_week':
                    self.time_features[feature] = self.time_features[feature] / 6.0  # 0-6
                elif feature == 'day_of_month':
                    self.time_features[feature] = self.time_features[feature] / 31.0  # 1-31
                elif feature == 'month':
                    self.time_features[feature] = self.time_features[feature] / 12.0  # 1-12
                elif feature == 'quarter':
                    self.time_features[feature] = self.time_features[feature] / 4.0  # 1-4
                elif feature == 'hour':
                    self.time_features[feature] = self.time_features[feature] / 23.0  # 0-23
        
        # Add risk management-related variables to state
        self.has_position = 0.0  # 0.0 = no position, 1.0 = has position
        self.profit_pct = 0.0  # Current profit/loss percentage
        self.stop_loss_set = 0.0  # 0.0 = no, 1.0 = yes
        self.take_profit_set = 0.0  # 0.0 = no, 1.0 = yes
        self.stop_loss_pct = 0.0  # Stop-loss percentage level
        self.take_profit_pct = 0.0  # Take-profit percentage level
        self.is_trailing_stop = 0.0  # 0.0 = no, 1.0 = yes
        self.is_trailing_take = 0.0  # 0.0 = no, 1.0 = yes
        self.exit_strategy = 0  # 0 = standard, 1 = trailing, 2 = partial
        self.partial_exits_executed = 0  # Number of partial exits executed
        self.position_size_pct = 0.0  # Current position size as percentage of portfolio
        
        # For market regime detection
        self.market_regime = 0  # 0 = unknown, 1 = trending, 2 = ranging, 3 = volatile
        self.atr_percent = 0.0  # ATR as percentage of price
        self.trend_strength = 0.0  # ADX or other trend strength indicator
        
        # Update observation space to include risk variables
        n_time_features = len(self.time_features) if self.has_time_features else 0
        n_risk_features = 12  # Extended risk features count
        feature_count = 3 + len(self.indicators) + n_time_features + n_risk_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)
        
        # Update action space for new actions:
        # 0: Hold
        # 1: Buy (full position)
        # 2: Sell entire position
        # 3: Buy with Kelly-optimized position
        # 4: Set stop-loss (X% below current price)
        # 5: Set trailing stop-loss (X% below highest price)
        # 6: Set take-profit (X% above current price)
        # 7: Set trailing take-profit (X% above current price)
        # 8: Sell 25% of position
        # 9: Sell 50% of position
        # 10: Sell 75% of position
        # 11: Deactivate stop-loss
        # 12: Deactivate take-profit
        # 13: Set partial exit strategy (with multiple take-profit levels)
        # 14: Set trailing strategy (both stop-loss and take-profit as trailing)
        # 15: Adjust risk parameters based on market conditions
        self.action_space = spaces.Discrete(16)
        self.prev_portfolio_value = initial_cash
        
        # Predefined levels for stop-loss and take-profit
        self.default_stop_loss_pct = self.risk_management_params.get('default_stop_loss_pct', 0.05)  # 5%
        self.default_take_profit_pct = self.risk_management_params.get('default_take_profit_pct', 0.10)  # 10%
        self.default_trailing_pct = self.risk_management_params.get('default_trailing_pct', 0.03)  # 3%
        
        # For tracking maximum portfolio value (for drawdown calculation)
        self.peak_portfolio_value = initial_cash
                
        # Store reward parameters
        self.reward_params = reward_params or {}
        
        # Store pattern parameters
        self.pattern_params = pattern_params or {}
        
        # SUCCESS PATTERN REINFORCEMENT VARIABLES
        self.successful_trade_patterns = []
        self.min_success_profit = 0.01  # Minimum profit to be considered "successful"
        self.max_patterns = 50  # Limit number of saved patterns
        self.similarity_threshold = 0.8  # How similar a pattern must be for reward
        
        # For tracking Sharpe ratio changes
        self.previous_sharpe = 0.0
        
        # For tracking market regimes and dynamics
        self.regime_history = deque(maxlen=50)  # Track recent market regimes
        self.volatility_history = deque(maxlen=50)  # Track recent volatility
        
        # For evaluating trading performance
        self.trade_stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'avg_profit_pct': 0.0,
            'avg_loss_pct': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0
        }
        
        # For calculating action probabilities
        self.action_probs = np.ones(self.action_space.n) / self.action_space.n
        
        # For adaptive action masking based on state
        self.action_masks = np.ones(self.action_space.n)
        
        # For tracking market context
        self.market_context = {}

    def calculate_state_similarity(self, state1, state2):
        """Calculate similarity between two states (0-1 scale)"""
        # Safety check - ensure states are numpy arrays
        try:
            s1 = np.array(state1, dtype=np.float32) if not isinstance(state1, np.ndarray) else state1.astype(np.float32)
            s2 = np.array(state2, dtype=np.float32) if not isinstance(state2, np.ndarray) else state2.astype(np.float32)
            
            # Check that shapes are identical
            if s1.shape != s2.shape:
                return 0.0
                
            # Calculate normalized Euclidean distance
            distance = np.linalg.norm(s1 - s2)
            max_possible_distance = np.sqrt(len(s1))  # Max possible distance for normalized values
            similarity = 1.0 - min(1.0, distance / max_possible_distance)
            
            return float(similarity)  # Convert to Python float for safety
        except Exception as e:
            # On error, return 0 (no similarity)
            print(f"Error calculating state similarity: {e}")
            return 0.0

    def add_successful_pattern(self, entry_state, action, profit):
        """Save a successful trading pattern"""
        if profit < self.min_success_profit:
            return  # Ignore marginally successful trades
        
        try:
            # Create a safe copy of the state
            if isinstance(entry_state, np.ndarray):
                state_copy = entry_state.copy()
            else:
                # Use simple conversion if not a numpy array
                state_copy = np.array(entry_state, dtype=np.float32)
                
            # Create a pattern with entry state, action and result
            pattern = {
                'state': state_copy,
                'action': int(action),  # Ensure action is an int
                'profit': float(profit),  # Ensure profit is a float
                'timestamp': int(self.current_step),
                'market_regime': self.market_regime,
                'atr_percent': self.atr_percent,
                'exit_strategy': self.position.exit_strategy if self.position else 'standard',
                'risk_reward_ratio': float(self.take_profit_pct / self.stop_loss_pct) if self.stop_loss_pct > 0 else 0.0
            }
            
            # Add to list and keep only the latest max_patterns
            self.successful_trade_patterns.append(pattern)
            if len(self.successful_trade_patterns) > self.max_patterns:
                self.successful_trade_patterns.pop(0)  # Remove oldest pattern
        except Exception as e:
            # On error, print the error but let the program continue
            print(f"Error adding successful pattern: {e}")

    def calculate_success_pattern_similarity(self, current_state, action):
        """Calculate how similar current state/action is to previous successful trades"""
        if not self.successful_trade_patterns:
            return 0.0
        
        try:
            max_similarity = 0.0
            max_profit = 0.0
            max_pattern = None
            
            # Compare with each successful pattern
            for pattern in self.successful_trade_patterns:
                # Only compare with patterns that took the same action
                if pattern['action'] == action:
                    similarity = self.calculate_state_similarity(current_state, pattern['state'])
                    
                    # Update max similarity and its associated profit
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_profit = pattern['profit']
                        max_pattern = pattern
            
            # Return reward based on similarity and how profitable the similar trade was
            if max_similarity > self.similarity_threshold and max_pattern:
                # Also consider market regime match for higher reward
                regime_match = 1.0 if max_pattern.get('market_regime', 0) == self.market_regime else 0.5
                return max_similarity * max_profit * regime_match * 0.5  # Scaling factor
            return 0.0
        except Exception as e:
            # On error, print error and return 0
            print(f"Error calculating pattern similarity: {e}")
            return 0.0

    def calculate_sharpe_ratio(self, annualized=True):
        """Calculate Sharpe ratio based on returns history"""
        if len(self.returns) < 2:
            return 0.0
        
        # Use only the latest window_size returns
        recent_returns = self.returns[-self.window_size:] if len(self.returns) > self.window_size else self.returns
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        # Annualize if requested (assuming daily returns)
        if annualized:
            sharpe = sharpe * np.sqrt(252)  # 252 trading days in a year
        
        return sharpe

    def calculate_sortino_ratio(self, annualized=True, target_return=0.0):
        """Calculate Sortino ratio based on returns history"""
        if len(self.returns) < 2:
            return 0.0
        
        # Use only the latest window_size returns
        recent_returns = self.returns[-self.window_size:] if len(self.returns) > self.window_size else self.returns
        
        mean_return = np.mean(recent_returns)
        
        # Calculate downside deviation (only for returns below target)
        downside_returns = [r for r in recent_returns if r < target_return]
        if not downside_returns:
            return 0.0  # No negative returns
        
        downside_deviation = np.sqrt(np.mean([(r - target_return)**2 for r in downside_returns]))
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = (mean_return - self.risk_free_rate) / downside_deviation
        
        # Annualize if requested (assuming daily returns)
        if annualized:
            sortino = sortino * np.sqrt(252)  # 252 trading days in a year
        
        return sortino

    def portfolio_value(self):
        """Calculate total portfolio value (cash + positions)"""
        if self.current_step < self.n_steps:
            current_price = self.prices[self.current_step]
            if self.position and self.position.active_size > 0:
                return self.cash + (current_price * self.position.active_size)
            else:
                return self.cash
        else:
            return self.cash

    def detect_market_regime(self):
        """
        Detect current market regime based on indicators and price action.
        Updates self.market_regime:
        0 = unknown, 1 = trending, 2 = ranging, 3 = volatile
        """
        if self.current_step < 20:  # Need enough data
            self.market_regime = 0
            return 0
        
        # Use ADX for trend strength if available
        if 'adx' in self.indicator_values:
            adx = self.indicator_values['adx'][self.current_step]
            adx_trend = 1 if adx > 25 else 0
        else:
            # Alternative trend detection
            last_20_prices = self.prices[max(0, self.current_step-20):self.current_step+1]
            price_diff = (last_20_prices[-1] - last_20_prices[0]) / last_20_prices[0]
            adx_trend = 1 if abs(price_diff) > 0.03 else 0
        
        # Use ATR for volatility if available
        if 'atr_percent' in self.indicator_values:
            atr_pct = self.indicator_values['atr_percent'][self.current_step]
            self.atr_percent = atr_pct
        else:
            # Calculate recent volatility as alternative
            last_20_returns = np.diff(self.prices[max(0, self.current_step-20):self.current_step+1]) / self.prices[max(0, self.current_step-20):self.current_step]
            atr_pct = np.std(last_20_returns) * 100  # In percentage
            self.atr_percent = atr_pct
        
        # High ATR indicates volatile market
        is_volatile = atr_pct > 2.0  # >2% daily move is volatile
        
        # Determine overall regime
        if is_volatile:
            regime = 3  # Volatile
        elif adx_trend:
            regime = 1  # Trending
        else:
            regime = 2  # Ranging
        
        # Store value
        self.market_regime = regime
        
        # Add to history
        self.regime_history.append(regime)
        self.volatility_history.append(atr_pct)
        
        return regime

    def get_market_features(self):
        """Get current market features for decision making"""
        # Default values
        features = {
            'price': self.prices[self.current_step],
            'market_regime': self.market_regime,
            'atr_percent': self.atr_percent,
            'trend_strength': 0.0,
            'volatility': 0.0
        }
        
        # Add available indicators
        if 'adx' in self.indicator_values:
            features['trend_strength'] = self.indicator_values['adx'][self.current_step]
            self.trend_strength = features['trend_strength']
        
        if 'volatility_20' in self.indicator_values:
            features['volatility'] = self.indicator_values['volatility_20'][self.current_step]
        
        # Add support/resistance levels if available
        if 'support_level' in self.indicator_values:
            features['support_level'] = self.indicator_values['support_level'][self.current_step]
        if 'resistance_level' in self.indicator_values:
            features['resistance_level'] = self.indicator_values['resistance_level'][self.current_step]
        
        # Add recommended risk parameters if available
        if 'optimal_sl_pct' in self.indicator_values:
            features['optimal_sl_pct'] = self.indicator_values['optimal_sl_pct'][self.current_step]
        if 'optimal_tp_pct' in self.indicator_values:
            features['optimal_tp_pct'] = self.indicator_values['optimal_tp_pct'][self.current_step]
        
        # Store in market context
        self.market_context = features
        
        return features

    def update_action_probabilities(self):
        """
        Update action probabilities based on current state.
        Used for more effective exploration and to guide the agent.
        """
        # Default uniform distribution
        action_probs = np.ones(self.action_space.n) / self.action_space.n
        
        # If no position, give higher probability to buy actions
        if not self.position or self.position.active_size == 0:
            # Increase buy actions probability
            action_probs[1] *= 3.0  # Buy
            action_probs[3] *= 3.0  # Kelly buy
            
            # Reduce sell/risk management actions probability
            for i in range(4, 16):
                action_probs[i] *= 0.2
        else:
            # If we have a position, give higher probability to risk management actions
            
            # Check if stop-loss and take-profit are not set
            has_stop_loss = self.position.stop_loss_level is not None
            has_take_profit = self.position.take_profit_level is not None
            
            # If risk levels are not set, prioritize setting them
            if not has_stop_loss:
                action_probs[4] *= 5.0  # Set stop-loss
                action_probs[5] *= 3.0  # Set trailing stop
            
            if not has_take_profit:
                action_probs[6] *= 5.0  # Set take-profit
                action_probs[7] *= 3.0  # Set trailing take
            
            # If we're profitable, give more weight to exit actions
            if self.profit_pct > 0.02:  # >2% profit
                action_probs[2] *= 2.0  # Sell
                action_probs[8] *= 3.0  # Sell 25%
                action_probs[9] *= 2.0  # Sell 50%
                action_probs[13] *= 3.0  # Set partial exit strategy
                
            # Based on market regime, adjust strategy probabilities
            if self.market_regime == 1:  # Trending
                action_probs[5] *= 2.0  # Trailing stop more useful in trend
                action_probs[14] *= 3.0  # Trailing strategy
                
            elif self.market_regime == 2:  # Ranging
                action_probs[13] *= 3.0  # Partial exit strategy more useful in range
                
            elif self.market_regime == 3:  # Volatile
                action_probs[4] *= 2.0  # Fixed stop-loss important in volatility
                action_probs[15] *= 3.0  # Adjust risk parameters
        
        # Normalize to sum to 1.0
        self.action_probs = action_probs / np.sum(action_probs)
        
        # Also update action masks
        self._update_action_masks()
        
        return self.action_probs

    def _update_action_masks(self):
        """
        Update action masks to prevent invalid actions.
        1 = allowed, 0 = not allowed in current state.
        """
        # Default all actions are allowed
        masks = np.ones(self.action_space.n)
        
        # If no position, mask sell and risk management actions
        if not self.position or self.position.active_size == 0:
            masks[2] = 0  # Sell
            masks[4:16] = 0  # All risk management actions
        else:
            # Have position, check specifics
            
            # If stop-loss already set, mask set stop-loss actions
            if self.position.stop_loss_level is not None:
                masks[4] = 0  # Set stop-loss
            else:
                masks[11] = 0  # Remove stop-loss
            
            # If take-profit already set, mask set take-profit actions
            if self.position.take_profit_level is not None:
                masks[6] = 0  # Set take-profit
            else:
                masks[12] = 0  # Remove take-profit
                
            # If position is too small, mask partial sells
            if self.position.active_size < 0.2:  # Less than 20% of a full position
                masks[8:11] = 0  # Partial sells
        
        self.action_masks = masks
        return masks

    def reset(self):
        """Reset the environment for a new episode"""
        self.current_step = 0
        self.cash = self.initial_cash
        self.owned = 0.0  # Reset holdings (backward compatibility)
        self.buy_price = 0  # Reset buy price (backward compatibility)
        self.trade_events = []
        self.active_trade = None  # Reset active trade
        self.position = None  # Reset position
        self.completed_trades = []  # Reset completed trades
        self.prev_portfolio_value = self.initial_cash
        # Reset risk tracking
        self.portfolio_values = [self.initial_cash]
        self.returns = []
        
        # Reset risk management variables
        self.has_position = 0.0
        self.profit_pct = 0.0
        self.stop_loss_set = 0.0
        self.take_profit_set = 0.0
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0
        self.is_trailing_stop = 0.0
        self.is_trailing_take = 0.0
        self.exit_strategy = 0
        self.partial_exits_executed = 0
        self.position_size_pct = 0.0
        
        # Reset market context
        self.market_regime = 0
        self.atr_percent = 0.0
        self.trend_strength = 0.0
        
        # Reset peak portfolio value
        self.peak_portfolio_value = self.initial_cash
        
        # Reset Sharpe tracking
        self.previous_sharpe = 0.0
        
        # Don't reset successful_trade_patterns - we want to keep them between episodes
        # as they represent previous knowledge
        
        # Initial state
        return self._get_state()

    def _get_state(self):
        """Get the current state representation"""
        if self.current_step >= self.n_steps:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Detect market regime
        self.detect_market_regime()
        
        # Get current price and calculate price change
        current_price = self.prices[self.current_step]
        price_change = 0.0 if self.current_step == 0 else (current_price - self.prices[self.current_step - 1]) / (self.prices[self.current_step - 1] or 1e-10)
        
        # Update risk management variables
        if self.position:
            self.has_position = 1.0
            self.profit_pct = (current_price - self.position.entry_price) / self.position.entry_price if self.position.is_long else (self.position.entry_price - current_price) / self.position.entry_price
            self.stop_loss_set = 1.0 if self.position.stop_loss_level is not None else 0.0
            self.take_profit_set = 1.0 if self.position.take_profit_level is not None else 0.0
            self.stop_loss_pct = self.position.stop_loss_pct if self.position.stop_loss_pct is not None else 0.0
            self.take_profit_pct = self.position.take_profit_pct if self.position.take_profit_pct is not None else 0.0
            self.is_trailing_stop = 1.0 if self.position.is_trailing_stop else 0.0
            self.is_trailing_take = 1.0 if self.position.is_trailing_take else 0.0
            
            # Normalize position size as percentage of portfolio
            portfolio_value = self.portfolio_value()
            if portfolio_value > 0:
                position_value = current_price * self.position.active_size
                self.position_size_pct = position_value / portfolio_value
            
            # Set exit strategy
            if self.position.exit_strategy == 'standard':
                self.exit_strategy = 0
            elif self.position.exit_strategy == 'trailing':
                self.exit_strategy = 1
            elif self.position.exit_strategy == 'partial':
                self.exit_strategy = 2
            
            # Count executed partial exits
            self.partial_exits_executed = len(self.position.closed_parts)
        else:
            self.has_position = 0.0
            self.profit_pct = 0.0
            self.stop_loss_set = 0.0
            self.take_profit_set = 0.0
            self.stop_loss_pct = 0.0
            self.take_profit_pct = 0.0
            self.is_trailing_stop = 0.0
            self.is_trailing_take = 0.0
            self.exit_strategy = 0
            self.partial_exits_executed = 0
            self.position_size_pct = 0.0
        
        # Base state components
        position_size = self.position.active_size if self.position else 0.0
        base_state = [current_price, position_size, price_change]
        
        # Add technical indicators
        if len(self.scaled_indicators) > 0 and self.current_step < len(self.scaled_indicators):
            indicator_values = self.scaled_indicators[self.current_step].tolist()
            state = base_state + indicator_values
        else:
            state = base_state
        
        # Add time-based features
        if self.has_time_features:
            for feature in self.time_features:
                state.append(self.time_features[feature][self.current_step])
        
        # Add risk management variables
        risk_state = [
            self.has_position, 
            self.profit_pct, 
            self.stop_loss_set, 
            self.take_profit_set,
            self.stop_loss_pct,
            self.take_profit_pct,
            self.is_trailing_stop,
            self.is_trailing_take,
            float(self.exit_strategy),  # Convert to float
            float(self.partial_exits_executed),  # Convert to float
            self.position_size_pct,
            float(self.market_regime)  # Convert to float
        ]
        
        state.extend(risk_state)
        
        return np.array(state, dtype=np.float32)

    def execute_action(self, action, current_price, current_state=None, agent=None):
        """
        Execute an action from the agent
        
        Args:
            action: Action selected by agent
            current_price: Current market price
            current_state: Current state (for pattern matching)
            agent: The agent instance (for consulting risk network)
        """
        # Actions:
        # 0: Hold
        # 1: Buy (full position)
        # 2: Sell entire position
        # 3: Buy with Kelly-optimized position
        # 4: Set stop-loss
        # 5: Set trailing stop-loss
        # 6: Set take-profit
        # 7: Set trailing take-profit
        # 8: Sell 25% of position
        # 9: Sell 50% of position
        # 10: Sell 75% of position
        # 11: Deactivate stop-loss
        # 12: Deactivate take-profit
        # 13: Set partial exit strategy
        # 14: Set trailing strategy
        # 15: Adjust risk parameters based on market conditions
        
        if action == 0:  # Hold
            pass
        
        elif action == 1 and (self.position is None or self.position.active_size == 0):  # Buy with full position
            self.owned = 1.0  # For backward compatibility
            
            # Create new position
            self.position = Position(entry_price=current_price, size=1.0, is_long=True)
            self.position.entry_time = self.current_step
            
            # Save current state for Success Pattern Reinforcement
            if current_state is not None:
                # Use safe copying
                try:
                    if isinstance(current_state, np.ndarray):
                        self.position.entry_state = current_state.copy()
                    else:
                        self.position.entry_state = np.array(current_state, dtype=np.float32)
                    self.position.entry_action = int(action)
                except Exception as e:
                    # If error, continue without saving state
                    print(f"Could not save entry state: {e}")
            
            # Set market context information
            self.position.set_market_context(
                atr=self.atr_percent / 100.0 * current_price if self.atr_percent > 0 else 0.01 * current_price,
                atr_percent=self.atr_percent,
                market_regime=self.market_regime,
                trend_strength=self.trend_strength
            )
            
            # Consult agent for risk parameters if available
            if agent and hasattr(agent, 'get_risk_management_levels') and self.use_risk_management:
                risk_levels = agent.get_risk_management_levels(current_state, self.get_market_features())
                
                # Set stop-loss and take-profit based on risk network recommendation
                self.position.set_stop_loss(percent=risk_levels['stop_loss_pct'], is_trailing=risk_levels['is_trailing_stop'])
                self.position.set_take_profit(percent=risk_levels['take_profit_pct'], is_trailing=risk_levels['is_trailing_take'])
                
                # Set exit strategy
                self.position.set_exit_strategy(
                    strategy=risk_levels['exit_strategy'],
                    levels=risk_levels.get('partial_exit_levels', []),
                    sizes=risk_levels.get('partial_exit_sizes', [])
                )
            elif self.use_risk_management:
                # If no agent or risk network available, use dynamic risk parameters based on market conditions
                risk_params = calculate_dynamic_risk_parameters(
                    price_data=self.df,
                    current_idx=self.current_step,
                    lookahead=20,
                    market_regime=["unknown", "trending", "ranging", "volatile"][self.market_regime]
                )
                
                # Set risk parameters
                self.position.set_stop_loss(percent=risk_params['stop_loss_pct'], is_trailing=risk_params['is_trailing_stop'])
                self.position.set_take_profit(percent=risk_params['take_profit_pct'], is_trailing=risk_params['is_trailing_take'])
                
                # Set exit strategy
                self.position.set_exit_strategy(
                    strategy=risk_params['exit_strategy'],
                    levels=risk_params.get('partial_exit_levels', []),
                    sizes=risk_params.get('partial_exit_sizes', [])
                )
            
            # Update cash
            self.cash -= current_price
            
            # Register event
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                1.0, 
                current_price, 
                self.position.stop_loss_level, 
                self.position.take_profit_level,
                self.position.exit_strategy
            ))
            
            # Register active trade for Kelly (backward compatibility)
            self.active_trade = {
                'step': self.current_step,
                'entry_price': current_price,
                'position_size': 1.0,
                'type': 'long'
            }
            
        elif action == 2 and self.position and self.position.active_size > 0:  # Sell all
            # Calculate sale value
            sell_value = current_price * self.position.active_size
            
            # If we have an active trade, calculate result and update Kelly (backward compatibility)
            if self.active_trade is not None:
                profit_loss = (current_price - self.active_trade['entry_price']) / self.active_trade['entry_price']
                is_win = profit_loss > 0
                
                trade_result = {
                    'entry_price': self.active_trade['entry_price'],
                    'exit_price': current_price,
                    'profit_loss': profit_loss,
                    'is_win': is_win,
                    'position_size': self.active_trade['position_size']
                }
                
                self.kelly_sizer.add_trade_result(trade_result)
                
                # Clear active trade
                self.active_trade = None
            
            # Close position and save trade information
            if self.position:
                trade_info = self.position.close_position(current_price, self.current_step)
                
                # Add entry_state and entry_action if available
                if hasattr(self.position, 'entry_state') and self.position.entry_state is not None:
                    trade_info['entry_state'] = self.position.entry_state
                    trade_info['entry_action'] = self.position.entry_action
                    
                    # Add trade pattern if profitable
                    profit = trade_info.get('profit_loss', 0.0)
                    if profit > 0:
                        self.add_successful_pattern(
                            self.position.entry_state, 
                            self.position.entry_action if self.position.entry_action is not None else action, 
                            profit
                        )
                
                self.completed_trades.append(trade_info)
                
                # Update cash
                self.cash += sell_value
                
                # Reset position
                self.position = None
            
            # Reset owned for backward compatibility
            self.owned = 0.0
            
            # Register event
            self.trade_events.append((self.current_step, current_price, action, 0.0, current_price, None, None, None))
            
        elif action == 3 and (self.position is None or self.position.active_size == 0):  # Buy with Kelly-optimized position
            # Calculate position size with Kelly if enabled
            if self.use_kelly:
                # Get investable capital 
                investable_capital = self.cash
                
                # Calculate Kelly-based position size (fraction of capital)
                kelly_fraction = self.kelly_sizer.calculate_kelly_fraction()
                
                # Calculate how many units we can buy
                position_size = kelly_fraction * investable_capital / current_price
                
                # Ensure we don't spend more than available capital
                position_cost = position_size * current_price
                if position_cost > self.cash:
                    position_size = self.cash / current_price
            else:
                # If Kelly is disabled, use default behavior (full position)
                position_size = 1.0
            
            # Create new position
            self.position = Position(entry_price=current_price, size=position_size, is_long=True)
            self.position.entry_time = self.current_step
            
            # Save current state for Success Pattern Reinforcement
            if current_state is not None:
                # Use safe copying
                try:
                    if isinstance(current_state, np.ndarray):
                        self.position.entry_state = current_state.copy()
                    else:
                        self.position.entry_state = np.array(current_state, dtype=np.float32)
                    self.position.entry_action = int(action)
                except Exception as e:
                    # If error, continue without saving state
                    print(f"Could not save entry state: {e}")
            
            # Set market context information
            self.position.set_market_context(
                atr=self.atr_percent / 100.0 * current_price if self.atr_percent > 0 else 0.01 * current_price,
                atr_percent=self.atr_percent,
                market_regime=self.market_regime,
                trend_strength=self.trend_strength
            )
            
            # Consult agent for risk parameters if available
            if agent and hasattr(agent, 'get_risk_management_levels') and self.use_risk_management:
                risk_levels = agent.get_risk_management_levels(current_state, self.get_market_features())
                
                # Set stop-loss and take-profit based on risk network recommendation
                self.position.set_stop_loss(percent=risk_levels['stop_loss_pct'], is_trailing=risk_levels['is_trailing_stop'])
                self.position.set_take_profit(percent=risk_levels['take_profit_pct'], is_trailing=risk_levels['is_trailing_take'])
                
                # Set exit strategy
                self.position.set_exit_strategy(
                    strategy=risk_levels['exit_strategy'],
                    levels=risk_levels.get('partial_exit_levels', []),
                    sizes=risk_levels.get('partial_exit_sizes', [])
                )
            elif self.use_risk_management:
                # If no agent or risk network available, use dynamic risk parameters based on market conditions
                risk_params = calculate_dynamic_risk_parameters(
                    price_data=self.df,
                    current_idx=self.current_step,
                    lookahead=20,
                    market_regime=["unknown", "trending", "ranging", "volatile"][self.market_regime]
                )
                
                # Set risk parameters
                self.position.set_stop_loss(percent=risk_params['stop_loss_pct'], is_trailing=risk_params['is_trailing_stop'])
                self.position.set_take_profit(percent=risk_params['take_profit_pct'], is_trailing=risk_params['is_trailing_take'])
                
                # Set exit strategy
                self.position.set_exit_strategy(
                    strategy=risk_params['exit_strategy'],
                    levels=risk_params.get('partial_exit_levels', []),
                    sizes=risk_params.get('partial_exit_sizes', [])
                )
            
            # Execute purchase
            position_cost = position_size * current_price
            self.cash -= position_cost
            self.owned = position_size  # For backward compatibility
            
            # Register active trade (backward compatibility)
            self.active_trade = {
                'step': self.current_step,
                'entry_price': current_price,
                'position_size': position_size,
                'type': 'long'
            }
            
            # Register event
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                position_size, 
                current_price, 
                self.position.stop_loss_level, 
                self.position.take_profit_level,
                self.position.exit_strategy
            ))
        
        elif action == 4 and self.position and self.position.active_size > 0:  # Set stop-loss
            # Set standard stop-loss (5% below current price)
            self.position.set_stop_loss(percent=self.default_stop_loss_pct, is_trailing=False)
            
            # Register event
            stop_level = self.position.stop_loss_level
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                stop_level, 
                self.position.take_profit_level,
                self.position.exit_strategy
            ))
        
        elif action == 5 and self.position and self.position.active_size > 0:  # Set trailing stop-loss
            # Set trailing stop-loss (3% below highest price)
            self.position.set_stop_loss(percent=self.default_trailing_pct, is_trailing=True)
            
            # Register event
            stop_level = self.position.stop_loss_level
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                stop_level, 
                self.position.take_profit_level,
                self.position.exit_strategy
            ))
        
        elif action == 6 and self.position and self.position.active_size > 0:  # Set take-profit
            # Set standard take-profit (10% above current price)
            self.position.set_take_profit(percent=self.default_take_profit_pct, is_trailing=False)
            
            # Register event
            take_level = self.position.take_profit_level
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                self.position.stop_loss_level, 
                take_level,
                self.position.exit_strategy
            ))
        
        elif action == 7 and self.position and self.position.active_size > 0:  # Set trailing take-profit
            # Set trailing take-profit (3% above lowest price)
            self.position.set_take_profit(percent=self.default_trailing_pct, is_trailing=True)
            
            # Register event
            take_level = self.position.take_profit_level
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                self.position.stop_loss_level, 
                take_level,
                self.position.exit_strategy
            ))
        
        elif action == 8 and self.position and self.position.active_size > 0:  # Sell 25% of position
            size_before = self.position.active_size
            closed_size = self.position.close_partial(0.25, current_price, self.current_step)
            
            # Update cash
            self.cash += closed_size * current_price
            self.owned = self.position.active_size  # Update owned for backward compatibility
            
            # Register event
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                self.position.stop_loss_level, 
                self.position.take_profit_level,
                self.position.exit_strategy
            ))
        
        elif action == 9 and self.position and self.position.active_size > 0:  # Sell 50% of position
            size_before = self.position.active_size
            closed_size = self.position.close_partial(0.5, current_price, self.current_step)
            
            # Update cash
            self.cash += closed_size * current_price
            self.owned = self.position.active_size  # Update owned for backward compatibility
            
            # Register event
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                self.position.stop_loss_level, 
                self.position.take_profit_level,
                self.position.exit_strategy
            ))
        
        elif action == 10 and self.position and self.position.active_size > 0:  # Sell 75% of position
            size_before = self.position.active_size
            closed_size = self.position.close_partial(0.75, current_price, self.current_step)
            
            # Update cash
            self.cash += closed_size * current_price
            self.owned = self.position.active_size  # Update owned for backward compatibility
            
            # Register event
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                self.position.stop_loss_level, 
                self.position.take_profit_level,
                self.position.exit_strategy
            ))
        
        elif action == 11 and self.position and self.position.active_size > 0:  # Deactivate stop-loss
            if self.position.stop_loss_level is not None:
                self.position.stop_loss_level = None
                self.position.is_trailing_stop = False
                self.position.stop_loss_pct = None
                
                # Register event
                self.trade_events.append((
                    self.current_step, 
                    current_price, 
                    action, 
                    self.position.active_size, 
                    self.position.entry_price, 
                    None, 
                    self.position.take_profit_level,
                    self.position.exit_strategy
                ))
        
        elif action == 12 and self.position and self.position.active_size > 0:  # Deactivate take-profit
            if self.position.take_profit_level is not None:
                self.position.take_profit_level = None
                self.position.is_trailing_take = False
                self.position.take_profit_pct = None
                
                # Register event
                self.trade_events.append((
                    self.current_step, 
                    current_price, 
                    action, 
                    self.position.active_size, 
                    self.position.entry_price, 
                    self.position.stop_loss_level, 
                    None,
                    self.position.exit_strategy
                ))
        
        elif action == 13 and self.position and self.position.active_size > 0:  # Set partial exit strategy
            # Get market features for context
            market_features = self.get_market_features()
            
            # If we have agent with risk network, consult it
            if agent and hasattr(agent, 'get_risk_management_levels') and self.use_risk_management:
                risk_levels = agent.get_risk_management_levels(current_state, market_features)
                
                # Get partial exit levels and sizes
                partial_exit_levels = risk_levels.get('partial_exit_levels', [])
                partial_exit_sizes = risk_levels.get('partial_exit_sizes', [])
                
                # Set exit strategy to partial exits
                self.position.set_exit_strategy('partial', partial_exit_levels, partial_exit_sizes)
            else:
                # Use default partial exit strategy
                target = market_features.get('optimal_tp_pct', self.default_take_profit_pct)
                
                # Create 3 partial exit levels
                partial_exit_levels = [
                    target * 0.33,  # 1/3 of the way to target
                    target * 0.67,  # 2/3 of the way to target
                    target * 1.0    # Full target
                ]
                
                # Default sizes for each level
                partial_exit_sizes = [0.25, 0.35, 0.40]  # Exit 25%, then 35%, then 40%
                
                # Set exit strategy
                self.position.set_exit_strategy('partial', partial_exit_levels, partial_exit_sizes)
            
            # Register event
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                self.position.stop_loss_level, 
                self.position.take_profit_level,
                'partial'
            ))
        
        elif action == 14 and self.position and self.position.active_size > 0:  # Set trailing strategy
            # Enable trailing for both stop-loss and take-profit if they exist
            
            # Set trailing stop-loss
            if self.position.stop_loss_level is not None:
                self.position.is_trailing_stop = True
                self.position.trailing_stop_distance = self.default_trailing_pct
            else:
                # Create trailing stop if none exists
                self.position.set_stop_loss(percent=self.default_stop_loss_pct, is_trailing=True)
            
            # Set trailing take-profit
            if self.position.take_profit_level is not None:
                self.position.is_trailing_take = True
                self.position.trailing_take_distance = self.default_trailing_pct
            else:
                # Create trailing take-profit if none exists
                self.position.set_take_profit(percent=self.default_take_profit_pct, is_trailing=True)
                
            # Set exit strategy to trailing
            self.position.exit_strategy = 'trailing'
            
            # Register event
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                self.position.stop_loss_level, 
                self.position.take_profit_level,
                'trailing'
            ))
            
        elif action == 15 and self.position and self.position.active_size > 0:  # Adjust risk parameters based on market conditions
            # Get market features for context
            market_features = self.get_market_features()
            
            # If we have agent with risk network, consult it
            if agent and hasattr(agent, 'get_risk_management_levels') and self.use_risk_management:
                risk_levels = agent.get_risk_management_levels(current_state, market_features)
                
                # Update stop-loss and take-profit based on agent
                if self.position.stop_loss_level is not None:
                    self.position.set_stop_loss(percent=risk_levels['stop_loss_pct'], is_trailing=risk_levels['is_trailing_stop'])
                
                if self.position.take_profit_level is not None:
                    self.position.set_take_profit(percent=risk_levels['take_profit_pct'], is_trailing=risk_levels['is_trailing_take'])
                
                # Update exit strategy
                self.position.set_exit_strategy(
                    strategy=risk_levels['exit_strategy'],
                    levels=risk_levels.get('partial_exit_levels', []),
                    sizes=risk_levels.get('partial_exit_sizes', [])
                )
            else:
                # Use dynamic risk parameters based on market conditions
                risk_params = calculate_dynamic_risk_parameters(
                    price_data=self.df,
                    current_idx=self.current_step,
                    lookahead=20,
                    market_regime=["unknown", "trending", "ranging", "volatile"][self.market_regime]
                )
                
                # Update risk parameters
                if self.position.stop_loss_level is not None:
                    self.position.set_stop_loss(percent=risk_params['stop_loss_pct'], is_trailing=risk_params['is_trailing_stop'])
                
                if self.position.take_profit_level is not None:
                    self.position.set_take_profit(percent=risk_params['take_profit_pct'], is_trailing=risk_params['is_trailing_take'])
                
                # Update exit strategy
                self.position.set_exit_strategy(
                    strategy=risk_params['exit_strategy'],
                    levels=risk_params.get('partial_exit_levels', []),
                    sizes=risk_params.get('partial_exit_sizes', [])
                )
            
            # Register event
            self.trade_events.append((
                self.current_step, 
                current_price, 
                action, 
                self.position.active_size, 
                self.position.entry_price, 
                self.position.stop_loss_level, 
                self.position.take_profit_level,
                self.position.exit_strategy
            ))

    def check_exit_conditions(self, current_price):
        """
        Check if stop-loss, take-profit, or partial exit conditions have been met
        
        Args:
            current_price: Current market price
            
        Returns:
            tuple: (triggered, reason, size_to_exit)
        """
        if not self.position or self.position.active_size <= 0:
            return False, "", 0.0
        
        # Let position check its exit conditions
        triggered, reason, size_to_exit = self.position.check_exit_conditions(current_price)
        
        if triggered:
            # Store the active_size before potentially modifying the position
            active_size = self.position.active_size
            
            # Handle exit
            if reason.startswith('partial_exit'):
                # Handle partial exit
                part_size = size_to_exit
                
                # Update cash
                self.cash += current_price * part_size
                
                # Update position
                self.position.active_size -= part_size
                self.owned = self.position.active_size  # Update owned for backward compatibility
                
                # Register the partial exit
                self.position.closed_parts.append((part_size, current_price, self.current_step))
                
                # Register event
                self.trade_events.append((
                    self.current_step, 
                    current_price, 
                    16,  # Custom code for partial exit
                    self.position.active_size, 
                    self.position.entry_price, 
                    self.position.stop_loss_level, 
                    self.position.take_profit_level,
                    self.position.exit_strategy
                ))
                
                # Don't close the entire position, only the partial exit
                return True, reason, size_to_exit
            else:
                # Handle full exit (stop-loss or take-profit)
                sell_value = current_price * active_size
                
                # Close position and save trade information
                trade_info = self.position.close_position(current_price, self.current_step, reason)
                
                # Add entry_state and entry_action if available
                if hasattr(self.position, 'entry_state') and self.position.entry_state is not None:
                    trade_info['entry_state'] = self.position.entry_state
                    trade_info['entry_action'] = self.position.entry_action
                    
                    # Add trade pattern if profitable
                    profit = trade_info.get('profit_loss', 0.0)
                    if profit > 0:
                        self.add_successful_pattern(
                            self.position.entry_state, 
                            self.position.entry_action if self.position.entry_action is not None else 1,
                            profit
                        )
                
                self.completed_trades.append(trade_info)
                
                # Update Kelly for backward compatibility
                if self.active_trade is not None:
                    profit_loss = (current_price - self.active_trade['entry_price']) / self.active_trade['entry_price']
                    is_win = profit_loss > 0
                    
                    trade_result = {
                        'entry_price': self.active_trade['entry_price'],
                        'exit_price': current_price,
                        'profit_loss': profit_loss,
                        'is_win': is_win,
                        'position_size': self.active_trade['position_size']
                    }
                    
                    self.kelly_sizer.add_trade_result(trade_result)
                    self.active_trade = None
                
                # Update cash
                self.cash += sell_value
                
                # Store the action code before resetting position
                action_code = 17 if reason == "stop_loss" else 18
                
                # Reset position and owned
                self.owned = 0.0
                self.position = None
                
                # Register event - use action code 17 for stop-loss and 18 for take-profit
                self.trade_events.append((self.current_step, current_price, action_code, 0.0, 0.0, None, None, None))
                
                return True, reason, active_size
        
        return False, "", 0.0

    def step(self, action, agent=None):
        """
        Take a step in the environment
        
        Args:
            action: Action to execute
            agent: Optional agent instance for risk management consultation
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        done = False
        current_price = self.prices[self.current_step]
        portfolio_before = self.portfolio_value()
        auto_exit_executed = False
        exit_reason = ""
        size_exited = 0.0
        
        # Save state before taking action for Success Pattern Reinforcement
        try:
            current_state = self._get_state()
        except Exception as e:
            print(f"Error getting state: {e}")
            current_state = None
        
        # 1. First, check if we have an active position that should be closed
        #    due to stop-loss, take-profit, or partial exit
        if self.use_risk_management and self.position and self.position.active_size > 0:
            auto_exit_executed, exit_reason, size_exited = self.check_exit_conditions(current_price)
        
        # 2. Execute agent's action (if no automatic exit was executed or it was just a partial exit)
        if not auto_exit_executed or exit_reason.startswith('partial_exit'):
            self.execute_action(action, current_price, current_state, agent)
        
        # 3. Go to next step
        self.current_step += 1
        
        # 4. At the end of episode, close all open positions
        if self.current_step >= self.n_steps:
            done = True
            if self.position and self.position.active_size > 0:
                final_price = self.prices[-1]
                
                # Close position and save trade information
                trade_info = self.position.close_position(final_price, self.current_step)
                
                # Add entry_state and entry_action if available
                if hasattr(self.position, 'entry_state') and self.position.entry_state is not None:
                    trade_info['entry_state'] = self.position.entry_state
                    trade_info['entry_action'] = self.position.entry_action
                    
                    # Add trade pattern if profitable
                    profit = trade_info.get('profit_loss', 0.0)
                    if profit > 0:
                        self.add_successful_pattern(
                            self.position.entry_state, 
                            self.position.entry_action if self.position.entry_action is not None else 1,
                            profit
                        )
                
                self.completed_trades.append(trade_info)
                
                # Update Kelly for backward compatibility
                if self.active_trade is not None:
                    profit_loss = (final_price - self.active_trade['entry_price']) / self.active_trade['entry_price']
                    is_win = profit_loss > 0
                    
                    trade_result = {
                        'entry_price': self.active_trade['entry_price'],
                        'exit_price': final_price,
                        'profit_loss': profit_loss,
                        'is_win': is_win,
                        'position_size': self.active_trade['position_size']
                    }
                    
                    self.kelly_sizer.add_trade_result(trade_result)
                    self.active_trade = None
                
                # Update cash
                sell_value = final_price * self.position.active_size
                self.cash += sell_value
                
                # Reset position and owned
                self.owned = 0.0
                self.position = None
                
                # Register event
                self.trade_events.append((self.n_steps - 1, final_price, 2, 0.0, 0.0, None, None, None))
        
        # 5. Calculate reward - ENHANCED REWARD FUNCTION
        portfolio_after = self.portfolio_value()
        
        # Update peak portfolio value for drawdown calculation
        if portfolio_after > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_after
        
        # Calculate return for this step
        if portfolio_before > 0:
            step_return = (portfolio_after - portfolio_before) / portfolio_before
            self.returns.append(step_return)
        else:
            step_return = 0
            self.returns.append(step_return)
        
        self.portfolio_values.append(portfolio_after)
        
        # Calculate risk-adjusted metrics
        current_sharpe = self.calculate_sharpe_ratio(annualized=False)
        current_sortino = self.calculate_sortino_ratio(annualized=False)
        
        # ENHANCED REWARD CALCULATION
        
        # Calculate return over a window instead of just the last step 
        if len(self.portfolio_values) > 10:  # Use a 10-step window 
            window_start_value = self.portfolio_values[-10] 
            direct_return = ((portfolio_after - window_start_value) / window_start_value) * 100.0 
        else: 
            direct_return = ((portfolio_after - self.initial_cash) / self.initial_cash) * 100.0
        
        # 2. Sharpe ratio improvement - reward consistency and risk-adjusted performance
        sharpe_change = current_sharpe - self.previous_sharpe
        self.previous_sharpe = current_sharpe
        
        # 3. Calculate drawdown penalty - discourage large drawdowns
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - portfolio_after) / self.peak_portfolio_value
            drawdown_penalty = -drawdown * 2.0
        else:
            drawdown = 0.0
            drawdown_penalty = 0.0
        
        # 4. Calculate regime-appropriate reward - encourage strategy adaptation
        regime_reward = 0.0
        
        if len(self.returns) > self.window_size:
            # Get market regime
            market_regime = self.market_regime
            
            # Check if we have an active position
            has_position = self.position is not None and self.position.active_size > 0
            
            # Reward for appropriate strategy in trending markets
            if market_regime == 1:  # Trending
                if has_position:
                    # Check if we have a trailing strategy
                    if self.position.exit_strategy == 'trailing' or self.position.is_trailing_stop:
                        # Reward for using trailing in trends
                        regime_reward += 0.2
                    
                    # Additional reward if profitable
                    if self.profit_pct > 0:
                        regime_reward += 0.1
            
            # Reward for appropriate strategy in ranging markets
            elif market_regime == 2:  # Ranging
                if has_position:
                    # Check if we have a partial exit strategy
                    if self.position.exit_strategy == 'partial':
                        # Reward for using partial exits in ranges
                        regime_reward += 0.2
                    
                    # Reward for taking profits in ranges
                    if action in [8, 9, 10]:  # Partial sells
                        regime_reward += 0.2
            
            # Reward for appropriate strategy in volatile markets
            elif market_regime == 3:  # Volatile
                if not has_position:
                    # Reward for staying out of volatile markets
                    regime_reward += 0.2
                elif action == 2:  # Full exit in volatile markets
                    regime_reward += 0.2
        
        # 5. Success pattern similarity reward - encourage learning from success
        pattern_reward = 0.0
        if current_state is not None:
            pattern_reward = self.calculate_success_pattern_similarity(current_state, action)
        
        # 6. Risk management quality reward - encourage proper risk management
        risk_reward = 0.0
        
        if self.position and self.position.active_size > 0:
            # Reward for setting appropriate stop-loss and take-profit
            has_stop_loss = self.position.stop_loss_level is not None
            has_take_profit = self.position.take_profit_level is not None
            
            if has_stop_loss and has_take_profit:
                # Calculate risk-reward ratio
                risk_reward_ratio = self.position.take_profit_pct / self.position.stop_loss_pct if self.position.stop_loss_pct > 0 else 0
                
                # Reward good risk-reward setups (>= 2.0)
                if risk_reward_ratio >= 2.0:
                    risk_reward += 0.2
                
                # Higher reward for better ratios
                if risk_reward_ratio >= 3.0:
                    risk_reward += 0.2
            
            # Penalty for no risk management
            if not has_stop_loss and not has_take_profit and self.use_risk_management:
                risk_reward -= 0.1
        
        # 7. Partial exit quality reward - for partial exit actions
        partial_exit_reward = 0.0
        if auto_exit_executed and exit_reason.startswith('partial_exit'):
            # Reward successful partial exit
            partial_exit_reward = 0.2
            
            # Higher reward if it's a larger profit
            if self.profit_pct > 0.05:  # >5% profit
                partial_exit_reward += 0.1
        
        # Combine reward components with appropriate weights
        # Weights should sum to 1.0 for proper scaling
        reward = (
            direct_return * self.reward_params.get('direct_return_weight', 0.7) +  # CHANGE THIS LINE
            sharpe_change * self.reward_params.get('sharpe_change_weight', 5.0) +
            drawdown_penalty * 0.1 +           # Small penalty for drawdowns
            regime_reward * 0.2 +              # Reward for regime-appropriate actions
            pattern_reward * 0.1 +             # Reward for similar successful patterns
            risk_reward * 0.1 +                # Risk management quality
            partial_exit_reward * 0.1          # Partial exit success
        )
        
        # Limit extreme values
        reward = np.clip(reward, -10.0, 10.0)
        
        # Handle NaN and Inf
        if np.isnan(reward) or np.isinf(reward):
            print(f"NaN or Inf reward detected at step {self.current_step}, resetting to 0.0")
            reward = 0.0
        
        # Separate risk management reward (continued support for this)
        risk_management_reward = 0.0
        
        if (auto_exit_executed or action in [4, 5, 6, 7, 13, 14, 15]) and self.position:
            # For auto exits or risk management actions
            entry_price = self.position.entry_price if hasattr(self.position, 'entry_price') else 0
            exit_price = current_price
            max_favorable_price = self.position.max_favorable_price if hasattr(self.position, 'max_favorable_price') else exit_price
            max_adverse_price = self.position.max_adverse_price if hasattr(self.position, 'max_adverse_price') else exit_price
            stop_loss_level = self.position.stop_loss_level if hasattr(self.position, 'stop_loss_level') else None
            take_profit_level = self.position.take_profit_level if hasattr(self.position, 'take_profit_level') else None
            is_stop_loss_exit = exit_reason == "stop_loss"
            is_take_profit_exit = exit_reason == "take_profit"
            
            # Get partial exits if any
            if self.position and hasattr(self.position, 'closed_parts') and self.position.closed_parts:
                # Convert from (size, price, time) to (price, size) format
                partial_exits = [(price, size) for size, price, _ in self.position.closed_parts]
            else:
                partial_exits = None
            
            # Calculate risk management reward with error handling
            try:
                risk_management_reward = calculate_risk_management_reward(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    max_favorable_price=max_favorable_price,
                    max_adverse_price=max_adverse_price,
                    stop_loss_level=stop_loss_level,
                    take_profit_level=take_profit_level,
                    is_stop_loss_exit=is_stop_loss_exit,
                    is_take_profit_exit=is_take_profit_exit,
                    partial_exits=partial_exits
                )
            except Exception as e:
                print(f"Error calculating risk management reward: {e}")
                risk_management_reward = 0.0
        
        # Store reward components for logging
        reward_components = {
            'direct_return': direct_return,
            'sharpe_change': sharpe_change,
            'drawdown_penalty': drawdown_penalty,
            'regime_reward': regime_reward,
            'pattern_reward': pattern_reward,
            'risk_reward': risk_reward,
            'partial_exit_reward': partial_exit_reward,
            'total': reward
        }
        
        # Log reward components periodically for debugging
        if self.current_step % 100 == 0 or abs(reward) > 5.0:
            comp_str = ", ".join([f"{k}={v:.2f}" for k,v in reward_components.items()])
            print(f"Step {self.current_step}, Reward breakdown: {comp_str}")
        
        # END OF ENHANCED REWARD CALCULATION
        
        self.prev_portfolio_value = portfolio_after
        next_state = self._get_state()
        
        # Calculate position_size for information
        position_size = self.position.active_size if self.position else 0.0
        
        # Add Kelly and risk management information to info dictionary
        kelly_fraction = self.kelly_sizer.calculate_kelly_fraction() if self.use_kelly else 0.0
        
        # Build info dictionary with detailed information
        info = {
            'portfolio_value': portfolio_after, 
            'price': current_price, 
            'trade_made': action in [1, 2, 3, 8, 9, 10] or auto_exit_executed,
            'sharpe': current_sharpe,
            'sortino': current_sortino,
            'sharpe_change': sharpe_change,
            'kelly_fraction': kelly_fraction,
            'win_rate': self.kelly_sizer.win_rate,
            'win_loss_ratio': self.kelly_sizer.win_loss_ratio,
            'position_size': position_size,
            'auto_exit_executed': auto_exit_executed,
            'exit_reason': exit_reason,
            'size_exited': size_exited,
            'risk_management_reward': risk_management_reward,
            'has_stop_loss': self.position and self.position.stop_loss_level is not None,
            'has_take_profit': self.position and self.position.take_profit_level is not None,
            'profit_pct': self.profit_pct,
            # Extra market info
            'market_regime': self.market_regime,
            'atr_percent': self.atr_percent,
            'trend_strength': self.trend_strength,
            # Reward components for diagnostics
            'direct_return': direct_return,
            'drawdown_penalty': drawdown_penalty,
            'regime_reward': regime_reward,
            'pattern_reward': pattern_reward,
            'risk_reward': risk_reward,
            'partial_exit_reward': partial_exit_reward,
            'drawdown': drawdown
        }
        
        # Add information about position if available
        if self.position:
            info.update({
                'entry_price': self.position.entry_price,
                'stop_loss_level': self.position.stop_loss_level,
                'take_profit_level': self.position.take_profit_level,
                'exit_strategy': self.position.exit_strategy,
                'max_runup': self.position.max_runup,
                'max_drawdown': self.position.max_drawdown,
                'holding_period': self.position.holding_period
            })
        
        # Update action probabilities for next step
        self.update_action_probabilities()
        
        return next_state, reward, done, info
        
    def close(self):
        """Close the environment and clean up resources"""
        # Clean up any resources
        pass

    def analyze_trading_performance(self):
        """
        Analyze trading performance after an episode
        
        Returns:
            dict: Performance metrics
        """
        # Get trades
        trades = self.completed_trades
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate win rate
        profits = [t['profit_loss'] for t in trades if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in trades if t['profit_loss'] <= 0]
        
        win_count = len(profits)
        loss_count = len(losses)
        total_trades = len(trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # Calculate average profit and loss
        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Calculate profit factor
        gross_profit = sum(profits)
        gross_loss = sum(losses) if losses else 0.0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = self.calculate_sharpe_ratio(annualized=True)
        sortino_ratio = self.calculate_sortino_ratio(annualized=True)
        
        # Calculate max drawdown
        if len(self.portfolio_values) > 1:
            peaks = np.maximum.accumulate(self.portfolio_values)
            drawdowns = (self.portfolio_values - peaks) / peaks
            max_drawdown = abs(min(drawdowns))
        else:
            max_drawdown = 0.0
        
        # Calculate stats by exit type
        sl_exits = [t for t in trades if t.get('is_stop_loss_exit', False)]
        tp_exits = [t for t in trades if t.get('is_take_profit_exit', False)]
        partial_exits_count = sum(len(t.get('closed_parts', [])) for t in trades)
        
        # Calculate exit strategy performance
        exit_strategy_stats = {}
        for strategy in ['standard', 'trailing', 'partial']:
            strategy_trades = [t for t in trades if t.get('exit_strategy') == strategy]
            if strategy_trades:
                strategy_profits = [t['profit_loss'] for t in strategy_trades if t['profit_loss'] > 0]
                strategy_win_rate = len(strategy_profits) / len(strategy_trades)
                exit_strategy_stats[strategy] = {
                    'count': len(strategy_trades),
                    'win_rate': strategy_win_rate,
                    'avg_profit': np.mean(strategy_profits) if strategy_profits else 0.0
                }
        
        # Return all metrics
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit * 100,  # Convert to percentage
            'avg_loss': avg_loss * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'stop_loss_exits': len(sl_exits),
            'take_profit_exits': len(tp_exits),
            'partial_exits': partial_exits_count,
            'exit_strategy_stats': exit_strategy_stats
        }