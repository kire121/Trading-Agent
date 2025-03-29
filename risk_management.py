import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import pickle
import os
import random

from logger_setup import logger

# Standard parameters that can be overridden by config.py
DEFAULT_RISK_PARAMS = {
    'learning_rate': 0.001,
    'min_stop_loss_pct': 0.01,  # 1%
    'max_stop_loss_pct': 0.15,  # 15%
    'min_take_profit_pct': 0.01,  # 1%
    'max_take_profit_pct': 0.30,  # 30%
    'min_trailing_pct': 0.005,  # 0.5%
    'max_trailing_pct': 0.10,  # 10%
    'stop_loss_adjustment_threshold': 0.01,  # Minimum price movement to adjust trailing stop-loss
    'model_update_frequency': 10,  # Update model every 10 steps
    'batch_size': 32,
    'memory_size': 10000,
    'max_partial_exits': 3,  # Maximum number of partial exit levels
    'reward_params': {
        'loss_prevention_weight': 0.8,   # Weight for preventing large losses
        'profit_capture_weight': 1.5,    # Weight for capturing profits
        'early_exit_penalty': 0.1,       # Penalty for exiting too early
        'stability_bonus': 0.2,          # Bonus for stable levels
        'optimal_exit_bonus': 0.5,       # Bonus for exiting near optimal point
        'risk_reward_bonus': 0.5,        # Bonus for favorable risk-reward ratio
        'partial_exit_bonus': 0.3        # Bonus for successful partial exits
    }
}

class RiskManagementMemory:
    """
    Memory structure for storing risk management experiences for training.
    Enhanced to store more detailed risk management information and partial exits.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def add(self, state, risk_params, reward, next_state, done, additional_info=None):
        """
        Add an experience to the buffer
        
        Args:
            state: Current state
            risk_params: Dictionary with risk parameters (stop_loss_pct, take_profit_pct, etc.)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            additional_info: Additional information about the experience
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Store experience with all risk parameters
        self.buffer[self.position] = (state, risk_params, reward, next_state, done, additional_info or {})
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of arrays for each component of the experience
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, risk_params_list, rewards, next_states, dones, additional_infos = zip(*batch)
        
        # Extract individual risk parameters
        stop_loss_pcts = np.array([rp.get('stop_loss_pct', 0) for rp in risk_params_list])
        take_profit_pcts = np.array([rp.get('take_profit_pct', 0) for rp in risk_params_list])
        trailing_stop_pcts = np.array([rp.get('trailing_stop_pct', 0) for rp in risk_params_list])
        is_trailing_stops = np.array([rp.get('is_trailing_stop', False) for rp in risk_params_list])
        is_trailing_takes = np.array([rp.get('is_trailing_take', False) for rp in risk_params_list])
        
        # Extract partial exit parameters if they exist
        partial_exit_pcts = []
        partial_exit_sizes = []
        for i in range(3):  # Up to 3 partial exits
            exit_pcts = np.array([rp.get(f'partial_exit_{i+1}_pct', 0) for rp in risk_params_list])
            exit_sizes = np.array([rp.get(f'partial_exit_{i+1}_size', 0) for rp in risk_params_list])
            partial_exit_pcts.append(exit_pcts)
            partial_exit_sizes.append(exit_sizes)
        
        return (
            np.array(states), 
            {
                'stop_loss_pct': stop_loss_pcts,
                'take_profit_pct': take_profit_pcts,
                'trailing_stop_pct': trailing_stop_pcts,
                'is_trailing_stop': is_trailing_stops,
                'is_trailing_take': is_trailing_takes,
                'partial_exit_pcts': partial_exit_pcts,
                'partial_exit_sizes': partial_exit_sizes
            },
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones),
            additional_infos
        )
    
    def __len__(self):
        return len(self.buffer)

class RiskManagementNetwork:
    """
    Neural network to predict optimal stop-loss, take-profit, and partial exit levels
    based on market conditions. Enhanced to output continuous values and support
    multiple exit strategies.
    """
    def __init__(self, state_size, **kwargs):
        self.state_size = state_size
        self.params = {**DEFAULT_RISK_PARAMS, **kwargs}
        self.learning_rate = self.params['learning_rate']
        self.min_stop_loss_pct = self.params['min_stop_loss_pct']
        self.max_stop_loss_pct = self.params['max_stop_loss_pct']
        self.min_take_profit_pct = self.params['min_take_profit_pct']
        self.max_take_profit_pct = self.params['max_take_profit_pct']
        self.min_trailing_pct = self.params.get('min_trailing_pct', 0.005)
        self.max_trailing_pct = self.params.get('max_trailing_pct', 0.10)
        self.max_partial_exits = self.params.get('max_partial_exits', 3)
        
        self.memory = RiskManagementMemory(capacity=self.params.get('memory_size', 10000))
        self.model = self._build_model()
        self.loss_history = []
        self.update_counter = 0
        
        # Track last few predictions for each strategy for smoothing
        self.prediction_history = {
            'stop_loss': [],
            'take_profit': [],
            'trailing': [],
            'partial_exits': []
        }
        self.history_max_len = 5  # Number of recent predictions to store
    
    def _build_model(self):
        """
        Build the neural network model for risk management
        Enhanced with multiple outputs for different risk parameters
        """
        # Shared layers for feature extraction
        inputs = Input(shape=(self.state_size,))
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Stop-loss branch
        stop_loss_branch = Dense(32, activation='relu')(x)
        stop_loss_output = Dense(1, activation='sigmoid', name='stop_loss')(stop_loss_branch)
        
        # Take-profit branch
        take_profit_branch = Dense(32, activation='relu')(x)
        take_profit_output = Dense(1, activation='sigmoid', name='take_profit')(take_profit_branch)
        
        # Strategy selection branch (which risk strategy to use)
        strategy_branch = Dense(32, activation='relu')(x)
        strategy_output = Dense(3, activation='softmax', name='strategy')(strategy_branch)
        # 3 options: [standard_stop_take, trailing_stop, partial_exits]
        
        # Trailing parameters branch
        trailing_branch = Dense(32, activation='relu')(x)
        trailing_stop_output = Dense(1, activation='sigmoid', name='trailing_stop_pct')(trailing_branch)
        is_trailing_stop = Dense(1, activation='sigmoid', name='is_trailing_stop')(trailing_branch)
        is_trailing_take = Dense(1, activation='sigmoid', name='is_trailing_take')(trailing_branch)
        
        # Partial exit parameters branch
        partial_exit_branch = Dense(32, activation='relu')(x)
        
        # Multiple partial exit levels and sizes
        partial_exit_outputs = []
        partial_exit_size_outputs = []
        
        for i in range(self.max_partial_exits):
            # Level (percentage from entry)
            level_output = Dense(1, activation='sigmoid', name=f'partial_exit_{i+1}_pct')(partial_exit_branch)
            partial_exit_outputs.append(level_output)
            
            # Size (percentage of position to close)
            size_output = Dense(1, activation='sigmoid', name=f'partial_exit_{i+1}_size')(partial_exit_branch)
            partial_exit_size_outputs.append(size_output)
        
        # Create model with all outputs
        model = Model(
            inputs=inputs, 
            outputs=[
                stop_loss_output, 
                take_profit_output,
                strategy_output,
                trailing_stop_output,
                is_trailing_stop,
                is_trailing_take,
                *partial_exit_outputs,
                *partial_exit_size_outputs
            ]
        )
        
        # Compile with appropriate losses
        losses = {
            'stop_loss': 'mse',
            'take_profit': 'mse',
            'strategy': 'categorical_crossentropy',
            'trailing_stop_pct': 'mse',
            'is_trailing_stop': 'binary_crossentropy',
            'is_trailing_take': 'binary_crossentropy'
        }
        
        # Add losses for partial exit outputs
        for i in range(self.max_partial_exits):
            losses[f'partial_exit_{i+1}_pct'] = 'mse'
            losses[f'partial_exit_{i+1}_size'] = 'mse'
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=losses,
            metrics=['mae']  # Mean absolute error for regression outputs
        )
        
        return model
    
    def _smooth_predictions(self, prediction_type, new_value):
        """
        Smooth predictions by averaging with recent history to reduce oscillations
        
        Args:
            prediction_type: Type of prediction to smooth
            new_value: New prediction value
        
        Returns:
            Smoothed prediction value
        """
        history = self.prediction_history.get(prediction_type, [])
        
        # Add new value to history
        history.append(new_value)
        
        # Keep only recent predictions
        if len(history) > self.history_max_len:
            history = history[-self.history_max_len:]
        
        # Update history
        self.prediction_history[prediction_type] = history
        
        # Return smoothed value (with higher weight for most recent value)
        if len(history) > 1:
            weights = np.linspace(0.5, 1.0, len(history))
            weighted_sum = np.sum(np.array(history) * weights)
            return weighted_sum / np.sum(weights)
        
        return new_value
    
    def predict_risk_levels(self, state, market_volatility=None, smooth=True):
        """
        Predict optimal stop-loss, take-profit, and exit strategy based on current market state.
        
        Args:
            state: Current market state
            market_volatility: Current market volatility (ATR percentage) if available
            smooth: Whether to smooth predictions with recent history
            
        Returns:
            Dictionary with risk management parameters:
            - stop_loss_pct: Stop-loss percentage
            - take_profit_pct: Take-profit percentage
            - exit_strategy: Recommended exit strategy ('standard', 'trailing', 'partial')
            - trailing_stop_pct: Trailing stop distance
            - is_trailing_stop: Whether to use trailing stop
            - is_trailing_take: Whether to use trailing take-profit
            - partial_exit_levels: List of partial exit levels as percentages
            - partial_exit_sizes: List of position sizes to exit at each level
        """
        state_reshaped = np.array(state).reshape(1, -1)
        predictions = self.model.predict(state_reshaped, verbose=0)
        
        # Extract predictions (unwrap from batch dimension)
        raw_stop_loss = predictions[0][0][0]  # stop_loss output
        raw_take_profit = predictions[1][0][0]  # take_profit output
        strategy_probs = predictions[2][0]  # strategy probabilities
        raw_trailing_stop_pct = predictions[3][0][0]  # trailing_stop_pct output
        raw_is_trailing_stop = predictions[4][0][0]  # is_trailing_stop output
        raw_is_trailing_take = predictions[5][0][0]  # is_trailing_take output
        
        # Extract partial exit levels and sizes
        partial_exit_levels = []
        partial_exit_sizes = []
        
        for i in range(self.max_partial_exits):
            # partial_exit_1_pct, partial_exit_2_pct, etc. outputs
            level_idx = 6 + i
            level = predictions[level_idx][0][0]
            partial_exit_levels.append(level)
            
            # partial_exit_1_size, partial_exit_2_size, etc. outputs
            size_idx = 6 + self.max_partial_exits + i
            size = predictions[size_idx][0][0]
            partial_exit_sizes.append(size)
        
        # Apply smoothing if enabled
        if smooth:
            raw_stop_loss = self._smooth_predictions('stop_loss', raw_stop_loss)
            raw_take_profit = self._smooth_predictions('take_profit', raw_take_profit)
            raw_trailing_stop_pct = self._smooth_predictions('trailing', raw_trailing_stop_pct)
            
            # Smooth the partial exit levels
            for i in range(len(partial_exit_levels)):
                partial_exit_levels[i] = self._smooth_predictions(f'partial_exit_{i+1}', partial_exit_levels[i])
                partial_exit_sizes[i] = self._smooth_predictions(f'partial_exit_size_{i+1}', partial_exit_sizes[i])
        
        # Scale stop-loss from [0,1] to [min, max]
        stop_loss_pct = (raw_stop_loss * (self.max_stop_loss_pct - self.min_stop_loss_pct) + 
                        self.min_stop_loss_pct)
        
        # Scale take-profit from [0,1] to [min, max]
        take_profit_pct = (raw_take_profit * 
                          (self.max_take_profit_pct - self.min_take_profit_pct) + 
                          self.min_take_profit_pct)
        
        # Scale trailing stop percentage from [0,1] to [min, max]
        trailing_stop_pct = (raw_trailing_stop_pct * 
                           (self.max_trailing_pct - self.min_trailing_pct) + 
                           self.min_trailing_pct)
        
        # Binary decisions for trailing stop/take
        is_trailing_stop = raw_is_trailing_stop >= 0.5
        is_trailing_take = raw_is_trailing_take >= 0.5
        
        # Determine exit strategy based on strategy probabilities
        strategy_idx = np.argmax(strategy_probs)
        exit_strategies = ['standard', 'trailing', 'partial']
        exit_strategy = exit_strategies[strategy_idx]
        
        # Scale partial exit levels
        scaled_partial_levels = []
        for level in partial_exit_levels:
            # Scale from [0,1] to percentage of take-profit target
            # This makes first exits closer to entry, later exits closer to take-profit
            scaled_level = level * take_profit_pct
            scaled_partial_levels.append(scaled_level)
        
        # Ensure partial exit levels are in ascending order
        scaled_partial_levels.sort()
        
        # Process partial exit sizes (must sum to <= 1.0)
        total_size = sum(partial_exit_sizes)
        if total_size > 0:
            # Normalize sizes to sum up to 0.9 (saving 10% for final exit)
            normalized_sizes = [size / total_size * 0.9 for size in partial_exit_sizes]
        else:
            # Default to equal distribution if all sizes are 0
            normalized_sizes = [0.3] * len(partial_exit_sizes)
        
        # Adjust stop-loss/take-profit based on market volatility if provided
        if market_volatility is not None:
            # If market is more volatile than usual, widen the stops
            avg_volatility = 0.015  # 1.5% daily ATR is typical
            volatility_ratio = market_volatility / avg_volatility
            
            # Adjust stop-loss and take-profit with guardrails
            stop_loss_pct = min(stop_loss_pct * volatility_ratio, self.max_stop_loss_pct)
            take_profit_pct = min(take_profit_pct * volatility_ratio, self.max_take_profit_pct)
        
        # Return all risk parameters in a structured dictionary
        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'exit_strategy': exit_strategy,
            'trailing_stop_pct': trailing_stop_pct,
            'is_trailing_stop': is_trailing_stop,
            'is_trailing_take': is_trailing_take,
            'partial_exit_levels': scaled_partial_levels,
            'partial_exit_sizes': normalized_sizes,
            'strategy_probs': strategy_probs.tolist()  # For debugging
        }
    
    def train(self, batch_size=None):
        """
        Train the risk management network using experiences from memory
        
        Args:
            batch_size: Batch size for training
        """
        if batch_size is None:
            batch_size = self.params['batch_size']
            
        if len(self.memory) < batch_size:
            return
        
        # Sample experiences from memory
        states, risk_params, rewards, next_states, dones, _ = self.memory.sample(batch_size)
        
        # Extract risk parameters
        stop_loss_pcts = risk_params['stop_loss_pct']
        take_profit_pcts = risk_params['take_profit_pct']
        trailing_stop_pcts = risk_params['trailing_stop_pct']
        is_trailing_stops = risk_params['is_trailing_stop']
        is_trailing_takes = risk_params['is_trailing_take']
        partial_exit_pcts = risk_params['partial_exit_pcts']
        partial_exit_sizes = risk_params['partial_exit_sizes']
        
        # Normalize values to [0,1] for the model
        normalized_stop_loss = ((stop_loss_pcts - self.min_stop_loss_pct) / 
                              (self.max_stop_loss_pct - self.min_stop_loss_pct))
        normalized_take_profit = ((take_profit_pcts - self.min_take_profit_pct) / 
                                (self.max_take_profit_pct - self.min_take_profit_pct))
        normalized_trailing = ((trailing_stop_pcts - self.min_trailing_pct) / 
                            (self.max_trailing_pct - self.min_trailing_pct))
        
        # Determine strategy labels based on risk parameters
        strategy_labels = np.zeros((batch_size, 3))  # [standard, trailing, partial]
        
        for i in range(batch_size):
            if any(partial_exit_sizes[j][i] > 0 for j in range(len(partial_exit_sizes))):
                strategy_labels[i, 2] = 1.0  # Partial exit strategy
            elif is_trailing_stops[i] or is_trailing_takes[i]:
                strategy_labels[i, 1] = 1.0  # Trailing strategy
            else:
                strategy_labels[i, 0] = 1.0  # Standard strategy
        
        # Prepare training inputs for partial exit parameters
        partial_exit_training_data = {}
        for i in range(self.max_partial_exits):
            if i < len(partial_exit_pcts):
                exit_pcts = partial_exit_pcts[i]
                # Normalize to [0,1] - we use take_profit_pct as the denominator
                # Since partial exits are fractions of the take-profit target
                normalized_exit_pcts = np.clip(exit_pcts / take_profit_pcts, 0, 1)
                partial_exit_training_data[f'partial_exit_{i+1}_pct'] = normalized_exit_pcts.reshape(-1, 1)
                
                exit_sizes = partial_exit_sizes[i]
                partial_exit_training_data[f'partial_exit_{i+1}_size'] = exit_sizes.reshape(-1, 1)
            else:
                # Default values for unused partial exits
                partial_exit_training_data[f'partial_exit_{i+1}_pct'] = np.zeros((batch_size, 1))
                partial_exit_training_data[f'partial_exit_{i+1}_size'] = np.zeros((batch_size, 1))
        
        # Combine all training targets
        training_targets = {
            'stop_loss': normalized_stop_loss.reshape(-1, 1),
            'take_profit': normalized_take_profit.reshape(-1, 1),
            'strategy': strategy_labels,
            'trailing_stop_pct': normalized_trailing.reshape(-1, 1),
            'is_trailing_stop': is_trailing_stops.reshape(-1, 1).astype(int),
            'is_trailing_take': is_trailing_takes.reshape(-1, 1).astype(int),
            **partial_exit_training_data
        }
        
        # Train the model with sample_weight based on rewards
        sample_weights = np.exp(np.clip(rewards / 3.0, -1.0, 2.0))  # Convert rewards to weights
        sample_weights = sample_weights / np.mean(sample_weights)  # Normalize weights
        
        history = self.model.fit(
            states, 
            training_targets,
            sample_weight=sample_weights,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )
        
        # Store loss history
        if 'loss' in history.history:
            self.loss_history.append(history.history['loss'][0])
    
    def update(self, state, risk_params, reward, next_state, done, additional_info=None):
        """
        Update risk management network with a new experience
        
        Args:
            state: Current state
            risk_params: Dictionary with risk parameters
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            additional_info: Additional information about the experience
        """
        # Clip reward to reasonable values before storing in memory
        reward = np.clip(reward, -3, 7)
        
        # Add experience to memory
        self.memory.add(state, risk_params, reward, next_state, done, additional_info)
        
        # Train model periodically
        self.update_counter += 1
        if self.update_counter >= self.params['model_update_frequency']:
            self.train()
            self.update_counter = 0
    
    def save(self, filepath):
        """Save the risk management model and its state"""
        model_path = f"{filepath}_risk_mgmt.keras"
        try:
            self.model.save(model_path)
            
            # Save parameters and state
            state_path = f"{filepath}_risk_mgmt_state.pkl"
            with open(state_path, 'wb') as f:
                state = {
                    'params': self.params,
                    'loss_history': self.loss_history,
                    'min_stop_loss_pct': self.min_stop_loss_pct,
                    'max_stop_loss_pct': self.max_stop_loss_pct,
                    'min_take_profit_pct': self.min_take_profit_pct,
                    'max_take_profit_pct': self.max_take_profit_pct,
                    'min_trailing_pct': self.min_trailing_pct,
                    'max_trailing_pct': self.max_trailing_pct,
                    'prediction_history': self.prediction_history
                }
                pickle.dump(state, f)
                
            logger.info(f"Risk management model and state saved to {model_path} and {state_path}")
            
        except Exception as e:
            logger.error(f"Error saving risk management model: {e}")
    
    def load(self, filepath):
        """Load the risk management model and its state"""
        model_path = f"{filepath}_risk_mgmt.keras"
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                
                # Load parameters and state
                state_path = f"{filepath}_risk_mgmt_state.pkl"
                if os.path.exists(state_path):
                    with open(state_path, 'rb') as f:
                        state = pickle.load(f)
                        self.params = state.get('params', self.params)
                        self.loss_history = state.get('loss_history', [])
                        self.min_stop_loss_pct = state.get('min_stop_loss_pct', self.min_stop_loss_pct)
                        self.max_stop_loss_pct = state.get('max_stop_loss_pct', self.max_stop_loss_pct)
                        self.min_take_profit_pct = state.get('min_take_profit_pct', self.min_take_profit_pct)
                        self.max_take_profit_pct = state.get('max_take_profit_pct', self.max_take_profit_pct)
                        self.min_trailing_pct = state.get('min_trailing_pct', self.min_trailing_pct)
                        self.max_trailing_pct = state.get('max_trailing_pct', self.max_trailing_pct)
                        self.prediction_history = state.get('prediction_history', self.prediction_history)
                
                logger.info(f"Risk management model and state loaded from {model_path}")
                
            except Exception as e:
                logger.error(f"Error loading risk management model: {e}")
                raise
        else:
            logger.warning(f"Could not find risk management model at {model_path}, using new model")


# Enhanced Position class to handle multiple exit strategies and partial closes
class Position:
    """
    A class to manage a trading position and its risk management.
    Enhanced to support multiple take-profit targets and partial position exits.
    """
    def __init__(self, entry_price, size, is_long=True):
        # Basic position attributes
        self.entry_price = entry_price
        self.size = size
        self.is_long = is_long
        self.entry_time = None
        self.exit_price = None
        self.exit_time = None
        
        # Risk management parameters
        self.stop_loss_level = None
        self.take_profit_level = None
        self.is_trailing_stop = False
        self.is_trailing_take = False
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.trailing_stop_distance = None
        self.trailing_take_distance = None
        
        # For partial closing
        self.closed_parts = []  # List of closed parts: (size, price, time)
        self.active_size = size  # Active position after partial closings
        
        # For performance analysis
        self.max_favorable_price = entry_price
        self.max_adverse_price = entry_price
        self.is_stop_loss_exit = False
        self.is_take_profit_exit = False
        
        # To track Risk/Reward
        self.stop_loss_pct = None
        self.take_profit_pct = None
        
        # NEW FIELDS FOR ENHANCED RISK MANAGEMENT
        self.exit_strategy = 'standard'  # 'standard', 'trailing', or 'partial'
        
        # For partial exit strategy
        self.partial_exit_levels = []  # List of price levels for partial exits
        self.partial_exit_sizes = []   # Corresponding sizes to exit at each level
        self.executed_partial_exits = []  # Track which partial exits have been executed
        
        # For dynamic adjustment based on market conditions
        self.market_conditions = {}
        self.atr_at_entry = None
        
        # For trailing stop/take-profit with acceleration
        self.trailing_acceleration = 0.0  # Acceleration factor for trailing stops
        self.trailing_step_count = 0      # Count steps for acceleration
        
        # For trade management metrics
        self.max_runup = 0.0  # Maximum unrealized profit
        self.max_drawdown = 0.0  # Maximum unrealized loss
        self.holding_period = 0  # Number of bars position held
        
        # For state-based risk management
        self.entry_state = None  # For storing the state at entry
        self.entry_action = None  # For storing which action created the position
        self.atr_percent = None  # ATR as percentage of price at entry
        
        # For trade evaluation
        self.win_loss = None  # 1 for win, 0 for loss
        self.risk_reward_realized = None  # Realized risk/reward ratio
        self.expected_value = None  # Expected value of the trade
    
    def set_stop_loss(self, level=None, percent=None, is_trailing=False, atr_multiple=None):
        """
        Set stop-loss for the position.
        
        Args:
            level: Direct price level
            percent: Percentage from entry (e.g., 0.05 for 5%)
            is_trailing: Whether this is a trailing stop-loss
            atr_multiple: Multiple of ATR for dynamic stop-loss (if atr_at_entry is set)
        """
        if level is not None:
            self.stop_loss_level = level
        elif percent is not None:
            self.stop_loss_pct = percent
            self.stop_loss_level = calculate_stop_loss_level(self.entry_price, percent, self.is_long)
            if is_trailing:
                self.trailing_stop_distance = percent
        elif atr_multiple is not None and self.atr_at_entry is not None:
            # Set stop-loss based on ATR
            stop_distance = atr_multiple * self.atr_at_entry
            self.stop_loss_pct = stop_distance / self.entry_price
            self.stop_loss_level = calculate_stop_loss_level(self.entry_price, self.stop_loss_pct, self.is_long)
            if is_trailing:
                self.trailing_stop_distance = self.stop_loss_pct
        
        self.is_trailing_stop = is_trailing
    
    def set_take_profit(self, level=None, percent=None, is_trailing=False, atr_multiple=None):
        """
        Set take-profit for the position.
        
        Args:
            level: Direct price level
            percent: Percentage from entry (e.g., 0.10 for 10%)
            is_trailing: Whether this is a trailing take-profit
            atr_multiple: Multiple of ATR for dynamic take-profit (if atr_at_entry is set)
        """
        if level is not None:
            self.take_profit_level = level
        elif percent is not None:
            self.take_profit_pct = percent
            self.take_profit_level = calculate_take_profit_level(self.entry_price, percent, self.is_long)
            if is_trailing:
                self.trailing_take_distance = percent
        elif atr_multiple is not None and self.atr_at_entry is not None:
            # Set take-profit based on ATR
            take_distance = atr_multiple * self.atr_at_entry
            self.take_profit_pct = take_distance / self.entry_price
            self.take_profit_level = calculate_take_profit_level(self.entry_price, self.take_profit_pct, self.is_long)
            if is_trailing:
                self.trailing_take_distance = self.take_profit_pct
        
        self.is_trailing_take = is_trailing
    
    def set_exit_strategy(self, strategy, levels=None, sizes=None):
        """
        Set the exit strategy for the position
        
        Args:
            strategy: 'standard', 'trailing', or 'partial'
            levels: List of price levels for partial exits (as percentage from entry)
            sizes: List of position sizes to exit at each level
        """
        self.exit_strategy = strategy
        
        if strategy == 'trailing':
            # Set both stop-loss and take-profit to trailing if they exist
            if self.stop_loss_level is not None:
                self.is_trailing_stop = True
            if self.take_profit_level is not None:
                self.is_trailing_take = True
        
        elif strategy == 'partial' and levels is not None and sizes is not None:
            # Set up partial exit levels
            self.partial_exit_levels = []
            self.partial_exit_sizes = []
            self.executed_partial_exits = [False] * len(levels)
            
            # Calculate actual price levels from percentages
            for i, pct in enumerate(levels):
                if pct > 0:  # Ignore zero or negative percentages
                    price_level = self.entry_price * (1 + pct) if self.is_long else self.entry_price * (1 - pct)
                    self.partial_exit_levels.append(price_level)
                    self.partial_exit_sizes.append(sizes[i])
    
    def set_market_context(self, atr=None, atr_percent=None, **kwargs):
        """
        Set market context information for dynamic risk management
        
        Args:
            atr: ATR value at entry
            atr_percent: ATR as percentage of price
            **kwargs: Additional market context parameters
        """
        if atr is not None:
            self.atr_at_entry = atr
        
        if atr_percent is not None:
            self.atr_percent = atr_percent
        
        # Store any additional market context
        self.market_conditions.update(kwargs)
    
    def update_price_extremes(self, price):
        """
        Update highest/lowest prices and trailing levels
        
        Args:
            price: Current price
        """
        self.holding_period += 1  # Increment holding period
        
        if self.is_long:
            # For long position, track highest price for trailing stop-loss
            if price > self.highest_price:
                self.highest_price = price
                
                # Update maximum runup (unrealized profit)
                current_runup = (price - self.entry_price) / self.entry_price
                self.max_runup = max(self.max_runup, current_runup)
                
                # Update trailing stop-loss if enabled
                if self.is_trailing_stop and self.trailing_stop_distance is not None:
                    # Calculate dynamic trailing distance with acceleration
                    trailing_distance = self.trailing_stop_distance
                    
                    # Apply acceleration if price continues moving favorably
                    self.trailing_step_count += 1
                    if self.trailing_step_count >= 3:  # After 3 consecutive favorable moves
                        # Accelerate trailing (tighten it)
                        acceleration = min(0.1, self.trailing_acceleration + 0.01)
                        self.trailing_acceleration = acceleration
                        trailing_distance = trailing_distance * (1.0 - acceleration)
                    
                    new_stop = update_trailing_stop(price, self.highest_price, trailing_distance, self.is_long)
                    self.stop_loss_level = max(self.stop_loss_level, new_stop) if self.stop_loss_level else new_stop
            else:
                # Reset acceleration if price doesn't make new high
                self.trailing_step_count = 0
                self.trailing_acceleration = 0.0
            
            # For long position, track lowest price for trailing take-profit
            if price < self.lowest_price:
                self.lowest_price = price
                
                # Update maximum drawdown (unrealized loss)
                current_drawdown = (price - self.entry_price) / self.entry_price
                self.max_drawdown = min(self.max_drawdown, current_drawdown)
                
                # Update trailing take-profit if enabled
                if self.is_trailing_take and self.trailing_take_distance is not None:
                    new_take = update_trailing_take_profit(price, self.lowest_price, self.trailing_take_distance, self.is_long)
                    self.take_profit_level = min(self.take_profit_level, new_take) if self.take_profit_level else new_take
        else:
            # For short position, track lowest price for trailing stop-loss
            if price < self.lowest_price:
                self.lowest_price = price
                
                # Update maximum runup (unrealized profit)
                current_runup = (self.entry_price - price) / self.entry_price
                self.max_runup = max(self.max_runup, current_runup)
                
                # Update trailing stop-loss if enabled
                if self.is_trailing_stop and self.trailing_stop_distance is not None:
                    # Calculate dynamic trailing distance with acceleration
                    trailing_distance = self.trailing_stop_distance
                    
                    # Apply acceleration if price continues moving favorably
                    self.trailing_step_count += 1
                    if self.trailing_step_count >= 3:  # After 3 consecutive favorable moves
                        # Accelerate trailing (tighten it)
                        acceleration = min(0.1, self.trailing_acceleration + 0.01)
                        self.trailing_acceleration = acceleration
                        trailing_distance = trailing_distance * (1.0 - acceleration)
                    
                    new_stop = update_trailing_stop(price, self.lowest_price, trailing_distance, self.is_long)
                    self.stop_loss_level = min(self.stop_loss_level, new_stop) if self.stop_loss_level else new_stop
            else:
                # Reset acceleration if price doesn't make new low
                self.trailing_step_count = 0
                self.trailing_acceleration = 0.0
            
            # For short position, track highest price for trailing take-profit
            if price > self.highest_price:
                self.highest_price = price
                
                # Update maximum drawdown (unrealized loss)
                current_drawdown = (self.entry_price - price) / self.entry_price
                self.max_drawdown = min(self.max_drawdown, current_drawdown)
                
                # Update trailing take-profit if enabled
                if self.is_trailing_take and self.trailing_take_distance is not None:
                    new_take = update_trailing_take_profit(price, self.highest_price, self.trailing_take_distance, self.is_long)
                    self.take_profit_level = max(self.take_profit_level, new_take) if self.take_profit_level else new_take
        
        # Update max favorable/adverse prices for performance analysis
        if self.is_long:
            if price > self.max_favorable_price:
                self.max_favorable_price = price
            if price < self.max_adverse_price:
                self.max_adverse_price = price
        else:
            if price < self.max_favorable_price:
                self.max_favorable_price = price
            if price > self.max_adverse_price:
                self.max_adverse_price = price
    
    def check_exit_conditions(self, price):
        """
        Check if stop-loss, take-profit, or partial exit conditions have been triggered.
        
        Args:
            price: Current price
            
        Returns:
            tuple: (triggered, reason, size_to_exit)
        """
        # First update trailing levels
        self.update_price_extremes(price)
        
        # Check stop-loss
        if self.stop_loss_level is not None:
            if check_stop_loss_triggered(price, self.stop_loss_level, self.is_long):
                self.is_stop_loss_exit = True
                return True, "stop_loss", self.active_size  # Exit full position
        
        # Check take-profit
        if self.take_profit_level is not None:
            if check_take_profit_triggered(price, self.take_profit_level, self.is_long):
                self.is_take_profit_exit = True
                return True, "take_profit", self.active_size  # Exit full position
        
        # Check partial exit levels
        if self.exit_strategy == 'partial' and self.partial_exit_levels:
            for i, (level, size_fraction) in enumerate(zip(self.partial_exit_levels, self.partial_exit_sizes)):
                if not self.executed_partial_exits[i]:  # Check only if this level hasn't been executed
                    # Check if price has reached the partial exit level
                    if ((self.is_long and price >= level) or 
                        (not self.is_long and price <= level)):
                        # Mark this level as executed
                        self.executed_partial_exits[i] = True
                        # Calculate size to exit
                        size_to_exit = self.size * size_fraction
                        size_to_exit = min(size_to_exit, self.active_size)  # Ensure we don't exit more than active size
                        return True, f"partial_exit_{i+1}", size_to_exit
        
        return False, None, 0
    
    def close_position(self, price, time=None, reason=None):
        """
        Close the entire position.
        
        Args:
            price: Price at which position was closed
            time: Time of closing
            reason: Reason for closing
            
        Returns:
            dict: Information about the closed position
        """
        self.exit_price = price
        self.exit_time = time
        
        if reason == "stop_loss":
            self.is_stop_loss_exit = True
        elif reason == "take_profit":
            self.is_take_profit_exit = True
        
        # Calculate trade metrics
        if self.entry_price > 0:
            profit_loss = (price - self.entry_price) / self.entry_price if self.is_long else (self.entry_price - price) / self.entry_price
            self.win_loss = 1 if profit_loss > 0 else 0
            
            # Calculate realized risk/reward ratio
            if self.stop_loss_pct is not None and self.stop_loss_pct > 0:
                self.risk_reward_realized = abs(profit_loss / self.stop_loss_pct)
            
            # Calculate expected value (win rate * avg win - loss rate * avg loss)
            # We use the actual outcome of this trade for a simple estimate
            win_rate = 1.0 if profit_loss > 0 else 0.0
            avg_win = profit_loss if profit_loss > 0 else 0.0
            loss_rate = 1.0 if profit_loss <= 0 else 0.0
            avg_loss = abs(profit_loss) if profit_loss <= 0 else 0.0
            self.expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        return self.get_trade_info()
    
    def close_partial(self, size_fraction, price, time=None):
        """
        Close a portion of the position.
        
        Args:
            size_fraction: Fraction of position size to close (0.25, 0.5, etc.)
            price: Price at closing
            time: Time of closing
            
        Returns:
            float: Amount closed
        """
        if size_fraction <= 0 or size_fraction > 1:
            return 0
        
        size_to_close = self.active_size * size_fraction
        if size_to_close > 0:
            self.closed_parts.append((size_to_close, price, time))
            self.active_size -= size_to_close
            return size_to_close
        
        return 0
    
    def get_average_exit_price(self):
        """
        Calculate average exit price for the entire position.
        
        Returns:
            float: Average exit price
        """
        if self.active_size <= 0 and self.exit_price is not None:
            # Entire position is closed
            total_value = sum(size * price for size, price, _ in self.closed_parts)
            total_value += (self.size - self.active_size - sum(size for size, _, _ in self.closed_parts)) * self.exit_price
            return total_value / self.size
        elif self.closed_parts:
            # Only parts are closed
            total_closed_size = sum(size for size, _, _ in self.closed_parts)
            total_value = sum(size * price for size, price, _ in self.closed_parts)
            return total_value / total_closed_size
        
        return None
    
    def get_trade_info(self):
        """
        Create a summary of trade information.
        
        Returns:
            dict: Information about the trade
        """
        trade_info = {
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'size': self.size,
            'is_long': self.is_long,
            'avg_exit_price': self.get_average_exit_price() or self.exit_price,
            'stop_loss_level': self.stop_loss_level,
            'take_profit_level': self.take_profit_level,
            'is_trailing_stop': self.is_trailing_stop,
            'is_trailing_take': self.is_trailing_take,
            'max_favorable_price': self.max_favorable_price,
            'max_adverse_price': self.max_adverse_price,
            'is_stop_loss_exit': self.is_stop_loss_exit,
            'is_take_profit_exit': self.is_take_profit_exit,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'closed_parts': [(size, price) for size, price, _ in self.closed_parts],
            'profit_loss': (self.exit_price - self.entry_price) / self.entry_price if self.is_long and self.exit_price else None,
            'exit_strategy': self.exit_strategy,
            'holding_period': self.holding_period,
            'max_runup': self.max_runup,
            'max_drawdown': self.max_drawdown,
            'win_loss': self.win_loss,
            'risk_reward_realized': self.risk_reward_realized,
            'expected_value': self.expected_value,
            'atr_at_entry': self.atr_at_entry,
            'atr_percent': self.atr_percent
        }
        
        # Add entry_state and entry_action if available
        if hasattr(self, 'entry_state') and self.entry_state is not None:
            trade_info['entry_state'] = self.entry_state
        if hasattr(self, 'entry_action') and self.entry_action is not None:
            trade_info['entry_action'] = self.entry_action
            
        return trade_info


# Functions to calculate and apply stop-loss and take-profit
def calculate_stop_loss_level(entry_price, stop_loss_pct, is_long=True):
    """
    Calculate stop-loss price level based on entry price and percentage.
    
    Args:
        entry_price: Price where position was opened
        stop_loss_pct: Stop-loss percentage (0.05 = 5%)
        is_long: Whether position is long (True) or short (False)
        
    Returns:
        float: Price level for stop-loss
    """
    if is_long:
        return entry_price * (1 - stop_loss_pct)
    else:
        return entry_price * (1 + stop_loss_pct)

def calculate_take_profit_level(entry_price, take_profit_pct, is_long=True):
    """
    Calculate take-profit price level based on entry price and percentage.
    
    Args:
        entry_price: Price where position was opened
        take_profit_pct: Take-profit percentage (0.10 = 10%)
        is_long: Whether position is long (True) or short (False)
        
    Returns:
        float: Price level for take-profit
    """
    if is_long:
        return entry_price * (1 + take_profit_pct)
    else:
        return entry_price * (1 - take_profit_pct)

def update_trailing_stop(current_price, highest_price, trailing_distance, is_long=True):
    """
    Update trailing stop-loss based on highest/lowest price since entry.
    
    Args:
        current_price: Current price
        highest_price: Highest price since entry (for long) or lowest (for short)
        trailing_distance: Percentage to trail (0.05 = 5%)
        is_long: Whether position is long (True) or short (False)
        
    Returns:
        float: Updated stop-loss level
    """
    if is_long:
        return highest_price * (1 - trailing_distance)
    else:
        return highest_price * (1 + trailing_distance)

def update_trailing_take_profit(current_price, lowest_price, trailing_distance, is_long=True):
    """
    Update trailing take-profit based on lowest/highest price since entry.
    
    Args:
        current_price: Current price
        lowest_price: Lowest price since entry (for long) or highest (for short)
        trailing_distance: Percentage to trail (0.05 = 5%)
        is_long: Whether position is long (True) or short (False)
        
    Returns:
        float: Updated take-profit level
    """
    if is_long:
        return lowest_price * (1 + trailing_distance)
    else:
        return lowest_price * (1 - trailing_distance)

def check_stop_loss_triggered(current_price, stop_level, is_long=True):
    """
    Check if stop-loss has been triggered.
    
    Args:
        current_price: Current price
        stop_level: Stop-loss level
        is_long: Whether position is long (True) or short (False)
        
    Returns:
        bool: True if stop-loss was triggered, False otherwise
    """
    if is_long:
        return current_price <= stop_level
    else:
        return current_price >= stop_level

def check_take_profit_triggered(current_price, take_level, is_long=True):
    """
    Check if take-profit has been triggered.
    
    Args:
        current_price: Current price
        take_level: Take-profit level
        is_long: Whether position is long (True) or short (False)
        
    Returns:
        bool: True if take-profit was triggered, False otherwise
    """
    if is_long:
        return current_price >= take_level
    else:
        return current_price <= take_level

# Enhanced risk management reward functions
def calculate_risk_management_reward(
    entry_price: float, 
    exit_price: float, 
    max_favorable_price: float, 
    max_adverse_price: float,
    stop_loss_level: Optional[float], 
    take_profit_level: Optional[float],
    is_stop_loss_exit: bool,
    is_take_profit_exit: bool,
    partial_exits: Optional[List[Tuple[float, float]]] = None,
    params: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate specialized reward for risk management decisions with enhanced metrics
    including optimal exit timing and partial exit evaluation.
    
    Args:
        entry_price: Price where position opened
        exit_price: Price where position closed
        max_favorable_price: Most favorable price during position's lifetime
        max_adverse_price: Most adverse price during position's lifetime
        stop_loss_level: Set stop-loss level
        take_profit_level: Set take-profit level
        is_stop_loss_exit: Whether position was closed via stop-loss
        is_take_profit_exit: Whether position was closed via take-profit
        partial_exits: List of (price, size) tuples for partial exits
        params: Reward parameters
        
    Returns:
        float: Reward value for risk management
    """
    # Use default params if none provided
    if params is None:
        params = DEFAULT_RISK_PARAMS['reward_params']
    
    # Prevent division by zero
    if entry_price == 0:
        return 0.0
    
    try:
        # Actual P&L - Limit to reasonable values
        actual_pnl = (exit_price - entry_price) / entry_price
        # Clip extreme values to reasonable limits (-100% to +100%)
        actual_pnl = np.clip(actual_pnl, -1.0, 1.0)
        
        # Maximum potential P&L
        max_potential_pnl = (max_favorable_price - entry_price) / entry_price
        max_potential_pnl = np.clip(max_potential_pnl, 0.0, 1.0)  # Should be >= 0
        
        # Maximum potential loss
        max_potential_loss = (max_adverse_price - entry_price) / entry_price
        max_potential_loss = np.clip(max_potential_loss, -1.0, 1.0)
        
        # === REWARD COMPONENTS ===
        
        # 1. Profit capture ratio - what percentage of maximum profit was captured
        profit_capture_ratio = 0.0
        if max_potential_pnl > 0:
            profit_capture_ratio = min(actual_pnl / max_potential_pnl, 1.0)
            # Handle negative values (losses when there could have been profit)
            if profit_capture_ratio < 0:
                profit_capture_ratio = 0.0
        
        # Apply weight and limit value
        profit_capture_weight = params.get('profit_capture_weight', 1.5)
        profit_capture_reward = profit_capture_ratio * profit_capture_weight
        profit_capture_reward = np.clip(profit_capture_reward, 0.0, 3.0)
        
        # 2. Loss prevention ratio - how much potential loss was prevented
        loss_prevention_ratio = 0.0
        if max_potential_loss < 0 and actual_pnl > max_potential_loss:
            # Calculate ratio of prevented loss
            prevented_loss = actual_pnl - max_potential_loss
            max_preventable_loss = abs(max_potential_loss)
            if max_preventable_loss > 0:
                loss_prevention_ratio = min(prevented_loss / max_preventable_loss, 1.0)
        
        # Apply weight and limit value
        loss_prevention_weight = params.get('loss_prevention_weight', 0.8)
        loss_prevention_reward = loss_prevention_ratio * loss_prevention_weight
        loss_prevention_reward = np.clip(loss_prevention_reward, 0.0, 2.0)
        
        # 3. Early exit penalty - if exited too early and missed significant upside
        early_exit_penalty = 0.0
        if actual_pnl > 0 and not is_take_profit_exit and profit_capture_ratio < 0.5:
            early_exit_penalty_weight = params.get('early_exit_penalty', 0.1)
            early_exit_penalty = (1 - profit_capture_ratio) * early_exit_penalty_weight
            early_exit_penalty = np.clip(early_exit_penalty, 0.0, 1.0)
        
        # 4. Optimal exit timing bonus
        optimal_exit_bonus = 0.0
        if actual_pnl > 0:
            # If exit price is close to maximum favorable price
            optimal_exit_threshold = 0.8  # 80% of max favorable
            if profit_capture_ratio > optimal_exit_threshold:
                optimal_exit_weight = params.get('optimal_exit_bonus', 0.5)
                # Scale bonus based on how close to optimal exit
                exit_quality = (profit_capture_ratio - optimal_exit_threshold) / (1.0 - optimal_exit_threshold)
                optimal_exit_bonus = exit_quality * optimal_exit_weight
                optimal_exit_bonus = np.clip(optimal_exit_bonus, 0.0, optimal_exit_weight)
        
        # 5. Reward for good risk-reward ratio setup
        risk_reward_bonus = 0.0
        if stop_loss_level is not None and take_profit_level is not None and entry_price > 0:
            # Calculate the risk-reward ratio of the setup
            stop_loss_pct = abs((stop_loss_level - entry_price) / entry_price)
            take_profit_pct = abs((take_profit_level - entry_price) / entry_price)
            
            if stop_loss_pct > 0:
                risk_reward_ratio = take_profit_pct / stop_loss_pct
                # Bonus for good risk-reward setups (>= 2.0)
                if risk_reward_ratio >= 2.0:
                    risk_reward_weight = params.get('risk_reward_bonus', 0.5)
                    # Scale bonus based on how good the ratio is (up to 5.0)
                    ratio_quality = min((risk_reward_ratio - 2.0) / 3.0, 1.0)
                    risk_reward_bonus = ratio_quality * risk_reward_weight
        
        # 6. Partial exit evaluation
        partial_exit_bonus = 0.0
        if partial_exits:
            partial_exit_weight = params.get('partial_exit_bonus', 0.3)
            
            # Calculate weighted average exit quality
            total_size = sum(size for _, size in partial_exits)
            if total_size > 0:
                total_quality = 0
                for price, size in partial_exits:
                    # Calculate quality of this exit (higher if close to max favorable)
                    exit_pnl = (price - entry_price) / entry_price
                    if max_potential_pnl > 0:
                        exit_quality = exit_pnl / max_potential_pnl
                        exit_quality = np.clip(exit_quality, 0.0, 1.0)
                        total_quality += exit_quality * (size / total_size)
                
                partial_exit_bonus = total_quality * partial_exit_weight
        
        # 7. Stability bonus - reward for manual control
        stability_bonus = 0.0
        if not is_stop_loss_exit and not is_take_profit_exit:
            stability_bonus_weight = params.get('stability_bonus', 0.2)
            stability_bonus = stability_bonus_weight
        
        # === COMBINE REWARDS ===
        weighted_profit_capture = profit_capture_reward * 0.5      # 50% weight
        weighted_loss_prevention = loss_prevention_reward * 0.25   # 25% weight
        weighted_early_exit_penalty = early_exit_penalty * 0.05    # 5% weight
        weighted_optimal_exit = optimal_exit_bonus * 0.1           # 10% weight
        weighted_risk_reward = risk_reward_bonus * 0.05            # 5% weight
        weighted_partial_exit = partial_exit_bonus * 0.1           # 10% weight
        weighted_stability = stability_bonus * 0.05                # 5% weight
        
        # Calculate total reward
        total_reward = (
            weighted_profit_capture + 
            weighted_loss_prevention - 
            weighted_early_exit_penalty + 
            weighted_optimal_exit +
            weighted_risk_reward +
            weighted_partial_exit +
            weighted_stability
        )
        
        # Limit to reasonable values
        total_reward = np.clip(total_reward, -2.0, 3.0)
        
        # Handle NaN and Inf values
        if np.isnan(total_reward) or np.isinf(total_reward):
            return 0.0
        
        return total_reward
        
    except Exception as e:
        logger.error(f"Error calculating risk management reward: {e}")
        return 0.0

def calculate_dynamic_risk_parameters(
    price_data: pd.DataFrame,
    current_idx: int,
    lookahead: int = 20,
    atr_multiple_sl: float = 1.5,
    atr_multiple_tp: float = 3.0,
    market_regime: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate optimal risk parameters based on price data and market analysis
    
    Args:
        price_data: DataFrame with price and indicator data
        current_idx: Current index in the DataFrame
        lookahead: Number of bars to look ahead for exit simulation
        atr_multiple_sl: Default ATR multiple for stop-loss
        atr_multiple_tp: Default ATR multiple for take-profit
        market_regime: Known market regime if available ('trending', 'ranging', etc.)
        
    Returns:
        Dictionary with optimal risk parameters
    """
    try:
        if current_idx >= len(price_data) or current_idx < 0:
            return {
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'is_trailing_stop': False,
                'is_trailing_take': False,
                'exit_strategy': 'standard',
                'partial_exit_levels': [],
                'partial_exit_sizes': []
            }
        
        # Get current data
        current_price = price_data.iloc[current_idx]['Close']
        
        # Get ATR if available
        atr_pct = 0.015  # Default 1.5% ATR
        if 'atr_percent' in price_data.columns:
            atr_pct = price_data.iloc[current_idx]['atr_percent'] / 100.0  # Convert to decimal
        
        # Determine market regime if not provided
        if market_regime is None:
            if 'regime' in price_data.columns:
                market_regime = price_data.iloc[current_idx]['regime']
            elif 'is_trending' in price_data.columns and price_data.iloc[current_idx]['is_trending']:
                market_regime = 'trending'
            else:
                # Estimate regime from recent price action
                lookback = min(current_idx, 20)
                recent_prices = price_data.iloc[current_idx-lookback:current_idx+1]['Close']
                price_change = abs((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0])
                price_volatility = recent_prices.pct_change().std()
                
                if price_change > 0.05:  # 5% move in recent bars
                    market_regime = 'trending'
                elif price_volatility > 0.015:  # High volatility
                    market_regime = 'volatile'
                else:
                    market_regime = 'ranging'
        
        # Adjust risk parameters based on market regime
        if market_regime == 'trending':
            # In trending markets: tighter stops, wider targets, use trailing
            sl_multiple = atr_multiple_sl * 0.8  # 20% tighter stops in trends
            tp_multiple = atr_multiple_tp * 1.2  # 20% wider targets in trends
            is_trailing_stop = True
            is_trailing_take = False
            exit_strategy = 'trailing'
            
        elif market_regime == 'volatile':
            # In volatile markets: wider stops, moderate targets
            sl_multiple = atr_multiple_sl * 1.5  # 50% wider stops in volatility
            tp_multiple = atr_multiple_tp * 0.9  # Slightly tighter targets
            is_trailing_stop = False
            is_trailing_take = False
            exit_strategy = 'standard'
            
        else:  # ranging or default
            # In ranging markets: moderate stops, use partial exits
            sl_multiple = atr_multiple_sl * 1.2  # 20% wider stops in ranges
            tp_multiple = atr_multiple_tp * 0.8  # 20% tighter targets in ranges
            is_trailing_stop = False
            is_trailing_take = False
            exit_strategy = 'partial'
        
        # Calculate stop-loss and take-profit percentages
        stop_loss_pct = atr_pct * sl_multiple
        take_profit_pct = atr_pct * tp_multiple
        
        # Ensure reasonable limits
        stop_loss_pct = min(max(0.01, stop_loss_pct), 0.15)  # 1% to 15%
        take_profit_pct = min(max(0.02, take_profit_pct), 0.30)  # 2% to 30%
        
        # Calculate partial exit levels if using partial exit strategy
        partial_exit_levels = []
        partial_exit_sizes = []
        
        if exit_strategy == 'partial':
            # Create 3 exit levels
            partial_exit_levels = [
                take_profit_pct * 0.33,  # 1/3 of the way to target
                take_profit_pct * 0.67,  # 2/3 of the way to target
                take_profit_pct * 1.0    # Full target
            ]
            
            # Size to exit at each level
            partial_exit_sizes = [0.25, 0.35, 0.40]  # Exit 25%, then 35%, then 40%
        
        # Return risk parameters
        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'is_trailing_stop': is_trailing_stop,
            'is_trailing_take': is_trailing_take,
            'exit_strategy': exit_strategy,
            'partial_exit_levels': partial_exit_levels,
            'partial_exit_sizes': partial_exit_sizes,
            'atr_percent': atr_pct,
            'market_regime': market_regime
        }
        
    except Exception as e:
        logger.error(f"Error calculating dynamic risk parameters: {e}")
        # Return default values
        return {
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'is_trailing_stop': False,
            'is_trailing_take': False,
            'exit_strategy': 'standard',
            'partial_exit_levels': [],
            'partial_exit_sizes': []
        }

# Advanced risk analysis functions
def analyze_position_risk(
    position: Position,
    current_price: float,
    market_volatility: float = 0.015,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Analyze risk metrics for an active position
    
    Args:
        position: Active trading position
        current_price: Current market price
        market_volatility: Current market volatility (daily)
        risk_free_rate: Risk-free rate (daily)
        
    Returns:
        Dict with risk metrics
    """
    try:
        entry_price = position.entry_price
        if entry_price == 0:
            return {}
        
        # Calculate current P&L
        if position.is_long:
            current_pnl_pct = (current_price - entry_price) / entry_price
        else:
            current_pnl_pct = (entry_price - current_price) / entry_price
        
        # Risk metrics
        risk_metrics = {
            'current_pnl_pct': current_pnl_pct,
            'max_runup': position.max_runup,
            'max_drawdown': position.max_drawdown,
            'holding_period': position.holding_period
        }
        
        # Calculate risk-adjusted metrics
        if position.stop_loss_level is not None:
            # Calculate current risk
            if position.is_long:
                risk_amount = (current_price - position.stop_loss_level) / current_price
            else:
                risk_amount = (position.stop_loss_level - current_price) / current_price
            
            # Risk/reward ratio
            if position.take_profit_level is not None:
                if position.is_long:
                    reward_amount = (position.take_profit_level - current_price) / current_price
                else:
                    reward_amount = (current_price - position.take_profit_level) / current_price
                
                if risk_amount > 0:
                    risk_metrics['current_risk_reward'] = reward_amount / risk_amount
            
            # Sharpe-like ratio for this trade
            if position.holding_period > 0:
                daily_return = current_pnl_pct / position.holding_period
                risk_metrics['trade_sharpe'] = (daily_return - risk_free_rate) / market_volatility
        
        # Check if any partial exits should be recommended now
        risk_metrics['recommend_partial_exit'] = False
        if current_pnl_pct > 0:
            # If we've captured significant profit but price might be stalling
            if current_pnl_pct > position.max_runup * 0.8 and position.holding_period > 5:
                risk_metrics['recommend_partial_exit'] = True
                risk_metrics['recommended_exit_size'] = 0.3  # Exit 30% of position
        
        # Check if stop adjustment is recommended
        risk_metrics['recommend_stop_adjustment'] = False
        if current_pnl_pct > 0.03 and position.stop_loss_level is not None:
            # If we're in profit, consider moving stop to breakeven
            if position.is_long and position.stop_loss_level < entry_price:
                risk_metrics['recommend_stop_adjustment'] = True
                risk_metrics['recommended_stop_level'] = entry_price
            elif not position.is_long and position.stop_loss_level > entry_price:
                risk_metrics['recommend_stop_adjustment'] = True
                risk_metrics['recommended_stop_level'] = entry_price
        
        return risk_metrics
        
    except Exception as e:
        logger.error(f"Error analyzing position risk: {e}")
        return {}

def analyze_risk_management_performance(trades, include_stopped_trades=True):
    """
    Analyze performance of risk management strategies.
    
    Args:
        trades: List of completed trades
        include_stopped_trades: Whether to include trades closed via stop-loss/take-profit
        
    Returns:
        dict: Performance statistics
    """
    if not trades:
        return {
            'stop_loss_hit_rate': 0.0,
            'take_profit_hit_rate': 0.0,
            'avg_loss_prevented': 0.0,
            'avg_profit_captured': 0.0,
            'risk_reward_ratio': 0.0,
            'avg_holding_period': 0.0,
            'partial_exit_effectiveness': 0.0
        }
    
    # Filter trades based on includes_stopped_trades
    if not include_stopped_trades:
        trades = [t for t in trades if not t.get('is_stop_loss_exit', False) and not t.get('is_take_profit_exit', False)]
        
    if not trades:
        return {
            'stop_loss_hit_rate': 0.0,
            'take_profit_hit_rate': 0.0,
            'avg_loss_prevented': 0.0,
            'avg_profit_captured': 0.0,
            'risk_reward_ratio': 0.0,
            'avg_holding_period': 0.0,
            'partial_exit_effectiveness': 0.0
        }
    
    # Perform analysis
    total_trades = len(trades)
    stop_loss_exits = sum(1 for t in trades if t.get('is_stop_loss_exit', False))
    take_profit_exits = sum(1 for t in trades if t.get('is_take_profit_exit', False))
    
    # Calculate loss prevention metrics
    loss_prevented = []
    for trade in trades:
        if trade.get('is_stop_loss_exit', False):
            max_adverse_price = trade.get('max_adverse_price', trade.get('exit_price', 0))
            exit_price = trade.get('exit_price', 0)
            entry_price = trade.get('entry_price', 0)
            if entry_price > 0:
                prevented = (exit_price - max_adverse_price) / entry_price
                loss_prevented.append(prevented)
    
    avg_loss_prevented = np.mean(loss_prevented) if loss_prevented else 0.0
    
    # Calculate profit capture metrics
    profit_captured = []
    for trade in trades:
        if trade.get('is_take_profit_exit', False):
            max_favorable_price = trade.get('max_favorable_price', trade.get('exit_price', 0))
            exit_price = trade.get('exit_price', 0)
            entry_price = trade.get('entry_price', 0)
            if entry_price > 0 and max_favorable_price > entry_price:
                captured = (exit_price - entry_price) / (max_favorable_price - entry_price)
                profit_captured.append(captured)
    
    avg_profit_captured = np.mean(profit_captured) if profit_captured else 0.0
    
    # Calculate risk/reward ratio
    risk_reward_ratios = []
    for trade in trades:
        stop_loss_pct = trade.get('stop_loss_pct', 0)
        take_profit_pct = trade.get('take_profit_pct', 0)
        if stop_loss_pct > 0:
            risk_reward_ratios.append(take_profit_pct / stop_loss_pct)
    
    avg_risk_reward_ratio = np.mean(risk_reward_ratios) if risk_reward_ratios else 0.0
    
    # Calculate holding periods
    holding_periods = [t.get('holding_period', 0) for t in trades if 'holding_period' in t]
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
    
    # Evaluate partial exit effectiveness
    partial_exit_effectiveness = 0.0
    partial_exit_trades = [t for t in trades if t.get('closed_parts', [])]
    
    if partial_exit_trades:
        total_potential_profit = 0.0
        total_realized_profit = 0.0
        
        for trade in partial_exit_trades:
            entry_price = trade.get('entry_price', 0)
            max_favorable_price = trade.get('max_favorable_price', 0)
            exit_price = trade.get('exit_price', 0)
            closed_parts = trade.get('closed_parts', [])
            size = trade.get('size', 1.0)
            
            # Calculate potential profit (if exit at max favorable price)
            potential_profit = ((max_favorable_price - entry_price) / entry_price) * size
            
            # Calculate realized profit (from partial exits and final exit)
            realized_profit = 0.0
            for part_size, part_price in closed_parts:
                realized_profit += ((part_price - entry_price) / entry_price) * part_size
                
            # Add final exit profit
            remaining_size = size - sum(part_size for part_size, _ in closed_parts)
            if remaining_size > 0 and exit_price > 0:
                realized_profit += ((exit_price - entry_price) / entry_price) * remaining_size
            
            # Add to totals
            if potential_profit > 0:
                total_potential_profit += potential_profit
                total_realized_profit += realized_profit
        
        # Calculate effectiveness ratio
        if total_potential_profit > 0:
            partial_exit_effectiveness = total_realized_profit / total_potential_profit
    
    # Calculate success by exit strategy
    success_by_strategy = {}
    for strategy in ['standard', 'trailing', 'partial']:
        strategy_trades = [t for t in trades if t.get('exit_strategy') == strategy]
        if strategy_trades:
            wins = sum(1 for t in strategy_trades if t.get('profit_loss', 0) > 0)
            success_by_strategy[strategy] = wins / len(strategy_trades)
    
    return {
        'stop_loss_hit_rate': stop_loss_exits / total_trades if total_trades > 0 else 0.0,
        'take_profit_hit_rate': take_profit_exits / total_trades if total_trades > 0 else 0.0,
        'avg_loss_prevented': avg_loss_prevented,
        'avg_profit_captured': avg_profit_captured,
        'risk_reward_ratio': avg_risk_reward_ratio,
        'avg_holding_period': avg_holding_period,
        'partial_exit_effectiveness': partial_exit_effectiveness,
        'success_by_strategy': success_by_strategy
    }