import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import pickle
from config import args
from logger_setup import logger
from risk_management import (
    RiskManagementNetwork, 
    Position, 
    calculate_risk_management_reward,
    calculate_dynamic_risk_parameters,
    analyze_position_risk
)

class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer for storing and sampling experiences with priorities.
    Enhanced with support for additional risk management data.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, experience):
        """
        Add an experience to the buffer with maximum priority.
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done, risk_info)
        """
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Parameter for importance sampling weights
            
        Returns:
            Tuple of (indices, experiences, weights)
        """
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        return indices, samples, weights

    def update_priorities(self, indices, new_priorities):
        """
        Update priorities for specific experiences.
        
        Args:
            indices: Indices of experiences to update
            new_priorities: New priority values
        """
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with support for optimization parameters via kwargs.
    Enhanced with sophisticated risk management and action parameterization.
    
    Expected hyperparameters (can be overridden via kwargs):
      - gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, memory_size,
      - per_alpha, per_beta, per_beta_increment.
    """
    def __init__(self, state_size, action_size, model_type='dqn', use_kelly=True, use_risk_management=True, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.model_type = model_type
        self.use_kelly = use_kelly  # Kelly criterion parameter
        self.use_risk_management = use_risk_management  # Risk management parameter
        
        # Core hyperparameters
        self.gamma = kwargs.get('gamma', args.gamma)
        self.epsilon = kwargs.get('epsilon', args.epsilon)
        self.epsilon_min = kwargs.get('epsilon_min', args.epsilon_min)
        self.epsilon_decay = kwargs.get('epsilon_decay', args.epsilon_decay)
        self.learning_rate = kwargs.get('learning_rate', args.learning_rate)
        self.memory_size = kwargs.get('memory_size', args.memory_size)
        
        # Prioritized experience replay parameters
        self.memory = PrioritizedReplayBuffer(self.memory_size, alpha=kwargs.get('per_alpha', 0.6))
        self.beta = kwargs.get('per_beta', 0.4)
        self.beta_increment_per_sampling = kwargs.get('per_beta_increment', 0.001)
        
        # Training metrics
        self.loss_history = []
        self.reward_history = []
        self.q_value_history = []
        
        # Create main model
        self.model = self._build_model()
        
        # Create target model for double DQN
        if self.model_type == 'double_dqn':
            self.target_model = self._build_model()
            self.update_target_model()
            
        # Risk management specific parameters
        self.risk_params = kwargs.get('risk_params', {})
        self.risk_learning_rate = self.risk_params.get('learning_rate', 0.001)
        
        # Create risk management network if enabled
        self.risk_network = None
        self.risk_memory = None
        if self.use_risk_management:
            # Create a copy to avoid modifying the original
            risk_params_copy = self.risk_params.copy()
            
            # Ensure we use our extracted risk_learning_rate
            risk_params_copy['learning_rate'] = self.risk_learning_rate
            
            # Initialize risk management network
            self.risk_network = RiskManagementNetwork(
                state_size=state_size,
                **risk_params_copy
            )
            
            # Risk management tracking information
            self.last_state = None
            self.last_action = None
            self.last_risk_levels = None
            self.accumulated_risk_reward = 0.0
            
            # For tracking recommended risk parameters
            self.recommended_risk_params = {}
            
            # For evaluating risk management performance
            self.risk_management_metrics = {
                'sl_hit_count': 0,
                'tp_hit_count': 0,
                'partial_exit_count': 0,
                'total_trades': 0,
                'successful_trades': 0,
                'risk_adjusted_returns': []
            }
        
        # For tracking and analyzing model performance
        self.step_counter = 0
        self.episode_counter = 0
        self.training_metrics = {
            'avg_q_values': [],
            'max_q_values': [],
            'loss_values': [],
            'epsilon_values': []
        }
        
        # Action mapping - extended with risk management actions
        self.action_mapping = {
            0: "HOLD",
            1: "BUY",
            2: "SELL",
            3: "KELLY_BUY",
            4: "SET_STOP_LOSS",
            5: "SET_TRAILING_STOP",
            6: "SET_TAKE_PROFIT",
            7: "SET_TRAILING_TAKE",
            8: "PARTIAL_SELL_25",
            9: "PARTIAL_SELL_50",
            10: "PARTIAL_SELL_75",
            11: "REMOVE_STOP_LOSS",
            12: "REMOVE_TAKE_PROFIT",
            13: "SET_PARTIAL_EXIT_STRATEGY",
            14: "SET_TRAILING_STRATEGY",
            15: "ADJUST_RISK_PARAMS"
        }
        
        # Success patterns tracking for state-based risk management
        self.success_patterns = {}
        
        # Initialize exploration noise for Ornstein-Uhlenbeck process
        # This helps with more effective exploration in continuous action spaces
        self.theta = 0.15
        self.ou_state = np.ones(self.action_size) * 0.0
        self.ou_sigma = 0.2
        self.ou_mu = 0.0

    def _build_model(self):
        """Build neural network model based on model_type."""
        if self.model_type in ['dqn', 'double_dqn']:
            # Standard DQN model with batch normalization and dropout for regularization
            model = Sequential([
                Dense(128, activation='relu', input_shape=(self.state_size,)),
                BatchNormalization(),
                Dense(128, activation='relu'),
                Dropout(0.2),
                BatchNormalization(),
                Dense(64, activation='relu'),
                Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model
            
        elif self.model_type == 'dueling_dqn':
            # Dueling DQN architecture that separates value and advantage streams
            inputs = tf.keras.Input(shape=(self.state_size,))
            
            # Shared layers
            x = Dense(128, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            
            # Value stream (estimates state value)
            value_stream = Dense(64, activation='relu')(x)
            value = Dense(1)(value_stream)
            
            # Advantage stream (estimates action advantages)
            advantage_stream = Dense(64, activation='relu')(x)
            advantage = Dense(self.action_size)(advantage_stream)
            
            # Combine value and advantage streams
            outputs = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model
            
        elif self.model_type == 'rainbow_dqn':
            # A simplified version of Rainbow DQN with some key components
            inputs = tf.keras.Input(shape=(self.state_size,))
            
            # Noisy networks for exploration (simplified)
            def noisy_dense(x, units, name):
                # Standard dense layer with added Gaussian noise
                dense = Dense(units, activation='relu')(x)
                noise_scale = 0.1 / np.sqrt(self.step_counter + 1)  # Reduce noise over time
                noise = tf.random.normal(shape=tf.shape(dense), stddev=noise_scale)
                return dense + noise
            
            # Network architecture
            x = noisy_dense(inputs, 128, 'noisy1')
            x = BatchNormalization()(x)
            x = noisy_dense(x, 128, 'noisy2')
            x = Dropout(0.2)(x)
            
            # Dueling architecture
            value_stream = Dense(64, activation='relu')(x)
            value = Dense(1)(value_stream)
            
            advantage_stream = Dense(64, activation='relu')(x)
            advantage = Dense(self.action_size)(advantage_stream)
            
            outputs = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.learning_rate))
            return model
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def update_target_model(self):
        """Copy weights from main model to target model for double DQN."""
        if self.model_type == 'double_dqn':
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, risk_info=None):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            risk_info: Additional risk management information
        """
        # Store experience with risk information if available
        self.memory.add((state, action, reward, next_state, done, risk_info or {}))

    def _ornstein_uhlenbeck_noise(self):
        """
        Generate Ornstein-Uhlenbeck noise for better exploration.
        This produces temporally correlated noise that's more effective for control tasks.
        
        Returns:
            numpy array: Noise values for each action
        """
        x = self.ou_state
        dx = self.theta * (self.ou_mu - x) + self.ou_sigma * np.random.randn(len(x))
        self.ou_state = x + dx
        return self.ou_state

    def act(self, state, training=True, additional_info=None):
        """
        Choose an action based on current state.
        Also handles consultation with risk management network for stop-loss/take-profit levels
        if a new position is opened.
        
        Args:
            state: Current state values as numpy array
            training: Whether agent is in training mode
            additional_info: Additional information for decision making (market context, etc.)
            
        Returns:
            int: Selected action
        """
        # Random action selection during exploration
        if training and np.random.rand() <= self.epsilon:
            # Special handling for risk management actions
            if self.use_risk_management:
                # If we have an active position (state[1] > 0), give higher chance for risk management actions
                if state[1] > 0:  # state[1] = position size
                    # 60% chance to choose risk management action (actions 4-15)
                    if np.random.rand() < 0.6:
                        # Choose among risk management actions, with weighted selection
                        risk_actions = list(range(4, 16))  # Actions 4 through 15
                        
                        # Higher weights for most important actions (stop-loss, take-profit, trailing)
                        risk_weights = [
                            0.20,  # SET_STOP_LOSS (4)
                            0.15,  # SET_TRAILING_STOP (5)
                            0.20,  # SET_TAKE_PROFIT (6)
                            0.15,  # SET_TRAILING_TAKE (7)
                            0.05,  # PARTIAL_SELL_25 (8)
                            0.05,  # PARTIAL_SELL_50 (9)
                            0.05,  # PARTIAL_SELL_75 (10)
                            0.02,  # REMOVE_STOP_LOSS (11)
                            0.02,  # REMOVE_TAKE_PROFIT (12)
                            0.05,  # SET_PARTIAL_EXIT_STRATEGY (13)
                            0.04,  # SET_TRAILING_STRATEGY (14)
                            0.02   # ADJUST_RISK_PARAMS (15)
                        ]
                        return np.random.choice(risk_actions, p=risk_weights)
                
                # Special handling for Kelly-based buys
                if self.use_kelly and np.random.rand() < 0.7:  # 70% preference for Kelly buys
                    return 3  # Kelly-based position
                
                # Updated weights for trading actions
                action_weights = [0.15, 0.45, 0.30, 0.10]  # Weights for actions 0-3
                return np.random.choice(range(min(4, self.action_size)), p=action_weights)
            else:
                # Traditional random action if risk management isn't used
                return random.randrange(min(4, self.action_size))  # Only use first 4 actions
        
        # Predict Q-values for the current state
        state_array = np.array(state).reshape(1, -1).astype(np.float32)
        act_values = self.model.predict(state_array, verbose=0)
        
        # Track Q-values for diagnostics
        if training:
            self.training_metrics['avg_q_values'].append(np.mean(act_values[0]))
            self.training_metrics['max_q_values'].append(np.max(act_values[0]))
        
        # Special case for Kelly-based buys if model values buying highly
        if self.use_kelly and self.action_size > 3 and state[1] <= 0:  # No position yet
            # If regular buy signal is highest
            if np.argmax(act_values[0]) == 1:
                # Compare with Kelly-based buy signal
                if act_values[0][3] > act_values[0][1] * 0.6:  # Lower threshold (0.6) for Kelly
                    return 3  # Use Kelly instead
        
        # Special case for risk management when we have a position
        if self.use_risk_management and state[1] > 0:  # Have a position
            # Check if stop-loss/take-profit are not set
            has_stop_loss = state[3] > 0  # state[3] = stop_loss_set
            has_take_profit = state[4] > 0  # state[4] = take_profit_set
            
            # Prioritize setting risk levels if agent recommends it highly
            if not has_stop_loss and act_values[0][4] > act_values[0][0] * 1.02:
                return 4  # Set stop-loss
            
            if not has_take_profit and act_values[0][6] > act_values[0][0] * 1.02:
                return 6  # Set take-profit
            
            # If both SL and TP are set, consider other risk management actions
            if has_stop_loss and has_take_profit:
                # Consider switching to trailing strategy if profitable
                if state[2] > 0.02:  # If price has moved >2% in our favor
                    trailing_value = act_values[0][14]  # SET_TRAILING_STRATEGY
                    if trailing_value > act_values[0][0] * 1.02:
                        return 14  # Set trailing strategy
                
                # Consider setting partial exit strategy if profitable
                if state[2] > 0.03:  # If price has moved >3% in our favor
                    partial_value = act_values[0][13]  # SET_PARTIAL_EXIT_STRATEGY
                    if partial_value > act_values[0][0] * 1.02:
                        return 13  # Set partial exit strategy
        
        # Apply exploration noise during training
        if training:
            # Add Ornstein-Uhlenbeck noise to Q-values for better exploration
            noise = self._ornstein_uhlenbeck_noise() * (self.epsilon * 0.5)
            act_values[0] = act_values[0] + noise
        
        # Choose action with highest (possibly noise-adjusted) Q-value
        return np.argmax(act_values[0])
    
    def get_risk_management_levels(self, state, market_data=None):
        """
        Consult risk management network to get optimal stop-loss and take-profit levels.
        
        Args:
            state: Current state
            market_data: Additional market data if available
            
        Returns:
            dict: Risk management parameters
        """
        if not self.use_risk_management or self.risk_network is None:
            # Return default values if risk management is not used
            return {
                'stop_loss_pct': 0.05, 
                'take_profit_pct': 0.10, 
                'is_trailing_stop': False, 
                'is_trailing_take': False,
                'exit_strategy': 'standard',
                'partial_exit_levels': [],
                'partial_exit_sizes': []
            }
        
        # Extract market volatility if available
        market_volatility = None
        if market_data is not None and 'atr_percent' in market_data:
            market_volatility = market_data['atr_percent']
        
        # Save state for later training
        self.last_state = state
        
        # Predict optimal risk levels from risk management network
        risk_levels = self.risk_network.predict_risk_levels(state, market_volatility)
        self.last_risk_levels = risk_levels
        
        # Store recommended parameters for evaluation
        self.recommended_risk_params = risk_levels
        
        return risk_levels
    
    def update_risk_management(self, state, reward, next_state, done, info):
        """
        Update risk management network based on position outcome.
        
        Args:
            state: State when position was active
            reward: Main reward
            next_state: New state
            done: Whether episode is done
            info: Information from environment, including risk_management_reward
        """
        if not self.use_risk_management or self.risk_network is None or self.last_state is None:
            return
        
        # Use specialized risk management reward if available
        risk_reward = info.get('risk_management_reward', 0.0)
        
        try:
            # Limit to reasonable values
            risk_reward = np.clip(risk_reward, -3.0, 7.0)
            
            # Accumulate reward but limit to prevent extreme values
            self.accumulated_risk_reward += risk_reward
            self.accumulated_risk_reward = np.clip(self.accumulated_risk_reward, -15.0, 25.0)
            
            # Log large values for diagnostics
            if abs(risk_reward) > 2.0 or abs(self.accumulated_risk_reward) > 10.0:
                print(f"Large risk reward: {risk_reward:.2f}, accumulated: {self.accumulated_risk_reward:.2f}")
                
        except Exception as e:
            # In case of errors, log and reset to avoid crashing training
            print(f"Error handling risk management reward: {e}")
            risk_reward = 0.0
            self.accumulated_risk_reward = 0.0
        
        # Check trade metrics for risk management evaluation
        if info.get('auto_exit_executed', False):
            self.risk_management_metrics['total_trades'] += 1
            
            if info.get('exit_reason') == 'stop_loss':
                self.risk_management_metrics['sl_hit_count'] += 1
            elif info.get('exit_reason') == 'take_profit':
                self.risk_management_metrics['tp_hit_count'] += 1
            elif info.get('exit_reason', '').startswith('partial_exit'):
                self.risk_management_metrics['partial_exit_count'] += 1
            
            # Check if trade was successful
            profit_loss = info.get('profit_loss', 0.0)
            if profit_loss > 0:
                self.risk_management_metrics['successful_trades'] += 1
                
            # Calculate risk-adjusted return
            if 'max_drawdown' in info and info['max_drawdown'] != 0:
                risk_adj = profit_loss / abs(info['max_drawdown'])
                self.risk_management_metrics['risk_adjusted_returns'].append(risk_adj)
        
        # Update risk management network if position was closed or episode ends
        if info.get('auto_exit_executed', False) or info.get('trade_made', False) or done:
            if self.last_risk_levels is not None:
                # Extract risk parameters from last prediction
                risk_params = self.last_risk_levels
                
                # Update risk management network
                self.risk_network.update(
                    state=self.last_state,
                    risk_params=risk_params,
                    reward=self.accumulated_risk_reward,
                    next_state=next_state,
                    done=done,
                    additional_info=info
                )
                
                # Reset accumulated reward and tracking variables
                self.accumulated_risk_reward = 0.0
                self.last_state = None
                self.last_risk_levels = None
    
    def _calculate_td_errors(self, states, actions, rewards, next_states, dones):
        """
        Calculate TD errors for prioritized replay.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            numpy array: TD errors for each experience
        """
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Get next Q values based on model type
        if self.model_type == 'double_dqn':
            # Double DQN: Main model selects actions, target model evaluates them
            next_q = self.model.predict(next_states, verbose=0)
            next_actions = np.argmax(next_q, axis=1)
            target_next = self.target_model.predict(next_states, verbose=0)
            target_q_values = np.array([target_next[i, next_actions[i]] for i in range(len(actions))])
        else:
            # Standard DQN: Use max Q value from next state
            target_next = self.model.predict(next_states, verbose=0)
            target_q_values = np.amax(target_next, axis=1)
        
        # Calculate target Q values
        targets = rewards + (1 - dones) * self.gamma * target_q_values
        
        # Calculate TD errors
        td_errors = []
        for i, action in enumerate(actions):
            # Add safety measures
            try:
                td_error = abs(current_q[i][action] - targets[i])
                # Limit extreme TD errors
                td_error = np.clip(td_error, 0.0, 100.0)
            except Exception as e:
                print(f"Error calculating TD error: {e}")
                td_error = 1.0  # Use default value in case of error
                
            td_errors.append(td_error)
            
        return np.array(td_errors), targets
    
    def replay(self, batch_size):
        """
        Train the agent with experiences from replay buffer.
        
        Args:
            batch_size: Number of samples to train on
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from replay buffer
        indices, minibatch, weights = self.memory.sample(batch_size, beta=self.beta)
        
        # Extract components from sampled experiences
        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch]).astype(int)
        
        # Extract risk info from experiences (if available)
        risk_infos = [sample[5] if len(sample) > 5 else {} for sample in minibatch]
        
        # Calculate TD errors and target Q values
        td_errors, targets = self._calculate_td_errors(states, actions, rewards, next_states, dones)
        
        # Update network
        target = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target[i][action] = targets[i]
        
        # Train the model with importance sampling weights
        history = self.model.fit(states, target, sample_weight=weights, epochs=1, verbose=0)
        
        # Store loss history
        if 'loss' in history.history:
            loss_value = history.history['loss'][0]
            self.loss_history.append(loss_value)
            self.training_metrics['loss_values'].append(loss_value)
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Update epsilon and beta parameters
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.training_metrics['epsilon_values'].append(self.epsilon)
        
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        # Update target model periodically
        if self.model_type == 'double_dqn':
            self.step_counter += 1
            if self.step_counter % 50 == 0:
                self.update_target_model()
        
        # Train risk management network if enabled
        if self.use_risk_management and self.risk_network is not None:
            # With 15% chance to train (higher frequency for risk training)
            if random.random() < 0.15:
                self.risk_network.train()
    
    def on_position_opened(self, state, market_data=None):
        """
        Called when a new position is opened to consult risk management network.
        
        Args:
            state: State at position opening
            market_data: Market data if available
            
        Returns:
            dict: Risk management parameters
        """
        if not self.use_risk_management or self.risk_network is None:
            # Return default values if risk management is not used
            return {
                'stop_loss_pct': 0.05, 
                'take_profit_pct': 0.10, 
                'is_trailing_stop': False, 
                'is_trailing_take': False,
                'exit_strategy': 'standard',
                'partial_exit_levels': [],
                'partial_exit_sizes': []
            }
        
        # Consult risk management network
        levels = self.get_risk_management_levels(state, market_data)
        self.last_risk_levels = levels
        return levels
    
    def on_position_closed(self, state, next_state, position_info, reward):
        """
        Called when a position is closed to update risk management network.
        
        Args:
            state: State when position was active
            next_state: State after position closing
            position_info: Information about the position
            reward: Reward from position closing
        """
        if not self.use_risk_management or self.risk_network is None:
            return
        
        # Extract position information
        info = {
            'entry_price': position_info.get('entry_price', 0.0),
            'exit_price': position_info.get('exit_price', 0.0),
            'max_favorable_price': position_info.get('max_favorable_price', position_info.get('exit_price', 0.0)),
            'max_adverse_price': position_info.get('max_adverse_price', position_info.get('exit_price', 0.0)),
            'stop_loss_level': position_info.get('stop_loss_level', None),
            'take_profit_level': position_info.get('take_profit_level', None),
            'is_stop_loss_exit': position_info.get('is_stop_loss_exit', False),
            'is_take_profit_exit': position_info.get('is_take_profit_exit', False),
            'exit_strategy': position_info.get('exit_strategy', 'standard'),
            'closed_parts': position_info.get('closed_parts', []),
            'profit_loss': position_info.get('profit_loss', 0.0),
            'holding_period': position_info.get('holding_period', 0),
            'max_runup': position_info.get('max_runup', 0.0),
            'max_drawdown': position_info.get('max_drawdown', 0.0)
        }
        
        # Extract risk parameters used for this position
        risk_params = {
            'stop_loss_pct': position_info.get('stop_loss_pct', 0.05),
            'take_profit_pct': position_info.get('take_profit_pct', 0.10),
            'is_trailing_stop': position_info.get('is_trailing_stop', False),
            'is_trailing_take': position_info.get('is_trailing_take', False)
        }
        
        # Add partial exit parameters if they exist
        if 'closed_parts' in position_info and position_info['closed_parts']:
            closed_parts = position_info['closed_parts']
            for i, (size, price) in enumerate(closed_parts):
                if i < 3:  # Support up to 3 partial exits
                    risk_params[f'partial_exit_{i+1}_pct'] = (price - position_info['entry_price']) / position_info['entry_price']
                    risk_params[f'partial_exit_{i+1}_size'] = size / position_info['size']
        
        # Calculate specialized risk management reward
        try:
            risk_reward = calculate_risk_management_reward(
                entry_price=info['entry_price'],
                exit_price=info['exit_price'],
                max_favorable_price=info['max_favorable_price'],
                max_adverse_price=info['max_adverse_price'],
                stop_loss_level=info['stop_loss_level'],
                take_profit_level=info['take_profit_level'],
                is_stop_loss_exit=info['is_stop_loss_exit'],
                is_take_profit_exit=info['is_take_profit_exit'],
                partial_exits=info['closed_parts'] if info['closed_parts'] else None
            )
        except Exception as e:
            print(f"Error calculating risk management reward: {e}")
            risk_reward = reward  # Fall back to regular reward
        
        # Update risk management network
        self.risk_network.update(
            state=state,
            risk_params=risk_params,
            reward=risk_reward,
            next_state=next_state,
            done=False,  # False because we're only closing position, not episode
            additional_info=info
        )
        
        # Track successful strategies in pattern database if profitable
        if info['profit_loss'] > 0.02:  # >2% profit is considered successful
            pattern_key = self._get_pattern_key(state)
            if pattern_key not in self.success_patterns:
                self.success_patterns[pattern_key] = {
                    'count': 0,
                    'total_profit': 0.0,
                    'best_strategy': None,
                    'best_profit': 0.0
                }
            
            # Update pattern statistics
            pattern = self.success_patterns[pattern_key]
            pattern['count'] += 1
            pattern['total_profit'] += info['profit_loss']
            
            # Track best strategy
            if info['profit_loss'] > pattern['best_profit']:
                pattern['best_profit'] = info['profit_loss']
                pattern['best_strategy'] = {
                    'stop_loss_pct': risk_params['stop_loss_pct'],
                    'take_profit_pct': risk_params['take_profit_pct'],
                    'is_trailing_stop': risk_params['is_trailing_stop'],
                    'is_trailing_take': risk_params['is_trailing_take'],
                    'exit_strategy': info['exit_strategy']
                }
    
    def _get_pattern_key(self, state):
        """
        Create a simple hash key for pattern recognition.
        
        Args:
            state: State to create key from
            
        Returns:
            str: Key for pattern dictionary
        """
        # Simplify state to key features for pattern matching
        try:
            # Use only the most important features to avoid overfitting
            key_features = []
            
            # Price and position features (always first in state)
            key_features.append(round(state[0], 2))  # Price (rounded)
            key_features.append(1 if state[1] > 0 else 0)  # Has position (binary)
            
            # Get technical indicators if present (state positions vary by implementation)
            for i, value in enumerate(state):
                # Add indicator features based on index or name
                if i >= 3 and i < 10:  # Assuming indicators start after position info
                    # Round and bin the indicator values to reduce key space
                    binned_value = round(value * 5) / 5  # Round to nearest 0.2
                    key_features.append(binned_value)
            
            # Convert to string key
            return str(key_features)
        except:
            # Fallback in case of errors
            return str(hash(str(state)))
    
    def recommend_risk_strategy(self, state, market_data=None):
        """
        Recommend risk strategy based on state pattern matching or risk network.
        
        Args:
            state: Current state
            market_data: Additional market data
            
        Returns:
            dict: Recommended risk parameters
        """
        # First try pattern matching from successful trades
        pattern_key = self._get_pattern_key(state)
        if pattern_key in self.success_patterns and self.success_patterns[pattern_key]['count'] >= 3:
            # Use best strategy from similar successful patterns
            pattern = self.success_patterns[pattern_key]
            return pattern['best_strategy']
        
        # Fall back to risk network
        if self.risk_network:
            return self.get_risk_management_levels(state, market_data)
        
        # Default values if no other source available
        return {
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'is_trailing_stop': False,
            'is_trailing_take': False,
            'exit_strategy': 'standard',
            'partial_exit_levels': [],
            'partial_exit_sizes': []
        }
    
    def on_episode_end(self, episode_reward, episode_trades):
        """
        Process end of episode metrics and adjust learning as needed.
        
        Args:
            episode_reward: Total reward for episode
            episode_trades: List of trades from episode
        """
        self.episode_counter += 1
        self.reward_history.append(episode_reward)
        
        # Adjust learning rate if needed
        if self.episode_counter % 50 == 0 and len(self.loss_history) > 100:
            # Calculate recent loss volatility
            recent_loss = self.loss_history[-100:]
            loss_std = np.std(recent_loss)
            loss_mean = np.mean(recent_loss)
            
            # If loss is unstable, reduce learning rate
            if loss_std > loss_mean * 0.5:
                self.learning_rate *= 0.8
                print(f"Reducing learning rate to {self.learning_rate} due to unstable loss")
                
                # Recompile model with new learning rate
                self.model.compile(
                    loss='mse', 
                    optimizer=Adam(learning_rate=self.learning_rate)
                )
        
        # Adjust risk learning parameters if needed
        if self.use_risk_management and self.risk_network and len(episode_trades) > 0:
            # Count successful vs unsuccessful trades
            success_count = sum(1 for t in episode_trades if t.get('profit_loss', 0) > 0)
            total_trades = len(episode_trades)
            
            # If success rate is poor, boost risk network learning rate
            if total_trades >= 5 and success_count / total_trades < 0.3:
                self.risk_network.learning_rate *= 1.2
                print(f"Boosting risk network learning rate to {self.risk_network.learning_rate}")
    
    def save(self, filepath):
        """Save the agent model and state."""
        # Save main model
        model_path = f"{filepath}.keras"
        if os.path.exists(model_path):
            backup_path = f"{model_path}.bak"
            logger.info(f"Creating backup: {backup_path}")
            try:
                os.rename(model_path, backup_path)
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        try:
            self.model.save(model_path)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return
        
        # Save agent state
        agent_state = {
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'model_type': self.model_type,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'use_kelly': self.use_kelly,
            'use_risk_management': self.use_risk_management,
            'success_patterns': self.success_patterns,
            'step_counter': self.step_counter,
            'episode_counter': self.episode_counter,
            'training_metrics': self.training_metrics,
            'risk_management_metrics': self.risk_management_metrics if self.use_risk_management else None
        }
        state_path = f"{filepath}_state.pkl"
        try:
            with open(state_path, 'wb') as f:
                pickle.dump(agent_state, f)
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
            return
        
        # Save risk management network if enabled
        if self.use_risk_management and self.risk_network is not None:
            try:
                self.risk_network.save(filepath)
            except Exception as e:
                logger.error(f"Error saving risk management network: {e}")
        
        logger.info(f"Model saved to {model_path}")

    def load(self, filepath):
        """Load the agent model and state."""
        # Load main model
        model_path = filepath
        if not filepath.endswith('.keras') and not filepath.endswith('.h5'):
            keras_path = f"{filepath}.keras"
            h5_path = f"{filepath}.h5"
            if os.path.exists(keras_path):
                model_path = keras_path
            elif os.path.exists(h5_path):
                model_path = h5_path
            else:
                raise FileNotFoundError(f"Model not found: {filepath}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Load agent state
        state_path = f"{filepath}_state.pkl"
        if os.path.exists(state_path):
            try:
                with open(state_path, 'rb') as f:
                    agent_state = pickle.load(f)
                self.epsilon = agent_state.get('epsilon', self.epsilon)
                self.loss_history = agent_state.get('loss_history', [])
                self.reward_history = agent_state.get('reward_history', [])
                self.model_type = agent_state.get('model_type', self.model_type)
                self.gamma = agent_state.get('gamma', self.gamma)
                self.learning_rate = agent_state.get('learning_rate', self.learning_rate)
                self.use_kelly = agent_state.get('use_kelly', self.use_kelly)
                self.use_risk_management = agent_state.get('use_risk_management', self.use_risk_management)
                self.success_patterns = agent_state.get('success_patterns', {})
                self.step_counter = agent_state.get('step_counter', 0)
                self.episode_counter = agent_state.get('episode_counter', 0)
                self.training_metrics = agent_state.get('training_metrics', {
                    'avg_q_values': [],
                    'max_q_values': [],
                    'loss_values': [],
                    'epsilon_values': []
                })
                if self.use_risk_management:
                    self.risk_management_metrics = agent_state.get('risk_management_metrics', {
                        'sl_hit_count': 0,
                        'tp_hit_count': 0,
                        'partial_exit_count': 0,
                        'total_trades': 0,
                        'successful_trades': 0,
                        'risk_adjusted_returns': []
                    })
                
                if self.model_type == 'double_dqn':
                    self.target_model = self._build_model()
                    self.update_target_model()
                logger.info(f"Agent state loaded with epsilon {self.epsilon}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        else:
            logger.warning(f"No state found for {filepath}, using default values")
        
        # Load risk management network if enabled
        if self.use_risk_management and self.risk_network is not None:
            risk_model_path = f"{filepath}_risk_mgmt.keras"
            if os.path.exists(risk_model_path):
                try:
                    self.risk_network.load(filepath)
                    logger.info(f"Risk management network loaded from {risk_model_path}")
                except Exception as e:
                    logger.warning(f"Could not load risk management network: {e}")
    
    def adjust_for_reward_scale(self, new_scale=10.0):
        """
        Adjust model for a different reward scale.
        This helps when switching between reward functions with different scales.
        
        Args:
            new_scale: New maximum reward scale
        """
        if not self.loss_history:
            return
            
        # Estimate old reward scale from loss history
        avg_loss = np.mean(self.loss_history[-100:]) if len(self.loss_history) > 100 else np.mean(self.loss_history)
        old_scale_estimate = np.sqrt(avg_loss) * 10.0
        
        # Adjust weights in output layer
        if old_scale_estimate > 0 and abs(old_scale_estimate - new_scale) > 1.0:
            scale_factor = new_scale / old_scale_estimate
            
            # Get model weights
            weights = self.model.get_weights()
            
            # Last layer is the output layer - scale its weights
            output_weights = weights[-2]  # Weights before biases
            output_biases = weights[-1]   # Biases
            
            # Scale output weights and biases
            output_weights *= scale_factor
            output_biases *= scale_factor
            
            # Update weights
            weights[-2] = output_weights
            weights[-1] = output_biases
            self.model.set_weights(weights)
            
            # Reset optimizer to handle new scale
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            logger.info(f"Adjusted model for new reward scale: {old_scale_estimate:.1f} -> {new_scale:.1f}")
            
        # Increase exploration temporarily to adapt to new reward function
        self.epsilon = max(self.epsilon, 0.3)
        logger.info(f"Increased epsilon to {self.epsilon} for adaptation to new reward scale")