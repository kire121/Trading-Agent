import pandas as pd
import datetime
import time
import numpy as np
import ta
from pandas_datareader import data as web
from scipy.signal import argrelextrema
from config import MAX_DATA_FETCH_RETRIES, FETCH_RETRY_DELAY
from logger_setup import logger

def calculate_indicators(df):
    """
    Calculates technical indicators on price data with enhanced features for risk management.
    New indicators include:
    - ATR for volatility-based risk scaling
    - Market regime indicators (trending/ranging)
    - Support/resistance detection
    - Momentum and reversal indicators
    """
    # Keep original index before making a copy
    original_index = df.index
    
    df = df.copy()
    
    # === TREND INDICATORS ===
    df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['ema_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    
    # MACD for trend strength and direction
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Directional movement index for trend strength
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # === VOLATILITY INDICATORS ===
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # Width for volatility measurement
    
    # ATR - crucial for dynamic stop loss and take profit calculations
    atr_indicator = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'])
    df['atr'] = atr_indicator.average_true_range()
    df['atr_percent'] = df['atr'] / df['Close'] * 100  # ATR as percentage of price
    
    # Keltner Channels for volatility context
    keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
    df['kc_upper'] = keltner.keltner_channel_hband()
    df['kc_middle'] = keltner.keltner_channel_mband()
    df['kc_lower'] = keltner.keltner_channel_lband()
    
    # === MOMENTUM INDICATORS ===
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['rsi_5'] = ta.momentum.rsi(df['Close'], window=5)  # Faster RSI for quicker signals
    df['momentum'] = ta.momentum.roc(df['Close'], window=10)
    
    # Stochastic oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # CMO (Chande Momentum Oscillator) - Using ROC as alternative since chande_momentum_oscillator is not available
    try:
        # Try using CMO if available in installed version
        df['cmo'] = ta.momentum.chande_momentum_oscillator(df['Close'])
    except AttributeError:
        # Fallback to ROC with different window as approximate alternative
        df['cmo'] = ta.momentum.roc(df['Close'], window=14)
        logger.info("Used ROC as fallback for CMO indicator")
    
    # === VOLUME INDICATORS ===
    if 'Volume' in df.columns:
        # On-balance volume
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume-weighted average price
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # === MARKET REGIME INDICATORS ===
    # Calculate rolling volatility
    df['volatility_5'] = df['Close'].pct_change().rolling(window=5).std() * 100
    df['volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * 100
    
    # Trend intensity - higher values indicate stronger trend
    df['trend_intensity'] = abs(df['Close'] - df['sma_50']) / df['atr']
    
    # Create market regime indicator (0=ranging, 1=trending)
    df['is_trending'] = ((df['adx'] > 25) & (df['trend_intensity'] > 2)).astype(int)
    
    # === SUPPORT/RESISTANCE DETECTION ===
    # Price distance from recent highs/lows as percentage
    df['dist_from_52w_high'] = df['Close'].rolling(252).max().fillna(df['Close']) - df['Close']
    df['dist_from_52w_high_pct'] = df['dist_from_52w_high'] / df['Close'] * 100
    
    df['dist_from_52w_low'] = df['Close'] - df['Close'].rolling(252).min().fillna(df['Close'])
    df['dist_from_52w_low_pct'] = df['dist_from_52w_low'] / df['Close'] * 100
    
    # Calculate recent support and resistance levels
    # We'll use a simplified approach - more advanced methods could be implemented
    window = 20  # Look back window
    
    # Resistance level (average of recent highs)
    df['resistance_level'] = df['High'].rolling(window=window).max()
    df['dist_to_resistance'] = (df['resistance_level'] - df['Close']) / df['Close'] * 100
    
    # Support level (average of recent lows)
    df['support_level'] = df['Low'].rolling(window=window).min()
    df['dist_to_support'] = (df['Close'] - df['support_level']) / df['Close'] * 100
    
    # === REVERSAL INDICATORS ===
    # RSI divergence detection (simplified)
    df['price_higher_high'] = (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2))
    df['rsi_lower_high'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'].shift(1) > df['rsi'].shift(2))
    df['bearish_divergence'] = df['price_higher_high'] & df['rsi_lower_high']
    
    df['price_lower_low'] = (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2))
    df['rsi_higher_low'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'].shift(1) < df['rsi'].shift(2))
    df['bullish_divergence'] = df['price_lower_low'] & df['rsi_higher_low']
    
    # === RISK MANAGEMENT FEATURES ===
    # Estimated slippage based on volatility and volume (if available)
    if 'Volume' in df.columns:
        # Higher volatility + lower volume = higher slippage
        vol_factor = df['volatility_5'] / df['volatility_5'].median()
        volume_factor = df['Volume'].median() / df['Volume']
        df['slippage_estimate'] = vol_factor * volume_factor * 0.001 * df['Close']  # 0.1% base slippage
    else:
        df['slippage_estimate'] = df['volatility_5'] / df['volatility_5'].median() * 0.001 * df['Close']
    
    # Dynamic stop-loss suggestions based on ATR
    df['suggested_sl_pct'] = df['atr_percent'] * 1.5  # 1.5x ATR
    df['suggested_sl_pct'] = df['suggested_sl_pct'].clip(0.5, 10.0)  # Limit between 0.5% and 10%
    
    # Dynamic take-profit suggestions based on ATR and volatility
    df['suggested_tp_pct'] = df['atr_percent'] * 3.0  # 3x ATR
    df['suggested_tp_pct'] = df['suggested_tp_pct'].clip(1.0, 20.0)  # Limit between 1% and 20%
    
    # Risk-reward ratio suggestion
    df['suggested_risk_reward'] = df['suggested_tp_pct'] / df['suggested_sl_pct']
    
    # === PATTERN DETECTION ===
    # Simple pattern detection for common candlestick patterns
    # Doji (open and close are very close)
    df['doji'] = abs(df['Open'] - df['Close']) <= 0.1 * (df['High'] - df['Low'])
    
    # Hammer (long lower shadow, small body, little or no upper shadow)
    df['hammer'] = ((df['High'] - df['Low']) > 3 * abs(df['Open'] - df['Close'])) & \
                   ((df['Close'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6) & \
                   ((df['Open'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6)
    
    # === MISC FEATURES ===
    # Time to key levels (in terms of ATR units)
    df['time_to_resistance_atr'] = (df['resistance_level'] - df['Close']) / df['atr']
    df['time_to_support_atr'] = (df['Close'] - df['support_level']) / df['atr']
    
    # Fill NaN values
    df.fillna(0, inplace=True)
    
    # Ensure we keep the original index
    if pd.api.types.is_datetime64_any_dtype(original_index):
        # Check that index is preserved
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = original_index
    
    return df

def safe_data_fetcher(instrument, start_date, end_date, data_source='stooq'):
    """Fetches data with retry and error handling"""
    retries = 0
    while retries < MAX_DATA_FETCH_RETRIES:
        try:
            logger.info(f"Trying to fetch data for {instrument} (attempt {retries + 1}/{MAX_DATA_FETCH_RETRIES})")
            df = web.DataReader(instrument, data_source, start_date, end_date)
            logger.info(f"Successfully fetched data for {instrument}: {len(df)} rows")
            return df
        except Exception as e:
            retries += 1
            logger.warning(f"Error fetching data for {instrument}: {e}")
            if retries < MAX_DATA_FETCH_RETRIES:
                logger.info(f"Waiting {FETCH_RETRY_DELAY} seconds before next attempt...")
                time.sleep(FETCH_RETRY_DELAY)
            else:
                logger.error(f"Could not fetch data for {instrument} after {MAX_DATA_FETCH_RETRIES} attempts")
                return None

def identify_support_resistance(df, window=20, order=5):
    """
    Identify support and resistance levels using local extrema
    
    Args:
        df: DataFrame with price data
        window: Window size for finding local extrema
        order: Required number of points on each side for peak/valley detection
        
    Returns:
        DataFrame with support and resistance levels
    """
    df = df.copy()
    
    # Find local maxima and minima
    df['resistance'] = df['High'].rolling(window=window, center=True).apply(
        lambda x: 1 if np.argmax(x) == len(x) // 2 else 0, raw=True)
    df['support'] = df['Low'].rolling(window=window, center=True).apply(
        lambda x: 1 if np.argmin(x) == len(x) // 2 else 0, raw=True)
    
    # Extract actual price levels
    resistance_levels = df.loc[df['resistance'] == 1, 'High'].tolist()
    support_levels = df.loc[df['support'] == 1, 'Low'].tolist()
    
    # Add columns with closest support/resistance
    if resistance_levels:
        df['closest_resistance'] = df['Close'].apply(
            lambda x: min(resistance_levels, key=lambda y: abs(y - x) if y > x else float('inf')))
    else:
        df['closest_resistance'] = df['High'].rolling(window=window).max()
        
    if support_levels:
        df['closest_support'] = df['Close'].apply(
            lambda x: min(support_levels, key=lambda y: abs(y - x) if y < x else float('inf')))
    else:
        df['closest_support'] = df['Low'].rolling(window=window).min()
    
    # Calculate distance to support/resistance as percentage
    df['distance_to_resistance_pct'] = (df['closest_resistance'] - df['Close']) / df['Close'] * 100
    df['distance_to_support_pct'] = (df['Close'] - df['closest_support']) / df['Close'] * 100
    
    # Fill NaN values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

def detect_market_regime(df, adx_threshold=25, lookback=20):
    """
    Detects market regime (trending or ranging)
    
    Args:
        df: DataFrame with price data and indicators
        adx_threshold: ADX threshold for trending market
        lookback: Period for measuring directional change
        
    Returns:
        DataFrame with market regime indicator
    """
    df = df.copy()
    
    # ADX for trend strength
    if 'adx' not in df.columns:
        adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        df['adx'] = adx_indicator.adx()
    
    # Directional change measurement
    df['directional_change'] = abs(df['Close'].pct_change(lookback))
    
    # Volatility measurement
    df['volatility'] = df['Close'].pct_change().rolling(lookback).std()
    
    # Trend classification
    df['regime'] = 'ranging'  # Default
    
    # Strong trend
    strong_trend_mask = (df['adx'] > adx_threshold) & (df['directional_change'] > 0.02)
    df.loc[strong_trend_mask, 'regime'] = 'trending'
    
    # Weak range
    weak_range_mask = (df['adx'] < adx_threshold/2) & (df['volatility'] < df['volatility'].median()/2)
    df.loc[weak_range_mask, 'regime'] = 'weak_range'
    
    # Convert to numeric for ML
    df['regime_numeric'] = df['regime'].map({'weak_range': 0, 'ranging': 1, 'trending': 2})
    
    return df

def calculate_dynamic_risk_levels(df):
    """
    Calculate dynamic stop-loss and take-profit levels based on volatility and market regime
    
    Args:
        df: DataFrame with price data and indicators
        
    Returns:
        DataFrame with suggested risk levels
    """
    df = df.copy()
    
    # Ensure ATR is calculated
    if 'atr' not in df.columns:
        atr_indicator = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'])
        df['atr'] = atr_indicator.average_true_range()
        df['atr_percent'] = df['atr'] / df['Close'] * 100
    
    # Ensure we have regime information
    if 'regime_numeric' not in df.columns:
        df = detect_market_regime(df)
    
    # Base stop-loss as percentage of price
    # Scale according to market regime (tighter for trending, wider for ranging)
    df['optimal_sl_pct'] = np.where(
        df['regime_numeric'] == 2,  # Trending
        df['atr_percent'] * 1.2,    # 1.2x ATR for trending
        np.where(
            df['regime_numeric'] == 1,  # Ranging
            df['atr_percent'] * 1.8,    # 1.8x ATR for ranging
            df['atr_percent'] * 1.5     # 1.5x ATR for weak range (default)
        )
    )
    
    # Base take-profit as percentage of price
    # Scale according to market regime (wider for trending, tighter for ranging)
    df['optimal_tp_pct'] = np.where(
        df['regime_numeric'] == 2,  # Trending
        df['atr_percent'] * 3.5,    # 3.5x ATR for trending
        np.where(
            df['regime_numeric'] == 1,  # Ranging
            df['atr_percent'] * 2.2,    # 2.2x ATR for ranging
            df['atr_percent'] * 2.8     # 2.8x ATR for weak range (default)
        )
    )
    
    # Adjust for proximity to support/resistance if available
    if 'distance_to_resistance_pct' in df.columns and 'distance_to_support_pct' in df.columns:
        # If close to resistance, take-profit should be tighter
        resistance_factor = 1.0 - (0.2 * np.exp(-0.1 * df['distance_to_resistance_pct']))
        df['optimal_tp_pct'] = df['optimal_tp_pct'] * resistance_factor
        
        # If close to support, stop-loss should be tighter
        support_factor = 1.0 - (0.2 * np.exp(-0.1 * df['distance_to_support_pct']))
        df['optimal_sl_pct'] = df['optimal_sl_pct'] * support_factor
    
    # Adjust for trend strength
    if 'adx' in df.columns:
        # Stronger trend = tighter stop-loss, wider take-profit
        adx_normalized = df['adx'] / 100  # Normalize to 0-1 range
        df['optimal_sl_pct'] = df['optimal_sl_pct'] * (1.0 - 0.3 * adx_normalized)
        df['optimal_tp_pct'] = df['optimal_tp_pct'] * (1.0 + 0.5 * adx_normalized)
    
    # Ensure values are within reasonable bounds
    df['optimal_sl_pct'] = df['optimal_sl_pct'].clip(0.5, 8.0)  # 0.5% to 8%
    df['optimal_tp_pct'] = df['optimal_tp_pct'].clip(1.0, 15.0)  # 1% to 15%
    
    # Calculate optimal risk-reward ratio
    df['optimal_risk_reward'] = df['optimal_tp_pct'] / df['optimal_sl_pct']
    
    # Add recommended risk_management_action
    df['risk_action'] = np.where(
        df['optimal_risk_reward'] >= 2.5, 'standard',  # Standard stop/take if RR is good 
        np.where(
            df['regime_numeric'] == 2, 'trailing',     # Trailing stop if trending
            'partial_exits'                            # Partial exits if ranging with low RR
        )
    )
    
    # Add recommended partial exit levels (for partial exit strategy)
    df['partial_exit_1_pct'] = df['optimal_tp_pct'] * 0.33  # 1/3 of the way to target
    df['partial_exit_2_pct'] = df['optimal_tp_pct'] * 0.67  # 2/3 of the way to target
    
    # Size of position to exit at each level
    df['partial_exit_1_size'] = 0.25  # Exit 25% at first level
    df['partial_exit_2_size'] = 0.33  # Exit 33% at second level
    
    # Add optimal trailing stop distance (as % of price)
    df['optimal_trailing_pct'] = df['atr_percent'] * 1.0  # 1x ATR
    df['optimal_trailing_pct'] = df['optimal_trailing_pct'].clip(0.5, 5.0)  # 0.5% to 5%
    
    return df

def create_synthetic_data(instrument, initial_cash=10000, days=252, with_enhanced_features=True):
    """
    Creates synthetic price data with realistic market patterns if real data cannot be loaded.
    Enhanced with more realistic market dynamics and full technical indicators.
    
    Args:
        instrument: Instrument name (for reference only)
        initial_cash: Initial cash amount
        days: Number of trading days to generate
        with_enhanced_features: Whether to add enhanced features
        
    Returns:
        DataFrame with synthetic price and indicator data
    """
    np.random.seed(42)
    n_days = days  # One year of trading days
    base_price = 100.0
    
    # Market regime parameters
    regime_changes = 4  # Number of regime changes
    regime_durations = np.random.randint(30, 80, size=regime_changes)  # Days per regime
    
    # Create regimes (0=ranging, 1=trending up, 2=trending down)
    regimes = []
    for i in range(regime_changes):
        regime = np.random.choice([0, 1, 2])
        regimes.extend([regime] * regime_durations[i])
    
    # Ensure we have enough regimes for all days
    while len(regimes) < n_days:
        regimes.append(np.random.choice([0, 1, 2]))
    regimes = regimes[:n_days]
    
    # Parameters for different market regimes
    regime_params = {
        0: {'trend': 0.0001, 'volatility': 0.01},    # Ranging
        1: {'trend': 0.0015, 'volatility': 0.008},   # Trending up
        2: {'trend': -0.0012, 'volatility': 0.012}   # Trending down
    }
    
    # Generate prices based on regimes
    prices = [base_price]
    for i in range(1, n_days):
        regime = regimes[i]
        params = regime_params[regime]
        
        # Add some mean reversion to the ranging regime
        if regime == 0 and len(prices) > 20:
            mean_price = np.mean(prices[-20:])
            mean_reversion = 0.02 * (mean_price - prices[-1]) / prices[-1]
            daily_return = np.random.normal(params['trend'] + mean_reversion, params['volatility'])
        else:
            daily_return = np.random.normal(params['trend'], params['volatility'])
        
        prices.append(prices[-1] * (1 + daily_return))
    
    # Create date index with proper business day dynamics
    start_date = datetime.datetime(2020, 1, 1)
    date_range = pd.date_range(start=start_date, periods=n_days*1.4)  # Add extra days to compensate for weekends
    trading_dates = [date for date in date_range if date.weekday() < 5][:n_days]  # Only trading days
    
    # Create basic OHLC data
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0, 0.006)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.006)) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in range(n_days)]
    }, index=trading_dates)
    
    # Add some seasonal and weekly effects
    # Weekday effects
    weekday_effect = {0: 0.001, 1: 0.0005, 2: 0, 3: -0.0005, 4: 0.002}  # Mon, Tue, Wed, Thu, Fri
    
    # Monthly effects
    month_effect = {
        1: 0.002,   # January - strong
        2: 0.001,   # February
        3: 0.0005,  # March
        4: 0.0005,  # April
        5: 0,       # May
        6: -0.001,  # June - weak
        7: -0.002,  # July - weak
        8: -0.001,  # August - weak
        9: 0,       # September
        10: 0.001,  # October
        11: 0.002,  # November - strong
        12: 0.003   # December - strong
    }
    
    # Apply day-of-week and monthly patterns
    for i, date in enumerate(df.index):
        if i > 0:  # Skip first day
            weekday = date.weekday()
            month = date.month
            
            # Add combined effect
            daily_effect = weekday_effect.get(weekday, 0) + month_effect.get(month, 0)
            
            # Apply effect to today's closing price
            df.loc[date, 'Close'] *= (1 + daily_effect)
            # Update Open, High, Low for consistency
            df.loc[date, 'Open'] *= (1 + daily_effect)
            df.loc[date, 'High'] *= (1 + daily_effect)
            df.loc[date, 'Low'] *= (1 + daily_effect)
    
    # Ensure High is always higher than Close and Open, and Low is always lower
    df['High'] = df[['High', 'Close', 'Open']].max(axis=1) * (1 + 0.001)
    df['Low'] = df[['Low', 'Close', 'Open']].min(axis=1) * (1 - 0.001)
    
    # Add market regime labels directly to the DataFrame
    df['synthetic_regime'] = regimes
    df['regime_label'] = df['synthetic_regime'].map({
        0: 'ranging', 1: 'trending_up', 2: 'trending_down'
    })
    
    # Calculate basic indicators
    df = calculate_indicators(df)
    
    # Add enhanced risk management features if requested
    if with_enhanced_features:
        # Add support/resistance levels
        df = identify_support_resistance(df)
        
        # Add dynamic risk levels
        df = calculate_dynamic_risk_levels(df)
    
    logger.info(f"Created synthetic data for {instrument} with market regime patterns and seasonal effects")
    return df