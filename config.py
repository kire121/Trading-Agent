import argparse
import sys

parser = argparse.ArgumentParser(description="DQN Trading Agent")

# Agent- och träningsparametrar
parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
parser.add_argument("--epsilon_min", type=float, default=0.02, help="Minsta värdet för epsilon") # Ökat från 0.01 för att behålla mer utforskning
parser.add_argument("--epsilon_decay", type=float, default=0.99, help="Epsilon decay rate") # Långsammare decay från 0.995 för mer utforskning
parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Lärhastighet")
parser.add_argument("--batch_size", type=int, default=32, help="Batchstorlek vid replay")
parser.add_argument("--episodes", type=int, default=100, help="Antal episoder")
parser.add_argument("--memory_size", type=int, default=100000, help="Storlek på replayminne")
parser.add_argument("--model_type", type=str, default="dqn", choices=["dqn", "double_dqn", "dueling_dqn"], help="Typ av DQN-modell")
parser.add_argument("--load_model", type=str, default="", help="Ladda modell från fil")
parser.add_argument("--save_dir", type=str, default="models", help="Mapp att spara modeller i")

# Prioritized Replay specifika parametrar
parser.add_argument("--per_alpha", type=float, default=0.6, help="Alpha parameter för prioriterad replay")
parser.add_argument("--per_beta", type=float, default=0.4, help="Beta parameter för prioriterad replay")
parser.add_argument("--per_beta_increment", type=float, default=0.001, help="Beta inkrement för prioriterad replay")

# Nya risk-adjusted parametrar
parser.add_argument("--risk_free_rate", type=float, default=0.0, help="Riskfri ränta för Sharpe/Sortino-beräkningar (daglig)")
parser.add_argument("--risk_window", type=int, default=20, help="Fönsterstorlek för beräkning av riskmått")

# Nya förenklade reward-komponenter (ersätter gamla)
parser.add_argument("--direct_return_weight", type=float, default=2.0, help="Vikt för portföljavkastning i belöningsfunktionen")
parser.add_argument("--sharpe_change_weight", type=float, default=1.0, help="Vikt för förbättring av Sharpe-ratio")
parser.add_argument("--drawdown_penalty_weight", type=float, default=0.2, help="Vikt för drawdown-straff")
parser.add_argument("--regime_reward_weight", type=float, default=1.0, help="Vikt för marknadsregim-belöning")

# Marknadsregim-detektionsparametrar
parser.add_argument("--volatility_threshold", type=float, default=0.015, help="Volatilitetströskel för att identifiera volatil marknad")
parser.add_argument("--trend_threshold", type=float, default=0.03, help="Tröskel för att identifiera trendande marknad")
parser.add_argument("--range_threshold", type=float, default=0.02, help="Tröskel för att identifiera sidledes marknad")

# Kelly-kriteriet parametrar
parser.add_argument("--use_kelly", action="store_true", default=True, help="Använd Kelly-kriteriet för positionsstorlek")
parser.add_argument("--max_kelly_fraction", type=float, default=0.4, help="Maximal fraktion av kapital att allokera (Half-Kelly)") # Minskat från 0.6 till 0.4 för mer konservativ positionering
parser.add_argument("--kelly_window", type=int, default=20, help="Antal senaste trades att använda för Kelly-beräkningar")
parser.add_argument("--kelly_initial_win_rate", type=float, default=0.55, help="Initial vinstfrekvens innan tillräcklig historik finns") # Ökat från 0.5 för optimistisk start
parser.add_argument("--kelly_initial_win_loss_ratio", type=float, default=1.2, help="Initialt vinstförlustförhållande innan tillräcklig historik finns") # Ökat från 1.0 för optimistisk start
parser.add_argument("--min_kelly_trades", type=int, default=3, help="Minsta antal trades innan Kelly-beräkningar används") # Minskat från 5 för snabbare adaption

# Risk Management-parametrar
parser.add_argument("--use_risk_management", action="store_true", default=True, help="Aktivera risk management med stop-loss/take-profit")
parser.add_argument("--default_stop_loss_pct", type=float, default=0.05, help="Standard stop-loss procentsats (5%)")
parser.add_argument("--default_take_profit_pct", type=float, default=0.10, help="Standard take-profit procentsats (10%)")
parser.add_argument("--default_trailing_pct", type=float, default=0.03, help="Standard glidande stop-loss/take-profit procentsats (3%)")
parser.add_argument("--min_stop_loss_pct", type=float, default=0.01, help="Minsta tillåtna stop-loss procentsats (1%)")
parser.add_argument("--max_stop_loss_pct", type=float, default=0.15, help="Högsta tillåtna stop-loss procentsats (15%)")
parser.add_argument("--min_take_profit_pct", type=float, default=0.01, help="Minsta tillåtna take-profit procentsats (1%)")
parser.add_argument("--max_take_profit_pct", type=float, default=0.30, help="Högsta tillåtna take-profit procentsats (30%)")
parser.add_argument("--stop_loss_adjustment_threshold", type=float, default=0.01, help="Minsta prisrörelse för att justera glidande stop-loss (1%)")

# Risk Management Network parametrar
parser.add_argument("--risk_learning_rate", type=float, default=0.001, help="Inlärningshastighet för risk management-nätverket")
parser.add_argument("--risk_memory_size", type=int, default=5000, help="Storlek på risk management-minne")
parser.add_argument("--risk_model_update_frequency", type=int, default=10, help="Uppdatera risk management-modellen var N:e steg")
parser.add_argument("--risk_batch_size", type=int, default=32, help="Batchstorlek för risk management-träning")

# Risk Management Belöningsparametrar (behålls för bakåtkompatibilitet)
parser.add_argument("--loss_prevention_weight", type=float, default=0.8, help="Vikt för belöning av förhindrad förlust")
parser.add_argument("--profit_capture_weight", type=float, default=1.5, help="Vikt för belöning av fångad vinst")
parser.add_argument("--early_exit_penalty", type=float, default=0.1, help="Straffvikt för för tidig exit")
parser.add_argument("--stability_bonus", type=float, default=0.2, help="Bonusvikt för stabila nivåer")

# Pattern Recognition-parametrar (justerade för pattern feature i state)
parser.add_argument("--use_pattern_features", action="store_true", default=True, help="Använd mönsterigenkänning som state-feature")
parser.add_argument("--pattern_similarity_threshold", type=float, default=0.7, help="Likhetströskel för att identifiera liknande mönster")
parser.add_argument("--max_pattern_count", type=int, default=50, help="Max antal mönster att spara")
parser.add_argument("--min_profit_for_pattern", type=float, default=0.01, help="Minsta vinst för att spara ett mönster")

# UI- och displaykonstanter
BACKGROUND_COLOR = (30, 30, 30)
HEADER_TOP_COLOR = (50, 80, 120)
HEADER_BOTTOM_COLOR = (20, 40, 70)
PANEL_BG_COLOR = (50, 50, 50)
LINE_COLOR = (0, 200, 255)
TRADE_BUY_COLOR = (0, 255, 0)
TRADE_SELL_COLOR = (255, 0, 0)
REWARD_LINE_COLOR = (255, 255, 0)
TEXT_COLOR = (255, 255, 255)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER_COLOR = (100, 100, 100)
CHART_GRID_COLOR = (70, 70, 70)
KELLY_BUY_COLOR = (0, 255, 128)  # Ljusare grön färg för Kelly-köp

# Färger för Stop-Loss och Take-Profit
STOP_LOSS_COLOR = (255, 100, 100)  # Ljusröd för stop-loss linjer
TAKE_PROFIT_COLOR = (100, 255, 100)  # Ljusgrön för take-profit linjer
TRAILING_STOP_COLOR = (255, 150, 150)  # Rosa för glidande stop-loss
TRAILING_TAKE_COLOR = (150, 255, 150)  # Mintgrön för glidande take-profit

# Tekniska indikatorer
INDICATORS = [
    'sma_10', 'sma_20', 'ema_10', 'ema_20', 'rsi', 'macd', 'macd_signal',
    'bb_upper', 'bb_middle', 'bb_lower', 'momentum', 'volatility'
]

# Parametrar för datahämtning
MAX_DATA_FETCH_RETRIES = 3
FETCH_RETRY_DELAY = 2  # sekunder

# Skapa en struktur för enklare åtkomst till risk management-parametrar
def get_risk_management_params():
    """
    Returnerar en dictionary med alla risk management-parametrar
    för enkel åtkomst i andra delar av koden.
    """
    return {
        'use_risk_management': args.use_risk_management,
        'default_stop_loss_pct': args.default_stop_loss_pct,
        'default_take_profit_pct': args.default_take_profit_pct,
        'default_trailing_pct': args.default_trailing_pct,
        'min_stop_loss_pct': args.min_stop_loss_pct,
        'max_stop_loss_pct': args.max_stop_loss_pct,
        'min_take_profit_pct': args.min_take_profit_pct,
        'max_take_profit_pct': args.max_take_profit_pct,
        'stop_loss_adjustment_threshold': args.stop_loss_adjustment_threshold,
        'learning_rate': args.risk_learning_rate,
        'memory_size': args.risk_memory_size,
        'model_update_frequency': args.risk_model_update_frequency,
        'batch_size': args.risk_batch_size,
        'reward_params': {
            'loss_prevention_weight': args.loss_prevention_weight,
            'profit_capture_weight': args.profit_capture_weight,
            'early_exit_penalty': args.early_exit_penalty,
            'stability_bonus': args.stability_bonus
        }
    }

# Skapa en struktur för enklare åtkomst till reward-parametrar
def get_reward_params():
    """
    Returnerar en dictionary med alla reward-parametrar
    för enkel åtkomst i andra delar av koden.
    """
    return {
        'direct_return_weight': args.direct_return_weight,  # Make sure this is explicitly set
        'sharpe_change_weight': args.sharpe_change_weight,
        'drawdown_penalty_weight': args.drawdown_penalty_weight,
        'regime_reward_weight': args.regime_reward_weight,
        'volatility_threshold': args.volatility_threshold,
        'trend_threshold': args.trend_threshold,
        'range_threshold': args.range_threshold
    }

# Skapa en struktur för enklare åtkomst till pattern recognition-parametrar
def get_pattern_params():
    """
    Returnerar en dictionary med alla pattern recognition-parametrar
    för enkel åtkomst i andra delar av koden.
    """
    return {
        'use_pattern_features': args.use_pattern_features,
        'similarity_threshold': args.pattern_similarity_threshold,
        'max_pattern_count': args.max_pattern_count,
        'min_profit_for_pattern': args.min_profit_for_pattern
    }

# Parsear argument 
args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])

# Skapa risk_management_params som en global variabel
risk_management_params = get_risk_management_params()

# Skapa reward_params som en global variabel
reward_params = get_reward_params()

# Skapa pattern_params som en global variabel
pattern_params = get_pattern_params()