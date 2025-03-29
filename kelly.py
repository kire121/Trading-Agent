# Först behöver vi lägga till en ny fil kelly.py som implementerar Kelly-logiken

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from logger_setup import logger

class KellyPositionSizer:
    """
    Implementerar Kelly-kriteriet för optimal positionsstorlek baserat på historiska resultat.
    """
    def __init__(self, 
                initial_win_rate: float = 0.5, 
                initial_win_loss_ratio: float = 1.0,
                window_size: int = 20,
                max_kelly_fraction: float = 0.5,
                min_trades_for_calculation: int = 5):
        """
        Initialisera Kelly Position Sizer.
        
        Args:
            initial_win_rate: Initial vinstfrekvens att använda innan tillräcklig data finns
            initial_win_loss_ratio: Initialt vinstförlustförhållande att använda initialt
            window_size: Antal senaste trades att använda för beräkning
            max_kelly_fraction: Maximal andel av kapitalet att investera (ofta används Kelly/2)
            min_trades_for_calculation: Minsta antal trades för att använda beräknad Kelly istället för initialt värde
        """
        self.win_rate = initial_win_rate
        self.win_loss_ratio = initial_win_loss_ratio
        self.window_size = window_size
        self.max_kelly_fraction = max_kelly_fraction
        self.min_trades_for_calculation = min_trades_for_calculation
        
        # Lista för historiska trades
        self.trades_history: List[Dict[str, float]] = []
        
    def add_trade_result(self, trade_result: Dict[str, float]) -> None:
        """
        Lägg till resultat från en trade i historiken.
        
        Args:
            trade_result: Dictionary med trade-resultat {'entry_price', 'exit_price', 'profit_loss', 'is_win'}
        """
        self.trades_history.append(trade_result)
        
        # Begränsa historiken till fönsterstorlek för att spara minne
        if len(self.trades_history) > self.window_size * 2:
            self.trades_history = self.trades_history[-self.window_size:]
        
        # Uppdatera statistik efter varje ny trade
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Uppdaterar vinstfrekvens och vinstförlustförhållande baserat på historiken"""
        if len(self.trades_history) < self.min_trades_for_calculation:
            logger.info(f"För få trades ({len(self.trades_history)}) för att beräkna Kelly, använder startvärden")
            return
        
        # Använd endast de senaste trades inom fönstret
        recent_trades = self.trades_history[-self.window_size:] if len(self.trades_history) > self.window_size else self.trades_history
        
        # Beräkna vinstfrekvens
        wins = sum(1 for trade in recent_trades if trade.get('is_win', False))
        self.win_rate = wins / len(recent_trades) if len(recent_trades) > 0 else self.win_rate
        
        # Beräkna genomsnittlig vinst och förlust
        win_trades = [trade['profit_loss'] for trade in recent_trades if trade.get('is_win', False)]
        loss_trades = [abs(trade['profit_loss']) for trade in recent_trades if not trade.get('is_win', False)]
        
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 1  # Undvik division med noll
        
        # Beräkna vinstförlustförhållande
        self.win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else self.win_loss_ratio
        
        logger.info(f"Uppdaterade Kelly-statistik: Win Rate: {self.win_rate:.2f}, Win/Loss Ratio: {self.win_loss_ratio:.2f}")
    
    def calculate_kelly_fraction(self) -> float:
        """
        Beräknar optimal positionsstorlek enligt Kelly-kriteriet.
        
        Returns:
            float: Andel av kapital att investera (0.0 - 1.0)
        """
        # Kelly-formeln: K = W - [(1 - W) / R]
        # Där W är vinstfrekvens och R är vinstförlustförhållande
        
        kelly = self.win_rate - ((1 - self.win_rate) / self.win_loss_ratio)
        
        # Begränsa till rimliga värden
        kelly = max(0.0, min(kelly, 1.0))
        
        # Många traders använder "half-Kelly" eller mindre för att minska risk
        kelly = min(kelly, self.max_kelly_fraction)
        
        return kelly
    
    def get_position_size(self, capital: float, max_position: Optional[float] = None) -> float:
        """
        Beräknar optimal positionsstorlek baserat på tillgängligt kapital.
        
        Args:
            capital: Tillgängligt kapital
            max_position: Maximal positionsstorlek (för att begränsa exponeringen)
            
        Returns:
            float: Rekommenderad positionsstorlek i kapitalenheter
        """
        kelly_fraction = self.calculate_kelly_fraction()
        position_size = capital * kelly_fraction
        
        if max_position is not None:
            position_size = min(position_size, max_position)
            
        return position_size