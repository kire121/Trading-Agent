import pygame
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Protocol, Union, Callable
import datetime
import os

from config import (BACKGROUND_COLOR, HEADER_TOP_COLOR, HEADER_BOTTOM_COLOR,
                    PANEL_BG_COLOR, LINE_COLOR, TRADE_BUY_COLOR, TRADE_SELL_COLOR,
                    REWARD_LINE_COLOR, TEXT_COLOR, BUTTON_COLOR, BUTTON_HOVER_COLOR, 
                    CHART_GRID_COLOR, INDICATORS, STOP_LOSS_COLOR, TAKE_PROFIT_COLOR,
                    TRAILING_STOP_COLOR, TRAILING_TAKE_COLOR, KELLY_BUY_COLOR)
from logger_setup import logger

# Define protocols for the expected interfaces
class EnvironmentInterface(Protocol):
    n_steps: int
    current_step: int
    prices: List[float]
    indicators: Dict[str, List[float]]
    indicator_values: Dict[str, List[float]]
    trade_events: List[Tuple]  # Modifierat för att stödja Kelly-info
    owned: float  # Ändrat från int till float för fraktionerade positioner
    initial_cash: float
    cash: float
    use_kelly: bool  # Egenskap för Kelly-aktivering
    kelly_sizer: Any  # Kelly Position Sizer-instans
    use_risk_management: bool  # Ny egenskap för risk management
    position: Any  # Position-objekt
    completed_trades: List[Dict[str, Any]]  # Lista över avslutade trades
    successful_trade_patterns: List[Dict[str, Any]]  # Lista över framgångsrika mönster
    
    def portfolio_value(self) -> float:
        ...
    
    def reset(self) -> Any:
        ...
    
    def close(self) -> None:
        ...
        
    def _get_state(self) -> np.ndarray:
        ...
        
    def calculate_state_similarity(self, state1, state2) -> float:
        ...

class AgentInterface(Protocol):
    epsilon: float
    gamma: float
    learning_rate: float
    memory: Any
    loss_history: List[float]
    model_type: str
    use_kelly: bool  # Egenskap för Kelly-aktivering
    use_risk_management: bool  # Ny egenskap för risk management
    risk_network: Any  # Risk management network
    
    def save(self, filepath: str) -> None:
        ...
    
    def act(self, state: np.ndarray) -> int:
        ...
    
    def replay(self, batch_size: int) -> None:
        ...

# -------------------------------------------------------------------------
# En enkel textinput för att mata in exempelvis antal episoder
# -------------------------------------------------------------------------
class TextInputBox:
    def __init__(self, rect: Tuple[int, int, int, int], font: pygame.font.Font, text: str = ''):
        self.rect = pygame.Rect(rect)
        self.color_inactive = (100, 100, 100)
        self.color_active = (150, 150, 150)
        self.color = self.color_inactive
        self.text = text
        self.font = font
        self.txt_surface = self.font.render(text, True, TEXT_COLOR)
        self.active = False

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive

        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                entered_text = self.text
                self.active = False
                self.color = self.color_inactive
                return entered_text
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                # Limit input to numbers for episode count
                if event.unicode.isdigit() or event.unicode == '.':
                    self.text += event.unicode
            self.txt_surface = self.font.render(self.text, True, TEXT_COLOR)
        return None

    def draw(self, screen: pygame.Surface) -> None:
        # Draw a background for better visibility
        pygame.draw.rect(screen, (60, 60, 60), self.rect)
        # Draw the text box outline
        pygame.draw.rect(screen, self.color, self.rect, 2)
        # Ensure text is centered vertically in the box
        text_y = self.rect.y + (self.rect.height - self.txt_surface.get_height()) // 2
        screen.blit(self.txt_surface, (self.rect.x + 5, text_y))

# -------------------------------------------------------------------------
# Enkel knapp
# -------------------------------------------------------------------------
class Button:
    def __init__(self, 
                rect: Union[Tuple[int, int, int, int], pygame.Rect], 
                text: str, 
                font: pygame.font.Font,
                color: Tuple[int, int, int] = BUTTON_COLOR,
                hover_color: Tuple[int, int, int] = BUTTON_HOVER_COLOR):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color

    def draw(self, surface: pygame.Surface) -> None:
        try:
            mouse_pos = pygame.mouse.get_pos()
            is_hover = self.rect.collidepoint(mouse_pos)
            draw_color = self.hover_color if is_hover else self.color
            
            # Draw button with rounded corners
            pygame.draw.rect(surface, draw_color, self.rect, border_radius=5)
            
            # Add a subtle border for better definition
            border_color = (180, 180, 180) if is_hover else (120, 120, 120)
            pygame.draw.rect(surface, border_color, self.rect, width=1, border_radius=5)
            
            # Render text and center it in the button
            text_surf = self.font.render(self.text, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
        except Exception as e:
            logger.error(f"Error drawing button '{self.text}': {e}")

    def is_clicked(self, event: pygame.event.Event) -> bool:
        return (event.type == pygame.MOUSEBUTTONDOWN 
                and event.button == 1
                and self.rect.collidepoint(event.pos))

# -------------------------------------------------------------------------
# Header-panel
# -------------------------------------------------------------------------
class Header:
    def __init__(self, 
                rect: Tuple[int, int, int, int], 
                font: pygame.font.Font,
                header_top_color: Tuple[int, int, int] = HEADER_TOP_COLOR,
                header_bottom_color: Tuple[int, int, int] = HEADER_BOTTOM_COLOR):
        self.rect = rect
        self.font = font
        self.top_color = header_top_color
        self.bottom_color = header_bottom_color
        self.margin = 20
        # Pre-calculate gradient colors for efficiency
        self._cached_gradient = self._calculate_gradient()

    def _calculate_gradient(self) -> List[Tuple[int, int, int]]:
        """Pre-calculate the gradient colors to improve rendering efficiency"""
        x, y, w, h = self.rect
        colors = []
        for i in range(h):
            ratio = i / h if h > 0 else 0
            r = int(self.top_color[0]*(1-ratio) + self.bottom_color[0]*ratio)
            g = int(self.top_color[1]*(1-ratio) + self.bottom_color[1]*ratio)
            b = int(self.top_color[2]*(1-ratio) + self.bottom_color[2]*ratio)
            colors.append((r, g, b))
        return colors

    def draw(self, surface: pygame.Surface, info_lines: List[str]) -> None:
        try:
            x, y, w, h = self.rect
            
            # Draw the gradient background using the cached colors
            for i, color in enumerate(self._cached_gradient[:h]):
                pygame.draw.line(surface, color, (x, y+i), (x+w, y+i))

            # Draw the text lines with proper spacing
            line_height = self.font.get_linesize() + 4
            max_lines = (h - 2 * self.margin) // line_height
            
            # If we have more lines than can fit, prioritize the most important ones
            display_lines = info_lines[:max_lines] if len(info_lines) > max_lines else info_lines
            
            # Draw the visible lines with proper spacing
            for i, line in enumerate(display_lines):
                text_surf = self.font.render(str(line), True, TEXT_COLOR)
                
                # Organize lines in columns if needed
                if i < 5:  # First column
                    x_pos = x + self.margin
                elif i < 10:  # Second column
                    x_pos = x + w // 2
                    line_height_offset = (i - 5) * line_height
                else:  # Third column (if needed)
                    x_pos = x + 3 * w // 4
                    line_height_offset = (i - 10) * line_height
                
                # Adjust y position
                y_pos = y + self.margin + (i % 5) * line_height
                
                surface.blit(text_surf, (x_pos, y_pos))
            
            # If there are more lines than can fit, indicate this
            if len(info_lines) > max_lines:
                more_text = self.font.render("...", True, TEXT_COLOR)
                surface.blit(more_text, (x + self.margin, y + h - self.margin - more_text.get_height()))
                
        except Exception as e:
            logger.error(f"Error drawing header: {e}")
            # Draw a fallback message if there's an error
            pygame.draw.rect(surface, HEADER_TOP_COLOR, self.rect)
            text_surf = self.font.render("Error displaying header", True, TEXT_COLOR)
            surface.blit(text_surf, (x + self.margin, y + self.margin))

# -------------------------------------------------------------------------
# PricePanel: visar kursgraf, trade-events och eventuella indikatorer
# -------------------------------------------------------------------------
class PricePanel:
    def __init__(self, rect: Union[Tuple[int, int, int, int], pygame.Rect]):
        self.rect = pygame.Rect(rect)
        self.zoom = 1.0
        self.offset = 0
        self.show_indicators = False
        self.current_indicator = None
        self.font = None
        # Nya färger för Kelly-trades
        self.kelly_buy_color = KELLY_BUY_COLOR  # Ljusare grön för Kelly-köp
        self.position_legend_shown = False
        # Risk management-relaterade inställningar
        self.show_risk_levels = True  # Visa stop-loss/take-profit linjer
        self.stop_loss_color = STOP_LOSS_COLOR
        self.take_profit_color = TAKE_PROFIT_COLOR
        self.trailing_stop_color = TRAILING_STOP_COLOR
        self.trailing_take_color = TRAILING_TAKE_COLOR
        
        # Improve legend spacing and position
        self.legend_padding = 10
        self.legend_item_height = 22
        self.legend_box_padding = 5

    def set_font(self, font: pygame.font.Font) -> None:
        self.font = font

    def _draw_placeholder(self, surface: pygame.Surface, message: str) -> None:
        """Draw a placeholder message when data cannot be displayed"""
        if self.font:
            x, y, w, h = self.rect
            text_surf = self.font.render(message, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(x + w//2, y + h//2))
            surface.blit(text_surf, text_rect)

    def draw(self, 
            surface: pygame.Surface, 
            env: Optional[EnvironmentInterface], 
            current_step: int, 
            trade_events: List[Tuple]) -> None:
        """
        Draw the price chart with indicators, trade events, and risk management levels
        
        Args:
            surface: Pygame surface to draw on
            env: Trading environment or None
            current_step: Current step in the episode
            trade_events: List of trade events
        """
        try:
            # Draw panel background with subtle border
            pygame.draw.rect(surface, PANEL_BG_COLOR, self.rect, border_radius=10)
            pygame.draw.rect(surface, (80, 80, 80), self.rect, width=1, border_radius=10)
            
            # Early return if environment is not available
            if env is None:
                self._draw_placeholder(surface, "Environment not available")
                return
                
            # Check for required data
            if current_step < 2 or not hasattr(env, 'prices') or len(env.prices) == 0:
                self._draw_placeholder(surface, "Insufficient price data")
                return
                
            x, y, w, h = self.rect
            safe_step = min(current_step, len(env.prices) - 1)
            episode_prices = env.prices[:safe_step+1]
            
            if len(episode_prices) < 2:
                self._draw_placeholder(surface, "Need at least 2 price points")
                return

            # Calculate visible price range
            visible_points = min(int(len(episode_prices) / self.zoom), len(episode_prices))
            start_idx = max(0, min(len(episode_prices) - visible_points, self.offset))
            end_idx = min(start_idx + visible_points, len(episode_prices))
            visible_prices = episode_prices[start_idx:end_idx]
            
            if len(visible_prices) < 2:
                self._draw_placeholder(surface, "Not enough visible price points")
                return

            # Find price range with safety checks
            min_price = np.min(visible_prices) * 0.99
            max_price = np.max(visible_prices) * 1.01
            price_range = max_price - min_price if max_price != min_price else 1

            # Draw grid lines with labels
            grid_lines = 5
            for i in range(grid_lines + 1):
                grid_y = y + h * i / grid_lines
                pygame.draw.line(surface, CHART_GRID_COLOR, (x, grid_y), (x + w, grid_y), 1)
                price_at_line = max_price - (i / grid_lines) * price_range
                if self.font:
                    label = self.font.render(f"{price_at_line:.2f}", True, TEXT_COLOR)
                    # Position price labels with proper spacing
                    label_x = x + w + 5
                    label_y = grid_y - (label.get_height() // 2)
                    surface.blit(label, (label_x, label_y))

            # Draw vertical grid lines
            for i in range(5):
                grid_x = x + w * i / 4
                pygame.draw.line(surface, CHART_GRID_COLOR, (grid_x, y), (grid_x, y + h), 1)
                
                # Add time/step indicators at the bottom
                if self.font and i < 4:  # Skip the last one to avoid overlap
                    step_at_line = start_idx + int((end_idx - start_idx) * (i / 4))
                    step_label = self.font.render(f"Step {step_at_line}", True, TEXT_COLOR)
                    step_label_x = grid_x - (step_label.get_width() // 2)
                    step_label_y = y + h + 5
                    surface.blit(step_label, (step_label_x, step_label_y))

            # Draw price line
            points = []
            for i, price in enumerate(visible_prices):
                px = x + (i / (len(visible_prices) - 1)) * w if len(visible_prices) > 1 else x + w/2
                py = y + h - ((price - min_price) / price_range * h)
                points.append((px, py))
                
            if len(points) >= 2:
                pygame.draw.lines(surface, LINE_COLOR, False, points, 2)

            # Draw selected indicator if enabled
            if self.show_indicators and self.current_indicator:
                indicator_values = env.indicator_values.get(self.current_indicator, [])
                if len(indicator_values) > safe_step:
                    indicator_slice = indicator_values[:safe_step+1][start_idx:end_idx]
                    if len(indicator_slice) > 0:
                        min_ind = np.min(indicator_slice)
                        max_ind = np.max(indicator_slice)
                        
                        if min_ind != max_ind:  # Avoid division by zero
                            # Normalize indicator values to price scale
                            norm_values = (indicator_slice - min_ind) / (max_ind - min_ind)
                            norm_values = norm_values * price_range + min_price
                            
                            # Draw indicator line
                            ind_points = []
                            for i, val in enumerate(norm_values):
                                px = x + (i / (len(visible_prices) - 1)) * w if len(visible_prices) > 1 else x + w/2
                                py = y + h - ((val - min_price) / price_range * h)
                                ind_points.append((px, py))
                                
                            if len(ind_points) >= 2:
                                pygame.draw.lines(surface, (255, 165, 0), False, ind_points, 2)
                                
                                # Draw indicator label with background for better visibility
                                if self.font and len(ind_points) > 0:
                                    ind_label = self.font.render(self.current_indicator, True, (255, 165, 0))
                                    label_bg_rect = pygame.Rect(
                                        ind_points[0][0], y + 10, 
                                        ind_label.get_width() + 6, ind_label.get_height() + 4
                                    )
                                    pygame.draw.rect(surface, (40, 40, 40, 180), label_bg_rect, border_radius=3)
                                    surface.blit(ind_label, (ind_points[0][0] + 3, y + 12))

            # Draw stop-loss and take-profit levels if they exist and should be shown
            if self.show_risk_levels and hasattr(env, 'position') and env.position is not None:
                position = env.position
                if position.active_size > 0:
                    # Hämta stop-loss och take-profit nivåer
                    stop_loss_level = position.stop_loss_level
                    take_profit_level = position.take_profit_level
                    is_trailing_stop = position.is_trailing_stop
                    is_trailing_take = position.is_trailing_take
                    
                    # Rita stop-loss om den finns
                    if stop_loss_level is not None and min_price <= stop_loss_level <= max_price:
                        # Beräkna y-position för stop-loss nivån
                        stop_y = y + h - ((stop_loss_level - min_price) / price_range * h)
                        # Använd olika färger för vanlig och glidande stop-loss
                        stop_color = self.trailing_stop_color if is_trailing_stop else self.stop_loss_color
                        # Rita linjen
                        pygame.draw.line(surface, stop_color, (x, stop_y), (x + w, stop_y), 2)
                        # Lägg till etikett om font finns
                        if self.font:
                            label_text = f"SL: {stop_loss_level:.2f}" + (" (T)" if is_trailing_stop else "")
                            label = self.font.render(label_text, True, stop_color)
                            
                            # Create background rectangle for better readability
                            label_bg_rect = pygame.Rect(
                                x + 5, stop_y - 20,
                                label.get_width() + 6, label.get_height() + 4
                            )
                            pygame.draw.rect(surface, (40, 40, 40, 180), label_bg_rect, border_radius=3)
                            
                            surface.blit(label, (x + 8, stop_y - 18))
                    
                    # Rita take-profit om den finns
                    if take_profit_level is not None and min_price <= take_profit_level <= max_price:
                        # Beräkna y-position för take-profit nivån
                        take_y = y + h - ((take_profit_level - min_price) / price_range * h)
                        # Använd olika färger för vanlig och glidande take-profit
                        take_color = self.trailing_take_color if is_trailing_take else self.take_profit_color
                        # Rita linjen
                        pygame.draw.line(surface, take_color, (x, take_y), (x + w, take_y), 2)
                        # Lägg till etikett om font finns
                        if self.font:
                            label_text = f"TP: {take_profit_level:.2f}" + (" (T)" if is_trailing_take else "")
                            label = self.font.render(label_text, True, take_color)
                            
                            # Create background rectangle for better readability
                            label_bg_rect = pygame.Rect(
                                x + w - label.get_width() - 15, take_y - 20,
                                label.get_width() + 6, label.get_height() + 4
                            )
                            pygame.draw.rect(surface, (40, 40, 40, 180), label_bg_rect, border_radius=3)
                            
                            surface.blit(label, (x + w - label.get_width() - 12, take_y - 18))

            # Draw trade events with variable circle sizes for Kelly positions
            has_kelly_trades = False
            for event in trade_events:
                # Handle both gamla (3 värden) och nya (5 värden) trade_events format
                if len(event) >= 4:
                    step, price, action, position_size = event[0], event[1], event[2], event[3]
                    # Kolla om det är ett Kelly-baserat köp (action == 3)
                    is_kelly_trade = action == 3
                    if is_kelly_trade:
                        has_kelly_trades = True
                else:
                    # Bakåtkompatibilitet med gamla event-formatet
                    step, price, action = event[0], event[1], event[2]
                    position_size = 1.0  # Anta full position för gamla event
                    is_kelly_trade = False
                
                if step < start_idx or step >= end_idx:
                    continue
                    
                tx = x + ((step - start_idx) / (len(visible_prices) - 1)) * w if len(visible_prices) > 1 else x + w/2
                ty = y + h - ((price - min_price) / price_range * h)
                
                # Anpassa cirkelstorlek baserat på position_size (minsta storlek 3, max 8)
                radius = 3 + 5 * min(1.0, max(0.0, position_size))
                
                # Använd olika färger beroende på action
                if action == 1:  # Normal köp
                    color = TRADE_BUY_COLOR
                elif action == 3:  # Kelly-köp
                    color = self.kelly_buy_color
                else:  # Sälj
                    color = TRADE_SELL_COLOR
                
                # Draw circle with a subtle border for better visibility
                pygame.draw.circle(surface, color, (int(tx), int(ty)), int(radius))
                pygame.draw.circle(surface, (40, 40, 40), (int(tx), int(ty)), int(radius), width=1)
            
            # Draw legend for different types of trades and risk management
            # First, calculate the number of legend items to display
            legend_items = []
            legend_items.append(("Köp (full position)", TRADE_BUY_COLOR))
            if has_kelly_trades:
                legend_items.append(("Köp (Kelly position)", self.kelly_buy_color))
            legend_items.append(("Sälj", TRADE_SELL_COLOR))
            
            # Add risk management items if relevant
            if self.show_risk_levels and hasattr(env, 'position') and env.position is not None:
                position = env.position
                if position.stop_loss_level is not None:
                    stop_color = self.trailing_stop_color if position.is_trailing_stop else self.stop_loss_color
                    legend_items.append(("Stop-Loss" + (" (Trailing)" if position.is_trailing_stop else ""), stop_color))
                if position.take_profit_level is not None:
                    take_color = self.trailing_take_color if position.is_trailing_take else self.take_profit_color
                    legend_items.append(("Take-Profit" + (" (Trailing)" if position.is_trailing_take else ""), take_color))
            
            # Only draw legend if we have items and font
            if legend_items and self.font:
                # Calculate legend box dimensions
                max_text_width = max([self.font.size(item[0])[0] for item in legend_items])
                legend_width = max_text_width + 30  # Circle/line + padding
                legend_height = len(legend_items) * self.legend_item_height + 2 * self.legend_box_padding
                
                # Draw legend background
                legend_rect = pygame.Rect(
                    x + self.legend_padding, 
                    y + self.legend_padding,
                    legend_width,
                    legend_height
                )
                pygame.draw.rect(surface, (40, 40, 40, 180), legend_rect, border_radius=5)
                pygame.draw.rect(surface, (80, 80, 80), legend_rect, width=1, border_radius=5)
                
                # Draw legend items
                for i, (text, color) in enumerate(legend_items):
                    item_y = y + self.legend_padding + self.legend_box_padding + i * self.legend_item_height
                    
                    # Draw circle or line based on item type
                    if "Stop-Loss" in text or "Take-Profit" in text:
                        # Draw line for stop-loss/take-profit
                        line_y = item_y + self.legend_item_height // 2
                        pygame.draw.line(
                            surface, 
                            color, 
                            (legend_rect.x + 10, line_y),
                            (legend_rect.x + 20, line_y),
                            2
                        )
                    else:
                        # Draw circle for buy/sell
                        pygame.draw.circle(
                            surface,
                            color,
                            (legend_rect.x + 15, item_y + self.legend_item_height // 2),
                            5
                        )
                    
                    # Draw text
                    text_surf = self.font.render(text, True, TEXT_COLOR)
                    surface.blit(text_surf, (legend_rect.x + 25, item_y + (self.legend_item_height - text_surf.get_height()) // 2))

            # Draw current step indicator
            if start_idx <= safe_step < end_idx:
                cx = x + ((safe_step - start_idx) / (len(visible_prices) - 1)) * w if len(visible_prices) > 1 else x + w/2
                pygame.draw.line(surface, (255, 255, 255), (cx, y), (cx, y + h), 1)
                
        except Exception as e:
            logger.error(f"Error drawing price panel: {e}")
            self._draw_placeholder(surface, f"Chart error: {str(e)}")

# -------------------------------------------------------------------------
# RewardPanel: visar reward-history som linje
# -------------------------------------------------------------------------
class RewardPanel:
    def __init__(self, rect: Union[Tuple[int, int, int, int], pygame.Rect]):
        self.rect = pygame.Rect(rect)
        self.font = None

    def set_font(self, font: pygame.font.Font) -> None:
        self.font = font

    def _draw_placeholder(self, surface: pygame.Surface, message: str) -> None:
        """Draw a placeholder message when data cannot be displayed"""
        if self.font:
            x, y, w, h = self.rect
            text_surf = self.font.render(message, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(x + w//2, y + h//2))
            surface.blit(text_surf, text_rect)

    def draw(self, surface: pygame.Surface, reward_history: List[float], running_avg_window: int = 10) -> None:
        """
        Draw the reward history chart with running average
        
        Args:
            surface: Pygame surface to draw on
            reward_history: List of rewards from previous episodes
            running_avg_window: Window size for calculating running average
        """
        try:
            # Draw panel background with subtle border
            pygame.draw.rect(surface, PANEL_BG_COLOR, self.rect, border_radius=10)
            pygame.draw.rect(surface, (80, 80, 80), self.rect, width=1, border_radius=10)
            
            if len(reward_history) < 2:
                self._draw_placeholder(surface, "Insufficient reward data")
                return
                
            x, y, w, h = self.rect
            
            # Add panel title
            if self.font:
                title = self.font.render("Reward History", True, TEXT_COLOR)
                title_x = x + 10
                title_y = y + 5
                surface.blit(title, (title_x, title_y))
            
            # Adjust chart area to accommodate title
            chart_y = y + 30
            chart_h = h - 30
            
            # Calculate min/max with safety check
            min_r = min(reward_history) if reward_history else 0
            max_r = max(reward_history) if reward_history else 0
            if min_r == max_r:  # Avoid division by zero
                r_range = 1
            else:
                r_range = max_r - min_r
                
            # Add some padding to the range
            r_range *= 1.1
            min_r -= r_range * 0.05
            max_r += r_range * 0.05
            r_range = max_r - min_r

            # Draw grid with improved visibility
            grid_lines = 4
            for i in range(grid_lines + 1):
                grid_y = chart_y + chart_h * i / grid_lines
                pygame.draw.line(surface, CHART_GRID_COLOR, (x, grid_y), (x + w, grid_y), 1)
                
                # Draw reward values at grid lines if font is available
                if self.font:
                    value_at_line = max_r - (i / grid_lines) * r_range
                    label = self.font.render(f"{value_at_line:.1f}", True, TEXT_COLOR)
                    # Position labels with proper spacing
                    label_x = x - label.get_width() - 5
                    label_y = grid_y - (label.get_height() // 2)
                    surface.blit(label, (label_x, label_y))
                    
            # Draw vertical grid lines with episode markers
            episodes_between_lines = max(1, len(reward_history) // 5)
            num_vertical_lines = min(5, len(reward_history) // episodes_between_lines)
            
            for i in range(num_vertical_lines + 1):
                if i == num_vertical_lines:  # Last line at the right edge
                    grid_x = x + w
                    ep_num = len(reward_history) - 1
                else:
                    grid_x = x + (w * i) / num_vertical_lines
                    ep_num = i * episodes_between_lines
                
                # Draw vertical grid line
                pygame.draw.line(surface, CHART_GRID_COLOR, (grid_x, chart_y), (grid_x, chart_y + chart_h), 1)
                
                # Draw episode numbers if font is available
                if self.font and ep_num < len(reward_history):
                    ep_label = self.font.render(f"Ep {ep_num}", True, TEXT_COLOR)
                    # Center label under grid line
                    label_x = grid_x - (ep_label.get_width() // 2)
                    label_y = chart_y + chart_h + 5
                    surface.blit(ep_label, (label_x, label_y))

            # Draw reward line with improved visibility
            points = []
            for i, r in enumerate(reward_history):
                rx = x + (i / (len(reward_history) - 1)) * w
                ry = chart_y + chart_h - ((r - min_r) / r_range * chart_h)
                points.append((rx, ry))
                
            if len(points) >= 2:
                pygame.draw.lines(surface, REWARD_LINE_COLOR, False, points, 2)

            # Draw running average if enough data
            if len(reward_history) >= running_avg_window:
                running_avg = [np.mean(reward_history[max(0, i - running_avg_window):i + 1]) 
                              for i in range(len(reward_history))]
                               
                avg_points = []
                for i, r in enumerate(running_avg):
                    rx = x + (i / (len(running_avg) - 1)) * w
                    ry = chart_y + chart_h - ((r - min_r) / r_range * chart_h)
                    avg_points.append((rx, ry))
                    
                if len(avg_points) >= 2:
                    pygame.draw.lines(surface, (255, 165, 0), False, avg_points, 2)
                    
                    # Draw legend with improved visibility
                    if self.font:
                        # Create legend background
                        legend_items = [
                            ("Reward", REWARD_LINE_COLOR),
                            (f"Avg ({running_avg_window})", (255, 165, 0))
                        ]
                        
                        # Calculate legend box dimensions
                        max_text_width = max([self.font.size(item[0])[0] for item in legend_items])
                        legend_width = max_text_width + 35  # Line + padding
                        legend_height = len(legend_items) * 25 + 10
                        
                        legend_rect = pygame.Rect(
                            x + w - legend_width - 10,
                            chart_y + 10,
                            legend_width,
                            legend_height
                        )
                        
                        # Draw legend background
                        pygame.draw.rect(surface, (40, 40, 40, 180), legend_rect, border_radius=5)
                        pygame.draw.rect(surface, (80, 80, 80), legend_rect, width=1, border_radius=5)
                        
                        # Draw legend items
                        for i, (text, color) in enumerate(legend_items):
                            # Draw line sample
                            line_y = legend_rect.y + 15 + i * 25
                            pygame.draw.line(
                                surface,
                                color,
                                (legend_rect.x + 10, line_y),
                                (legend_rect.x + 25, line_y),
                                2
                            )
                            
                            # Draw text
                            text_surf = self.font.render(text, True, TEXT_COLOR)
                            surface.blit(text_surf, (legend_rect.x + 30, line_y - text_surf.get_height() // 2))
                        
        except Exception as e:
            logger.error(f"Error drawing reward panel: {e}")
            self._draw_placeholder(surface, f"Reward chart error: {str(e)}")

# -------------------------------------------------------------------------
# PatternPanel: visar sparade framgångsrika handelsmönster
# -------------------------------------------------------------------------
class PatternPanel:
    """Panel för att visa sparade framgångsrika handelsmönster"""
    
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.font = None
        self.small_font = None
        self.header_font = None
        self.patterns_to_show = 5  # Antal mönster att visa samtidigt
        self.current_page = 0
        self.sort_by = "similarity"  # Standardsortering: similarity, profit, timestamp
        
        # Förenkla vilka features som visas - endast de mest relevanta
        self.feature_names = [
            "Pris", "Position", "Prisändring", "RSI", "Momentum", "Volatilitet"
        ]
        
        # Bara 3-4 viktiga features att visa
        self.important_features = [0, 1, 3, 4]  # Pris, Position, RSI, Momentum
        
        # Färgschema för tydligare visualisering
        self.header_bg = (45, 55, 70)  # Mörkblå för header
        self.positive_color = (100, 255, 100)  # Ljusgrön för positiva värden
        self.negative_color = (255, 100, 100)  # Ljusröd för negativa värden
        self.neutral_color = (220, 220, 220)  # Ljusgrå för neutrala värden
        self.highlight_color = (255, 215, 0)  # Guld för viktiga värden
        
    def set_fonts(self, font, small_font):
        self.font = font
        self.small_font = small_font
        # Skapa header_font baserat på font men något större
        self.header_font = pygame.font.Font(None, int(font.get_height() * 1.2))
        
    def get_action_name(self, action_code):
        """Returnerar en läsbar beskrivning av en action-kod"""
        action_names = {
            0: "Håll",
            1: "Köp (Full)",
            2: "Sälj allt",
            3: "Köp (Kelly)",
            4: "Sätt SL",
            5: "Sätt TSL",
            6: "Sätt TP",
            7: "Sätt TTP",
            8: "Sälj 25%",
            9: "Sälj 50%",
            10: "Sälj 75%",
            11: "Ta bort SL",
            12: "Ta bort TP"
        }
        return action_names.get(action_code, f"A{action_code}")
    
    def draw(self, surface, env, current_state=None):
        """Ritar panelen med förenklad, tydligare layout"""
        # Rita panel-bakgrund med subtil kant
        pygame.draw.rect(surface, (40, 40, 45), self.rect, border_radius=10)
        pygame.draw.rect(surface, (80, 80, 80), self.rect, width=1, border_radius=10)
        
        if not env or not hasattr(env, 'successful_trade_patterns') or not self.font:
            # Rita placeholder om data saknas
            text = self.font.render("Inga sparade mönster tillgängliga", True, (200, 200, 200))
            text_rect = text.get_rect(center=(self.rect.centerx, self.rect.centery))
            surface.blit(text, text_rect)
            return
            
        patterns = env.successful_trade_patterns
        if not patterns:
            text = self.font.render("Inga sparade mönster än", True, (200, 200, 200))
            text_rect = text.get_rect(center=(self.rect.centerx, self.rect.centery))
            surface.blit(text, text_rect)
            return
            
        # Beräkna similaritet om tillgänglig
        if current_state is not None and hasattr(env, 'calculate_state_similarity'):
            try:
                for pattern in patterns:
                    similarity = env.calculate_state_similarity(pattern.get('state', []), current_state)
                    pattern['similarity'] = similarity
            except:
                pass
                
        # Sortera mönster baserat på läge
        if self.sort_by == "similarity" and any('similarity' in p for p in patterns):
            sorted_patterns = sorted(patterns, key=lambda p: p.get('similarity', 0), reverse=True)
        elif self.sort_by == "profit":
            sorted_patterns = sorted(patterns, key=lambda p: p.get('profit', 0), reverse=True)
        elif self.sort_by == "timestamp":
            sorted_patterns = sorted(patterns, key=lambda p: p.get('timestamp', 0), reverse=True)
        else:
            # Kombinerad sortering som standard
            sorted_patterns = sorted(patterns, 
                                    key=lambda p: (p.get('similarity', 0) * 0.7 + p.get('profit', 0) * 0.3), 
                                    reverse=True)
        
        # Rita header
        header_height = 30
        header_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, header_height)
        pygame.draw.rect(surface, self.header_bg, header_rect)
        
        # Rita titel
        title = self.header_font.render("Framgångsrika Mönster", True, (220, 220, 220))
        surface.blit(title, (header_rect.x + 10, header_rect.y + (header_height - title.get_height()) // 2))
        
        # Rita sidnavigering
        pattern_count = len(sorted_patterns)
        max_pages = max(1, (pattern_count + self.patterns_to_show - 1) // self.patterns_to_show)
        self.current_page = min(self.current_page, max_pages - 1)
        
        page_text = f"{self.current_page + 1}/{max_pages}"
        page_label = self.small_font.render(page_text, True, (200, 200, 200))
        surface.blit(page_label, (header_rect.right - page_label.get_width() - 60, 
                               header_rect.y + (header_height - page_label.get_height()) // 2))
        
        # Rita navigeringsknappar
        btn_width = 20
        btn_height = 20
        btn_y = header_rect.y + (header_height - btn_height) // 2
        
        prev_btn = pygame.Rect(header_rect.right - 45, btn_y, btn_width, btn_height)
        next_btn = pygame.Rect(header_rect.right - 20, btn_y, btn_width, btn_height)
        
        for btn, symbol in [(prev_btn, "<"), (next_btn, ">")]:
            pygame.draw.rect(surface, (60, 60, 70), btn)
            btn_text = self.small_font.render(symbol, True, (200, 200, 200))
            surface.blit(btn_text, (btn.x + (btn.width - btn_text.get_width()) // 2, 
                                  btn.y + (btn.height - btn_text.get_height()) // 2))
        
        # Beräkna vilka mönster att visa
        start_idx = self.current_page * self.patterns_to_show
        end_idx = min(start_idx + self.patterns_to_show, pattern_count)
        visible_patterns = sorted_patterns[start_idx:end_idx]
        
        # Beräkna layoutparametrar
        content_rect = pygame.Rect(
            self.rect.x, 
            header_rect.bottom, 
            self.rect.width, 
            self.rect.height - header_height
        )
        
        card_height = min(60, (content_rect.height - 10) // self.patterns_to_show)
        card_spacing = 5
        
        # Rita varje mönsterkort
        for i, pattern in enumerate(visible_patterns):
            card_y = content_rect.y + i * (card_height + card_spacing) + card_spacing
            card_rect = pygame.Rect(
                content_rect.x + 5, 
                card_y,
                content_rect.width - 10,
                card_height
            )
            
            # Hämta information från mönstret
            action = pattern.get('action', 0)
            profit = pattern.get('profit', 0) * 100  # Konvertera till procent
            similarity = pattern.get('similarity', 0)
            
            # Bestäm bakgrundsfärg baserat på vinst
            if profit >= 1.0:
                bg_color = (30, 60, 30)  # Mörkgrön för bra vinst
            elif profit > 0:
                bg_color = (30, 50, 30)  # Ljusare grön för liten vinst
            else:
                bg_color = (60, 30, 30)  # Röd för förlust
            
            # Rita mönsterkortet med rundade hörn och border
            pygame.draw.rect(surface, bg_color, card_rect, border_radius=5)
            
            # Justera border-färg baserat på likhet
            if similarity > 0.8:
                border_color = (255, 215, 0)  # Guld för hög likhet
                border_width = 2
            elif similarity > 0.6:
                border_color = (180, 180, 200)  # Silver för medium likhet
                border_width = 1
            else:
                border_color = (100, 100, 100)  # Mörkgrå för låg likhet
                border_width = 1
                
            pygame.draw.rect(surface, border_color, card_rect, width=border_width, border_radius=5)
            
            # Rita informationstext
            text_color = (220, 220, 220)
            action_name = self.get_action_name(action)
            
            # Left column: Action & Profit
            action_x = card_rect.x + 10
            action_y = card_rect.y + 5
            
            # Action med tydlig indikator
            action_text = f"Action: {action_name}"
            action_label = self.small_font.render(action_text, True, text_color)
            surface.blit(action_label, (action_x, action_y))
            
            # Profit med färgindikation
            profit_y = action_y + action_label.get_height() + 5
            profit_text = f"Profit: "
            profit_label = self.small_font.render(profit_text, True, text_color)
            
            profit_value_color = self.positive_color if profit > 0 else self.negative_color
            profit_value_text = f"{profit:.2f}%"
            profit_value = self.small_font.render(profit_value_text, True, profit_value_color)
            
            surface.blit(profit_label, (action_x, profit_y))
            surface.blit(profit_value, (action_x + profit_label.get_width(), profit_y))
            
            # Right column: Similarity
            right_col_x = card_rect.centerx + 20
            
            # Likhet med färgindikation
            similarity_text = f"Likhet: "
            similarity_label = self.small_font.render(similarity_text, True, text_color)
            
            similarity_value_color = self.highlight_color if similarity > 0.7 else text_color
            similarity_value_text = f"{similarity:.2f}"
            similarity_value = self.small_font.render(similarity_value_text, True, similarity_value_color)
            
            surface.blit(similarity_label, (right_col_x, action_y))
            surface.blit(similarity_value, (right_col_x + similarity_label.get_width(), action_y))
            
            # Visa 1-2 viktiga features om det finns utrymme
            state = pattern.get('state', None)
            if state is not None and len(state) > 0 and card_height >= 50:
                feature_y = profit_y + profit_label.get_height() + 2
                
                # Välj de viktigaste features att visa
                feature_idx = self.important_features[0]  # Bara visa 1 feature
                if feature_idx < len(state):
                    feature_name = self.feature_names[min(feature_idx, len(self.feature_names)-1)]
                    feature_value = state[feature_idx]
                    
                    # Visa feature med värde
                    if isinstance(feature_value, float):
                        feature_text = f"{feature_name}: {feature_value:.2f}"
                    else:
                        feature_text = f"{feature_name}: {feature_value}"
                        
                    feature_color = text_color
                    if isinstance(feature_value, float):
                        if feature_value > 0:
                            feature_color = self.positive_color
                        elif feature_value < 0:
                            feature_color = self.negative_color
                    
                    feature_label = self.small_font.render(feature_text, True, feature_color)
                    surface.blit(feature_label, (right_col_x, feature_y))
    
    def handle_events(self, event, mouse_pos):
        """Hanterar interaktioner med panelen"""
        if not self.rect.collidepoint(mouse_pos):
            return False
            
        # Hantera knappklick
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Header-relaterade knappar
            header_height = 30
            header_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, header_height)
            
            # Previous button
            prev_btn = pygame.Rect(header_rect.right - 45, 
                                  header_rect.y + (header_height - 20) // 2, 
                                  20, 20)
            if prev_btn.collidepoint(mouse_pos) and self.current_page > 0:
                self.current_page -= 1
                return True
                
            # Next button
            next_btn = pygame.Rect(header_rect.right - 20, 
                                  header_rect.y + (header_height - 20) // 2, 
                                  20, 20)
            if next_btn.collidepoint(mouse_pos):
                self.current_page += 1
                return True
                
            # Klick på titeln för att ändra sortering
            title_area = pygame.Rect(header_rect.x, header_rect.y, header_rect.width // 2, header_rect.height)
            if title_area.collidepoint(mouse_pos):
                # Växla sorteringsmetod
                if self.sort_by == "similarity":
                    self.sort_by = "profit"
                elif self.sort_by == "profit":
                    self.sort_by = "timestamp"
                else:
                    self.sort_by = "similarity"
                self.current_page = 0  # Återställ sidan vid ändrad sortering
                return True
                
        return False

# -------------------------------------------------------------------------
# UIManager - hanterar pre-run-menyn och huvud-UI
# -------------------------------------------------------------------------
class UIManager:
    def __init__(self, 
                screen: pygame.Surface, 
                instruments: List[str], 
                initial_instrument_index: int = 0, 
                default_episodes: int = 100):
        from config import args
        self.screen = screen
        self.screen_width, self.screen_height = screen.get_size()

        # ---------- PRE-RUN MENY ----------
        self.show_pre_run_menu = True
        self.pre_run_choice: Optional[str] = None  # 'train' eller 'optimize'
        self.use_optimized = False  # Toggle för att använda senaste optimerade parametrar vid träning
        self.use_kelly = True  # Toggle för Kelly-kriteriet
        self.use_risk_management = True  # Ny toggle för risk management

        # Initialize fonts with error handling
        try:
            self.font_small = pygame.font.SysFont("Arial", int(self.screen_height * 0.025))
            self.font_header = pygame.font.SysFont("Arial", int(self.screen_height * 0.03))
            self.font_title = pygame.font.SysFont("Arial", int(self.screen_height * 0.04), bold=True)
        except Exception as e:
            logger.error(f"Failed to initialize fonts: {e}")
            # Fallback to default font
            self.font_small = pygame.font.Font(None, 24)
            self.font_header = pygame.font.Font(None, 30)
            self.font_title = pygame.font.Font(None, 36)

        # Inmatning för antal episoder och andra parametrar
        self.n_episodes = default_episodes
        self.n_trials = 50  # Default antal trials för optimering
        self.max_kelly_fraction = 0.5  # Default värde för max Kelly-fraktion
        
        # Nya risk management parametrar
        self.default_stop_loss_pct = 0.05  # Default värde (5%)
        self.default_take_profit_pct = 0.10  # Default värde (10%)
        self.default_trailing_pct = 0.03  # Default värde (3%)
        
        # Calculate layout for pre-run menu
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Improved layout with more spacing
        input_width = 150
        input_height = 35
        label_width = 250
        label_spacing = 45  # More space between inputs
        col_spacing = 20   # Space between label and input box
        
        # Row positions (y-coordinates)
        row1_y = center_y - 180
        row2_y = row1_y + label_spacing
        row3_y = row2_y + label_spacing
        row4_y = row3_y + label_spacing
        row5_y = row4_y + label_spacing
        
        # Button positions
        button_y = row5_y + 60
        toggle_y = button_y + 50
        toggle_spacing = 40
        
        # Create input boxes with improved positioning
        self.episode_input_box = TextInputBox(
            rect=(center_x + col_spacing, row1_y, input_width, input_height),
            font=self.font_small,
            text=str(self.n_episodes)
        )
        
        self.trials_input_box = TextInputBox(
            rect=(center_x + col_spacing, row2_y, input_width, input_height),
            font=self.font_small,
            text=str(self.n_trials)
        )
        
        self.kelly_fraction_input_box = TextInputBox(
            rect=(center_x + col_spacing, row3_y, input_width, input_height),
            font=self.font_small,
            text=str(self.max_kelly_fraction)
        )
        
        self.stop_loss_input_box = TextInputBox(
            rect=(center_x + col_spacing, row4_y, input_width, input_height),
            font=self.font_small,
            text=str(self.default_stop_loss_pct)
        )
        
        self.take_profit_input_box = TextInputBox(
            rect=(center_x + col_spacing, row5_y, input_width, input_height),
            font=self.font_small,
            text=str(self.default_take_profit_pct)
        )

        # Create buttons with improved layout
        self.start_button = Button(
            rect=(center_x - 170, button_y, 150, 40),
            text="Starta träning",
            font=self.font_small
        )

        self.optimize_button = Button(
            rect=(center_x + 20, button_y, 150, 40),
            text="Kör optimering",
            font=self.font_small
        )

        # Create toggle buttons with improved spacing
        self.toggle_optimized_button = Button(
            rect=(center_x - 170, toggle_y, 340, 35),
            text="Använd senaste optimering: Av",
            font=self.font_small
        )
        
        self.toggle_kelly_button = Button(
            rect=(center_x - 170, toggle_y + toggle_spacing, 340, 35),
            text="Använd Kelly-kriteriet: På",
            font=self.font_small
        )
        
        self.toggle_risk_mgmt_button = Button(
            rect=(center_x - 170, toggle_y + toggle_spacing * 2, 340, 35),
            text="Använd Risk Management: På",
            font=self.font_small
        )

        # ---------- HUVUD-UI ----------
        self.instruments = instruments
        self.instrument_index = initial_instrument_index
        self.model_type = args.model_type
        self.paused = False
        self.indicator_index = 0
        self.show_patterns = False  # Toggle för att visa mönsterpanelen

        # Improved UI layout calculations
        header_height = int(self.screen_height * 0.18)
        side_panel_width = int(self.screen_width * 0.18)
        main_panel_width = self.screen_width - side_panel_width - 20  # Margins
        
        self.header_rect = (0, 0, self.screen_width, header_height)
        self.price_panel_rect = (
            10, 
            header_height + 10,
            main_panel_width,
            int(self.screen_height * 0.45)
        )
        self.reward_panel_rect = (
            10, 
            header_height + int(self.screen_height * 0.45) + 20,
            main_panel_width,
            int(self.screen_height * 0.30)
        )
        
        self.pattern_panel_rect = (
            10, 
            header_height + int(self.screen_height * 0.45) + 20,
            main_panel_width,
            int(self.screen_height * 0.30)
        )

        # Initialize UI components
        self.header = Header(self.header_rect, self.font_header)
        self.price_panel = PricePanel(self.price_panel_rect)
        self.price_panel.set_font(self.font_small)
        self.reward_panel = RewardPanel(self.reward_panel_rect)
        self.reward_panel.set_font(self.font_small)
        self.pattern_panel = PatternPanel(self.pattern_panel_rect)
        self.pattern_panel.set_fonts(self.font_small, self.font_small)

        # Create buttons with improved layout
        button_width = side_panel_width - 10
        button_height = 40
        button_x = self.screen_width - side_panel_width - 5
        button_spacing = 5
        
        # Function to calculate button y-position
        def get_button_y(index):
            return header_height + 10 + (button_height + button_spacing) * index
        
        self.stop_button = Button(
            (button_x, get_button_y(0), button_width, button_height),
            "Stop", self.font_small)

        self.pause_button = Button(
            (button_x, get_button_y(1), button_width, button_height),
            "Pause", self.font_small)

        self.switch_button = Button(
            (button_x, get_button_y(2), button_width, button_height),
            "Byt instrument", self.font_small)

        # For zoom buttons, split the width into two
        zoom_button_width = (button_width - 5) // 2
        self.zoom_in_button = Button(
            (button_x, get_button_y(3), zoom_button_width, button_height),
            "Zoom +", self.font_small)

        self.zoom_out_button = Button(
            (button_x + zoom_button_width + 5, get_button_y(3), zoom_button_width, button_height),
            "Zoom -", self.font_small)

        self.toggle_indicators_button = Button(
            (button_x, get_button_y(4), button_width, button_height),
            "Indikatorer", self.font_small)

        self.cycle_indicator_button = Button(
            (button_x, get_button_y(5), button_width, button_height),
            "Nästa ind.", self.font_small)

        self.save_model_button = Button(
            (button_x, get_button_y(6), button_width, button_height),
            "Spara modell", self.font_small)
            
        self.toggle_kelly_runtime_button = Button(
            (button_x, get_button_y(7), button_width, button_height),
            "Kelly: På", self.font_small)
            
        self.toggle_risk_mgmt_runtime_button = Button(
            (button_x, get_button_y(8), button_width, button_height),
            "Risk Mgmt: På", self.font_small)
            
        self.toggle_risk_levels_button = Button(
            (button_x, get_button_y(9), button_width, button_height),
            "Visa SL/TP", self.font_small)
            
        self.toggle_patterns_button = Button(
            (button_x, get_button_y(10), button_width, button_height),
            "Visa Mönster", self.font_small)

    def draw_pre_run_menu(self) -> None:
        """Draw the pre-run menu for selecting training options"""
        try:
            self.screen.fill(BACKGROUND_COLOR)
            center_x = self.screen_width // 2
            
            # Draw title with improved styling
            title_surf = self.font_title.render("Trading Agent - Konfiguration", True, TEXT_COLOR)
            title_rect = title_surf.get_rect(center=(center_x, 80))
            self.screen.blit(title_surf, title_rect)

            # Layout calculations for improved readability
            label_x = center_x - 230  # Right-align labels
            input_x = center_x + 20   # Left-align input boxes
            row1_y = 180              # Starting Y position
            row_spacing = 45          # Vertical spacing between rows
            
            # Draw input labels with right alignment
            labels = [
                "Antal episoder:",
                "Antal optimeringsförsök:",
                "Max Kelly-fraktion (0-1):",
                "Stop-loss % (0.01-0.15):",
                "Take-profit % (0.01-0.30):"
            ]
            
            for i, label_text in enumerate(labels):
                label_surf = self.font_small.render(label_text, True, TEXT_COLOR)
                label_rect = label_surf.get_rect(midright=(label_x, row1_y + i * row_spacing))
                self.screen.blit(label_surf, label_rect)
            
            # Draw input boxes
            self.episode_input_box.draw(self.screen)
            self.trials_input_box.draw(self.screen)
            self.kelly_fraction_input_box.draw(self.screen)
            self.stop_loss_input_box.draw(self.screen)
            self.take_profit_input_box.draw(self.screen)

            # Draw buttons with improved styling
            self.start_button.draw(self.screen)
            self.optimize_button.draw(self.screen)
            self.toggle_optimized_button.draw(self.screen)
            self.toggle_kelly_button.draw(self.screen)
            self.toggle_risk_mgmt_button.draw(self.screen)
            
            # Draw version and help text at the bottom
            footer_text = "V1.0 - Trading Agent med förbättrat UI"
            help_text = "Konfigurera parametrar och välj körläge"
            
            footer_surf = self.font_small.render(footer_text, True, (150, 150, 150))
            help_surf = self.font_small.render(help_text, True, (150, 150, 150))
            
            self.screen.blit(footer_surf, (20, self.screen_height - 50))
            self.screen.blit(help_surf, (20, self.screen_height - 30))

            pygame.display.flip()
        except Exception as e:
            logger.error(f"Error drawing pre-run menu: {e}")
            # Draw fallback menu
            self.screen.fill(BACKGROUND_COLOR)
            error_text = self.font_header.render(f"Error drawing menu: {str(e)}", True, (255, 0, 0))
            self.screen.blit(error_text, (50, 50))
            pygame.display.flip()

    def handle_pre_run_events(self) -> str:
        """
        Hanterar händelser i pre-run-menyn.
        Returnerar:
          - "train" om användaren valt att starta träning.
          - "optimize" om användaren valt att köra optimering.
          - "quit" vid avslut.
          - "continue" om ingen ändring gjorts.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "quit"

            # Handle text input for episodes
            result = self.episode_input_box.handle_event(event)
            if result is not None:
                try:
                    new_episodes = int(result)
                    if new_episodes > 0:
                        self.n_episodes = new_episodes
                    else:
                        logger.warning("Episode count must be positive")
                except ValueError:
                    logger.warning(f"Invalid episode count: {result}")
                    
            # Handle text input for trials
            result = self.trials_input_box.handle_event(event)
            if result is not None:
                try:
                    new_trials = int(result)
                    if new_trials > 0:
                        self.n_trials = new_trials
                    else:
                        logger.warning("Trial count must be positive")
                except ValueError:
                    logger.warning(f"Invalid trial count: {result}")
                    
            # Handle text input for Kelly fraction
            result = self.kelly_fraction_input_box.handle_event(event)
            if result is not None:
                try:
                    new_fraction = float(result)
                    if 0.0 <= new_fraction <= 1.0:
                        self.max_kelly_fraction = new_fraction
                    else:
                        logger.warning("Kelly fraction must be between 0 and 1")
                except ValueError:
                    logger.warning(f"Invalid Kelly fraction: {result}")
                    
            # Handle text input for stop-loss
            result = self.stop_loss_input_box.handle_event(event)
            if result is not None:
                try:
                    new_stop_loss = float(result)
                    if 0.01 <= new_stop_loss <= 0.15:
                        self.default_stop_loss_pct = new_stop_loss
                    else:
                        logger.warning("Stop-loss percentage must be between 0.01 and 0.15")
                except ValueError:
                    logger.warning(f"Invalid stop-loss percentage: {result}")
                    
            # Handle text input for take-profit
            result = self.take_profit_input_box.handle_event(event)
            if result is not None:
                try:
                    new_take_profit = float(result)
                    if 0.01 <= new_take_profit <= 0.30:
                        self.default_take_profit_pct = new_take_profit
                    else:
                        logger.warning("Take-profit percentage must be between 0.01 and 0.30")
                except ValueError:
                    logger.warning(f"Invalid take-profit percentage: {result}")

            # Check button clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.start_button.is_clicked(event):
                    self.pre_run_choice = "train"
                    self.show_pre_run_menu = False
                    logger.info(f"User selected training with {self.n_episodes} episodes, "
                               f"Kelly={self.use_kelly}, Risk Management={self.use_risk_management}, "
                               f"SL={self.default_stop_loss_pct}, TP={self.default_take_profit_pct}")
                    return "train"
                elif self.optimize_button.is_clicked(event):
                    self.pre_run_choice = "optimize"
                    self.show_pre_run_menu = False
                    logger.info(f"User selected hyperparameter optimization with {self.n_trials} trials")
                    return "optimize"
                elif self.toggle_optimized_button.is_clicked(event):
                    self.use_optimized = not self.use_optimized
                    status = "På" if self.use_optimized else "Av"
                    self.toggle_optimized_button.text = f"Använd senaste optimering: {status}"
                elif self.toggle_kelly_button.is_clicked(event):
                    self.use_kelly = not self.use_kelly
                    status = "På" if self.use_kelly else "Av"
                    self.toggle_kelly_button.text = f"Använd Kelly-kriteriet: {status}"
                elif self.toggle_risk_mgmt_button.is_clicked(event):
                    self.use_risk_management = not self.use_risk_management
                    status = "På" if self.use_risk_management else "Av"
                    self.toggle_risk_mgmt_button.text = f"Använd Risk Management: {status}"
        
        return "continue" if self.pre_run_choice is None else self.pre_run_choice

    def update_header(self, 
                     episode: int, 
                     env: Optional[EnvironmentInterface], 
                     agent: Optional[AgentInterface], 
                     reward_history: List[float]) -> List[str]:
        """
        Generate header text based on current state
        
        Args:
            episode: Current episode number
            env: Trading environment
            agent: DQN agent
            reward_history: List of rewards from previous episodes
            
        Returns:
            List of text lines to display in the header
        """
        try:
            if env is None:
                return ["Laddar data..."]

            if self.show_pre_run_menu:
                header_lines = [
                    "Pre-run: Välj körläge",
                    f"Episoder: {self.n_episodes}",
                    f"Optimeringsförsök: {self.n_trials}",
                    f"Max Kelly-fraktion: {self.max_kelly_fraction}",
                    f"Stop-loss %: {self.default_stop_loss_pct}",
                    f"Take-profit %: {self.default_take_profit_pct}",
                    f"Använd senaste optimering: {'Ja' if self.use_optimized else 'Nej'}",
                    f"Använd Kelly-kriteriet: {'Ja' if self.use_kelly else 'Nej'}",
                    f"Använd Risk Management: {'Ja' if self.use_risk_management else 'Nej'}"
                ]
                return header_lines

            # Initialize defaults
            percentage_return = buy_hold_return = learning_percent = mean_reward = 0.0
            current_price = 0.0
            
            # Calculate returns if we have enough data
            if (hasattr(env, 'n_steps') and hasattr(env, 'current_step') and 
                env.n_steps > 1 and env.current_step > 0 and env.current_step < env.n_steps):
                
                # Calculate portfolio return
                percentage_return = ((env.portfolio_value() - env.initial_cash) / env.initial_cash) * 100
                
                # Calculate buy and hold return
                if len(env.prices) > 0 and env.prices[0] != 0:
                    current_idx = min(env.current_step, len(env.prices) - 1)
                    buy_hold_return = ((env.prices[current_idx] - env.prices[0]) / env.prices[0]) * 100
                
                # Calculate learning performance
                learning_percent = (percentage_return / buy_hold_return * 100) if buy_hold_return != 0 else 0.0
                
                # Calculate average reward
                mean_reward = np.mean(reward_history[-10:]) if reward_history and len(reward_history) > 0 else 0
                
                # Get current price
                if len(env.prices) > 0 and env.current_step < env.n_steps:
                    current_price = env.prices[min(env.current_step, len(env.prices) - 1)]

            # Safely access agent attributes
            agent_epsilon = getattr(agent, 'epsilon', 0.0) if agent else 0.0
            agent_gamma = getattr(agent, 'gamma', 0.0) if agent else 0.0
            agent_lr = getattr(agent, 'learning_rate', 0.0) if agent else 0.0
            
            # Organize header lines into logical sections
            header_lines = [
                f"Episod: {episode}  |  Modell: {self.model_type.upper()}  |  Instrument: {self.instruments[self.instrument_index]}",
                f"Avkastning: {percentage_return:.2f}%  |  Buy & Hold: {buy_hold_return:.2f}%  |  Inlärningsprestanda: {learning_percent:.2f}%",
                f"Portföljvärde: {env.portfolio_value():.2f}  |  Kontanter: {env.cash:.2f}  |  Innehav: {env.owned:.3f}",
                f"Pris: {current_price:.2f}  |  Epsilon: {agent_epsilon:.3f}  |  Gamma: {agent_gamma:.2f}  |  LR: {agent_lr:.4f}  |  Avg Belöning: {mean_reward:.2f}"
            ]
            
            # Add Success Pattern information
            if hasattr(env, 'successful_trade_patterns'):
                pattern_count = len(env.successful_trade_patterns)
                pattern_info = f"Sparade mönster: {pattern_count}"
                
                # Add to header only if we have patterns
                if pattern_count > 0:
                    header_lines.append(pattern_info)
            
            # Add Kelly information if available
            if hasattr(env, 'use_kelly') and hasattr(env, 'kelly_sizer'):
                kelly_status = "Aktiv" if env.use_kelly else "Inaktiv"
                kelly_fraction = 0.0
                win_rate = 0.0
                win_loss_ratio = 0.0
                
                if env.use_kelly:
                    kelly_fraction = env.kelly_sizer.calculate_kelly_fraction()
                    win_rate = env.kelly_sizer.win_rate
                    win_loss_ratio = env.kelly_sizer.win_loss_ratio
                
                kelly_line = f"Kelly: {kelly_status}  |  Fraktion: {kelly_fraction:.2f}  |  Win Rate: {win_rate:.2f}  |  W/L Ratio: {win_loss_ratio:.2f}"
                header_lines.append(kelly_line)
            
            # Add Risk Management information if available
            if hasattr(env, 'use_risk_management') and hasattr(env, 'completed_trades'):
                risk_status = "Aktiv" if env.use_risk_management else "Inaktiv"
                
                # Collect trade statistics
                total_trades = len(env.completed_trades)
                stop_loss_exits = sum(1 for t in env.completed_trades if t.get('is_stop_loss_exit', False))
                take_profit_exits = sum(1 for t in env.completed_trades if t.get('is_take_profit_exit', False))
                manual_exits = total_trades - stop_loss_exits - take_profit_exits
                
                # Calculate average profit/loss for different exit types
                sl_profits = [t.get('profit_loss', 0.0) for t in env.completed_trades if t.get('is_stop_loss_exit', False)]
                tp_profits = [t.get('profit_loss', 0.0) for t in env.completed_trades if t.get('is_take_profit_exit', False)]
                
                avg_sl_profit = np.mean(sl_profits) * 100 if sl_profits else 0  # To percentage
                avg_tp_profit = np.mean(tp_profits) * 100 if tp_profits else 0  # To percentage
                
                # Show active positions with stop-loss/take-profit
                position_info = ""
                if hasattr(env, 'position') and env.position is not None and env.position.active_size > 0:
                    sl_level = env.position.stop_loss_level
                    tp_level = env.position.take_profit_level
                    sl_info = f"SL:{sl_level:.2f}" if sl_level is not None else "ingen SL"
                    tp_info = f"TP:{tp_level:.2f}" if tp_level is not None else "ingen TP"
                    position_info = f"  |  Aktiv position: {env.position.active_size:.2f} | {sl_info} | {tp_info}"
                
                # Add information to header
                if total_trades > 0:
                    risk_line = f"Risk Management: {risk_status}  |  Trades: {total_trades}  |  SL: {stop_loss_exits} ({avg_sl_profit:.1f}%)  |  TP: {take_profit_exits} ({avg_tp_profit:.1f}%){position_info}"
                else:
                    risk_line = f"Risk Management: {risk_status}  |  Inga trades än{position_info}"
                
                header_lines.append(risk_line)
            
            return header_lines
            
        except Exception as e:
            logger.error(f"Error updating header: {e}")
            return [
                f"Error: {str(e)}",
                f"Episode: {episode}",
                f"Model: {self.model_type.upper()}"
            ]

    def draw_main_ui(self, 
                   episode: int, 
                   env: Optional[EnvironmentInterface], 
                   agent: Optional[AgentInterface], 
                   reward_history: List[float]) -> None:
        """Draw the main UI with charts and controls"""
        try:
            self.screen.fill(BACKGROUND_COLOR)
            
            # Safe access to environment attributes
            current_step = getattr(env, 'current_step', 0) if env else 0
            trade_events = getattr(env, 'trade_events', []) if env else []
            
            # Draw UI elements
            self.price_panel.draw(self.screen, env, current_step, trade_events)
            
            # Hämta current state om det behövs för pattern panel
            current_state = None
            if self.show_patterns and env and hasattr(env, '_get_state'):
                try:
                    current_state = env._get_state()
                except Exception as e:
                    logger.error(f"Error getting current state: {e}")
            
            # Rita antingen belöningspanelen eller mönsterpanelen baserat på toggle
            if self.show_patterns:
                self.pattern_panel.draw(self.screen, env, current_state)
            else:
                self.reward_panel.draw(self.screen, reward_history)
            
            # Update and draw header
            header_text = self.update_header(episode, env, agent, reward_history)
            self.header.draw(self.screen, header_text)
            
            # Draw all buttons
            self.stop_button.draw(self.screen)
            
            # Update pause button text based on state
            self.pause_button.text = "Resume" if self.paused else "Pause"
            self.pause_button.draw(self.screen)
            
            self.switch_button.draw(self.screen)
            self.zoom_in_button.draw(self.screen)
            self.zoom_out_button.draw(self.screen)
            self.toggle_indicators_button.draw(self.screen)
            self.cycle_indicator_button.draw(self.screen)
            self.save_model_button.draw(self.screen)
            
            # Update Kelly button text based on status
            kelly_status = "På" if (env and hasattr(env, 'use_kelly') and env.use_kelly) else "Av"
            self.toggle_kelly_runtime_button.text = f"Kelly: {kelly_status}"
            self.toggle_kelly_runtime_button.draw(self.screen)
            
            # Update Risk Management button text based on status
            risk_status = "På" if (env and hasattr(env, 'use_risk_management') and env.use_risk_management) else "Av"
            self.toggle_risk_mgmt_runtime_button.text = f"Risk Mgmt: {risk_status}"
            self.toggle_risk_mgmt_runtime_button.draw(self.screen)
            
            # Update show SL/TP button text based on status
            levels_status = "På" if self.price_panel.show_risk_levels else "Av"
            self.toggle_risk_levels_button.text = f"Visa SL/TP: {levels_status}"
            self.toggle_risk_levels_button.draw(self.screen)
            
            # Update patterns button text based on status
            patterns_status = "Dölj" if self.show_patterns else "Visa"
            self.toggle_patterns_button.text = f"{patterns_status} Mönster"
            self.toggle_patterns_button.draw(self.screen)
            
            # Draw indicator info if enabled
            if self.price_panel.show_indicators and self.price_panel.current_indicator:
                indicator_bg = pygame.Rect(
                    self.price_panel_rect[0], 
                    self.price_panel_rect[1] - 30,
                    200, 25
                )
                pygame.draw.rect(self.screen, (40, 40, 40, 180), indicator_bg, border_radius=3)
                
                indicator_text = self.font_small.render(
                    f"Indicator: {self.price_panel.current_indicator}", 
                    True, 
                    (255, 165, 0)
                )
                self.screen.blit(indicator_text, (self.price_panel_rect[0] + 5, self.price_panel_rect[1] - 25))
            
            # Draw status indicator for paused state
            if self.paused:
                pause_text = self.font_header.render("PAUSED", True, (255, 100, 100))
                pause_rect = pause_text.get_rect(center=(self.screen_width // 2, 40))
                self.screen.blit(pause_text, pause_rect)
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            logger.error(f"Error drawing main UI: {e}")
            # Draw fallback UI
            self.screen.fill(BACKGROUND_COLOR)
            error_text = self.font_header.render(f"Error drawing UI: {str(e)}", True, (255, 0, 0))
            self.screen.blit(error_text, (50, 50))
            
            # Still draw critical buttons so user can exit
            try:
                self.stop_button.draw(self.screen)
            except:
                pass
                
            pygame.display.flip()

    def handle_main_ui_events(self) -> str:
        """
        Handle events for the main UI
        
        Returns:
            String indicating the action to take:
            - "continue": Continue normal operation
            - "quit": Exit the application
            - "switch": Switch to a different instrument
            - "save_model": Save the current model
            - "toggle_kelly": Toggle Kelly criterion on/off
            - "toggle_risk_mgmt": Toggle Risk Management on/off
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.stop_button.is_clicked(event):
                    return "quit"
                if self.pause_button.is_clicked(event):
                    self.paused = not self.paused
                    logger.info("Paused!" if self.paused else "Continuing...")
                if self.switch_button.is_clicked(event):
                    self.instrument_index = (self.instrument_index + 1) % len(self.instruments)
                    return "switch"
                if self.zoom_in_button.is_clicked(event):
                    self.price_panel.zoom = min(self.price_panel.zoom * 1.5, 10.0)
                if self.zoom_out_button.is_clicked(event):
                    self.price_panel.zoom = max(self.price_panel.zoom / 1.5, 1.0)
                if self.toggle_indicators_button.is_clicked(event):
                    self.price_panel.show_indicators = not self.price_panel.show_indicators
                    if self.price_panel.show_indicators and not self.price_panel.current_indicator and INDICATORS:
                        self.price_panel.current_indicator = INDICATORS[0]
                if self.cycle_indicator_button.is_clicked(event):
                    if self.price_panel.show_indicators and INDICATORS:
                        self.indicator_index = (self.indicator_index + 1) % len(INDICATORS)
                        self.price_panel.current_indicator = INDICATORS[self.indicator_index]
                if self.save_model_button.is_clicked(event):
                    return "save_model"
                if self.toggle_kelly_runtime_button.is_clicked(event):
                    return "toggle_kelly"
                if self.toggle_risk_mgmt_runtime_button.is_clicked(event):
                    return "toggle_risk_mgmt"
                if self.toggle_risk_levels_button.is_clicked(event):
                    self.price_panel.show_risk_levels = not self.price_panel.show_risk_levels
                    levels_status = "på" if self.price_panel.show_risk_levels else "av"
                    logger.info(f"Visualization of stop-loss/take-profit levels turned {levels_status}")
                if self.toggle_patterns_button.is_clicked(event):
                    self.show_patterns = not self.show_patterns
                    # Justera reward_panel och pattern_panel baserat på visningsstatus
                    if self.show_patterns:
                        self.reward_panel_rect = (
                            10, 
                            self.header_rect[3] + int(self.screen_height * 0.45) + 20,
                            self.screen_width // 2 - 20,
                            int(self.screen_height * 0.30)
                        )
                        self.pattern_panel_rect = (
                            10 + self.screen_width // 2, 
                            self.header_rect[3] + int(self.screen_height * 0.45) + 20,
                            self.screen_width // 2 - 20,
                            int(self.screen_height * 0.30)
                        )
                    else:
                        self.reward_panel_rect = (
                            10, 
                            self.header_rect[3] + int(self.screen_height * 0.45) + 20,
                            self.screen_width - 20,
                            int(self.screen_height * 0.30)
                        )
                    # Uppdatera paneler med nya storlekar
                    self.reward_panel = RewardPanel(self.reward_panel_rect)
                    self.reward_panel.set_font(self.font_small)
                    self.pattern_panel = PatternPanel(self.pattern_panel_rect)
                    self.pattern_panel.set_fonts(self.font_small, self.font_small)
                    logger.info(f"Mönstervisning {'aktiverad' if self.show_patterns else 'inaktiverad'}")
                
                # Hantera mönsterpanelinteraktioner
                if self.show_patterns:
                    self.pattern_panel.handle_events(event, pygame.mouse.get_pos())
        
        return "continue"

    def draw(self, 
            episode: int, 
            env: Optional[EnvironmentInterface], 
            agent: Optional[AgentInterface], 
            reward_history: List[float]) -> None:
        """
        Draw the appropriate UI based on current state
        
        Args:
            episode: Current episode number
            env: Trading environment
            agent: DQN agent
            reward_history: List of rewards from previous episodes
        """
        if self.show_pre_run_menu:
            self.draw_pre_run_menu()
        else:
            self.draw_main_ui(episode, env, agent, reward_history)

    def handle_events(self) -> str:
        """
        Handle UI events based on current state
        
        Returns:
            String indicating the action to take
        """
        if self.show_pre_run_menu:
            return self.handle_pre_run_events()
        else:
            return self.handle_main_ui_events()