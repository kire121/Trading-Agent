import datetime
import numpy as np
import optuna
import pygame
import time
import json
import os

from factory import create_training_session
from config import args
from logger_setup import logger


def objective(trial, ui_manager=None, screen=None, n_trials=10):
    """
    Objektiv funktion för hyperparameteroptimering.
    Tränar agenten under ett antal episoder med de föreslagna hyperparametrarna
    och returnerar medelbelöningen som optimeringsmått.
    
    Parametrar:
        trial: Optuna trial-objekt
        ui_manager: UIManager-instans för UI-uppdateringar
        screen: Pygame screen för rendering
        n_trials: Totalt antal trials som ska köras
    """
    # Definiera hyperparametrar som ska optimeras
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.99)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.90, 0.999)
    batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
    per_alpha = trial.suggest_float('per_alpha', 0.5, 0.7)
    per_beta = trial.suggest_float('per_beta', 0.3, 0.6)
    
    # Lägg till Kelly-relaterade parametrar om Kelly är aktiverat
    use_kelly = ui_manager.use_kelly if ui_manager else getattr(args, 'use_kelly', True)
    if use_kelly:
        max_kelly_fraction = trial.suggest_float('max_kelly_fraction', 0.2, 0.8)
    else:
        max_kelly_fraction = 0.5  # Standardvärde om Kelly är inaktiverat

    # Custom agent parameters for optimization
    agent_params = {
        'gamma': gamma,
        'epsilon': 1.0,  # Start with high exploration
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': epsilon_decay,
        'learning_rate': learning_rate,
        'per_alpha': per_alpha,
        'per_beta': per_beta
    }
    
    # Miljöparametrar med Kelly-inställningar
    env_params = {
        'use_kelly': use_kelly,
        'max_kelly_fraction': max_kelly_fraction,
        'kelly_window': getattr(args, 'kelly_window', 20)
    }
    
    # Create environment and agent using the factory
    instrument = "AAPL"  # Default instrument for optimization
    try:
        env, agent = create_training_session(
            instrument=instrument,
            model_type=args.model_type,
            custom_agent_params=agent_params,
            custom_env_params=env_params,
            use_kelly=use_kelly
        )
    except Exception as e:
        logger.error(f"Failed to create optimization session: {e}")
        return float('-inf')  # Return worst possible value if session creation fails

    total_reward = 0.0
    n_episodes = 10  # Använd ett mindre antal episoder för snabb utvärdering
    reward_history = []
    
    for episode in range(n_episodes):
        state = env.reset().astype(np.float32)
        done = False
        episode_reward = 0.0
        
        # För varje episod, uppdatera UI:n då och då
        update_freq = max(1, env.n_steps // 5)  # Uppdatera UI:n 5 gånger per episod
        step_counter = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state.astype(np.float32), done)
            state = next_state
            episode_reward += reward
            
            # Uppdatera UI:n periodiskt
            step_counter += 1
            if ui_manager and screen and step_counter % update_freq == 0:
                # Hantera händelser för att förhindra att programmet slutar svara
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return float('-inf')  # Avbryt optimeringen om användaren försöker avsluta
                
                # Hämta bästa värdet på ett säkert sätt
                try:
                    best_value = trial.study.best_value
                    best_value_str = f"{best_value:.2f}"
                except ValueError:
                    # Inga försök har avslutats ännu
                    best_value_str = "N/A"
                
                # Uppdatera UI med optimeringsinfo
                trial_info = [
                    f"Optimering - Trial: {trial.number + 1}",
                    f"Episod: {episode + 1}/{n_episodes}",
                    f"Gamma: {gamma:.3f}, LR: {learning_rate:.5f}",
                    f"Kelly: {max_kelly_fraction:.2f}" if use_kelly else "Kelly: Av",
                    f"Reward: {episode_reward:.2f}",
                    f"Bästa hittills: {best_value_str}"
                ]
                
                # Rita UI med temporär information
                screen.fill((30, 30, 30))  # Bakgrundsfärg
                
                # Rita en enkel informationspanel
                font = ui_manager.font_header
                y_pos = 50
                for line in trial_info:
                    text_surf = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surf, (50, y_pos))
                    y_pos += 30
                
                # Rita optimeringsförloppet
                progress = (trial.number * n_episodes + episode) / (n_trials * n_episodes)
                pygame.draw.rect(screen, (70, 70, 70), (50, y_pos + 20, screen.get_width() - 100, 30))
                pygame.draw.rect(screen, (0, 200, 255), (50, y_pos + 20, int((screen.get_width() - 100) * progress), 30))
                
                # Uppdatera UI
                pygame.display.flip()
                pygame.time.delay(10)  # Liten fördröjning för att se till att UI uppdateras
        
        total_reward += episode_reward
        reward_history.append(episode_reward)
        
        # Kör replay om tillräckligt många erfarenheter finns
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    average_reward = total_reward / n_episodes
    
    # Loggning av testresultat inklusive Kelly-information
    kelly_info = ""
    if use_kelly and hasattr(env, 'kelly_sizer'):
        kelly_fraction = env.kelly_sizer.calculate_kelly_fraction()
        win_rate = env.kelly_sizer.win_rate
        win_loss_ratio = env.kelly_sizer.win_loss_ratio
        kelly_info = f", Kelly={max_kelly_fraction:.2f}, WinRate={win_rate:.2f}, W/L={win_loss_ratio:.2f}"
        
    logger.info(f"Trial {trial.number} avslutat: medelbelöning = {average_reward:.2f}{kelly_info}")
    return average_reward


def save_optimization_results(study):
    """Spara optimeringsresultat till en fil"""
    
    os.makedirs('optimization_results', exist_ok=True)
    
    # Konvertera parametrarna till dict och spara
    best_params = study.best_trial.params
    best_value = study.best_trial.value
    
    # Inkludera timestamp för att hålla reda på när optimeringen gjordes
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Notera om Kelly-optimering gjordes
    kelly_optimized = 'max_kelly_fraction' in best_params
    kelly_note = "Kelly-fraktion optimerad." if kelly_optimized else "Kelly-fraktion ej optimerad."
    
    result = {
        "timestamp": timestamp,
        "value": best_value,
        "params": best_params,
        "kelly_optimized": kelly_optimized
    }
    
    filepath = os.path.join('optimization_results', 'latest_optimization.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=4)
    
    # Skapa också en tidsstämplad kopia för historisk referens
    history_filepath = os.path.join('optimization_results', f'optimization_{timestamp}.json')
    with open(history_filepath, 'w') as f:
        json.dump(result, f, indent=4)
    
    logger.info(f"Optimeringsresultat sparade till {filepath}")
    logger.info(f"Historisk kopia sparad till {history_filepath}")
    logger.info(f"{kelly_note}")


def run_optimization(n_trials=50, ui_manager=None, screen=None):
    """
    Kör hyperparameteroptimering med ett specificerat antal försök.
    Returnerar Optuna-studien med de bästa parametrarna.
    
    Parametrar:
        n_trials: Antal försök för optimeringen
        ui_manager: UIManager-instans för UI-uppdateringar
        screen: Pygame screen för rendering
    """
    study = optuna.create_study(direction='maximize')
    
    # Skapa en anpassad objektiv funktion som inkluderar UI-parametrarna
    def objective_with_ui(trial):
        return objective(trial, ui_manager, screen, n_trials)
    
    # Visa en startskärm för optimeringen
    if ui_manager and screen:
        screen.fill((30, 30, 30))
        font = ui_manager.font_header
        
        # Kontrollera om Kelly är aktiverat
        use_kelly = ui_manager.use_kelly if hasattr(ui_manager, 'use_kelly') else getattr(args, 'use_kelly', True)
        kelly_status = "aktiverat" if use_kelly else "inaktiverat"
        
        info_text = [
            "Startar hyperparameteroptimering",
            f"Antal försök: {n_trials}",
            f"Kelly-kriteriet: {kelly_status}",
            "Detta kan ta en stund...",
            "",
            "Optimerar följande parametrar:",
            "- Learning rate",
            "- Gamma (diskonteringsfaktor)",
            "- Epsilon decay",
            "- Batch size",
            "- Prioritized experience replay (alpha, beta)"
        ]
        
        # Lägg till Kelly-fraktion i listan över parametrar som optimeras
        if use_kelly:
            info_text.append("- Kelly-fraktion (max_kelly_fraction)")
        
        y_pos = 100
        for line in info_text:
            text_surf = font.render(line, True, (255, 255, 255))
            screen.blit(text_surf, (50, y_pos))
            y_pos += 30
        
        pygame.display.flip()
        pygame.time.delay(2000)  # Visa startskärmen i 2 sekunder
    
    # Kör optimeringen
    study.optimize(objective_with_ui, n_trials=n_trials)
    
    # Spara resultaten till fil
    save_optimization_results(study)
    
    # Visa resultatet när optimeringen är klar
    if ui_manager and screen:
        screen.fill((30, 30, 30))
        font = ui_manager.font_header
        
        # Kontrollera om Kelly-fraktion optimerades
        kelly_optimized = 'max_kelly_fraction' in study.best_params
        
        result_text = [
            "Optimering klar!",
            f"Bästa belöning: {study.best_value:.2f}",
            "",
            "Bästa parametrar:"
        ]
        
        for key, value in study.best_params.items():
            result_text.append(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
        
        if kelly_optimized:
            kelly_value = study.best_params.get('max_kelly_fraction', 0.5)
            result_text.append(f"- max_kelly_fraction: {kelly_value:.2f} (optimerad)")
        
        result_text.extend([
            "",
            "Fortsätter till träning om 5 sekunder..."
        ])
        
        y_pos = 100
        for line in result_text:
            text_surf = font.render(line, True, (255, 255, 255))
            screen.blit(text_surf, (50, y_pos))
            y_pos += 30
        
        pygame.display.flip()
        
        # Vänta en stund så användaren kan se resultaten innan träningen börjar
        start_time = time.time()
        while time.time() - start_time < 5:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return study
            pygame.time.delay(100)
    
    logger.info("Bästa försök:")
    best_trial = study.best_trial
    logger.info(f"  Värde: {best_trial.value}")
    logger.info("  Parametrar:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")
    
    return study


if __name__ == '__main__':
    # Exekvera optimeringen med önskat antal försök
    study = run_optimization(n_trials=5)