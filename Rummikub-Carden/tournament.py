"""Tournament system for training and evaluating Rummikub agents.

Allows multiple agents to play against each other and tracks statistics.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

from ml_environment import RummikubMLEnv
from agent import RummikubAgent


class ProgressTracker:
    """Tracks and displays tournament progress with statistics."""
    
    def __init__(self, total_games: int, update_interval: float = 1.0):
        """Initialize progress tracker.
        
        Args:
            total_games: Total number of games to run
            update_interval: How often to update display (seconds)
        """
        self.total_games = total_games
        self.completed_games = 0
        self.start_time = None
        self.update_interval = update_interval
        self.last_update = 0
        self.agent_stats = defaultdict(lambda: {'wins': 0, 'games': 0})
        self.current_matchup = ""
        self.win_streaks = defaultdict(int)
        self.last_winner = None
        
    def start(self):
        """Start tracking progress."""
        self.start_time = time.time()
        self.last_update = self.start_time
        self._print_header()
        
    def _print_header(self):
        """Print progress header."""
        print("\n" + "="*80)
        print(f"{'TOURNAMENT PROGRESS':^80}")
        print("="*80)
        print(f"Total Games: {self.total_games}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80)
        
    def update(self, game_result: Dict):
        """Update progress with a completed game.
        
        Args:
            game_result: Result from run_game()
        """
        self.completed_games += 1
        
        # Update agent stats
        agents = game_result['agents']
        winner = game_result['winner_name']
        
        for agent in agents:
            self.agent_stats[agent]['games'] += 1
        
        if winner != 'None':
            self.agent_stats[winner]['wins'] += 1
            
            # Track win streaks
            if winner == self.last_winner:
                self.win_streaks[winner] += 1
            else:
                if self.last_winner:
                    self.win_streaks[self.last_winner] = 0
                self.win_streaks[winner] = 1
                self.last_winner = winner
        
        # Update current matchup info
        self.current_matchup = " vs ".join(agents)
        
        # Check if we should display update
        # Update on: first game, interval elapsed, or last game
        current_time = time.time()
        if self.completed_games == 1 or \
           current_time - self.last_update >= self.update_interval or \
           self.completed_games == self.total_games:
            self._display_progress()
            self.last_update = current_time
            
    def _display_progress(self):
        """Display current progress."""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        progress_pct = (self.completed_games / self.total_games) * 100
        
        # Calculate ETA
        if self.completed_games > 0 and elapsed > 0:
            games_per_sec = self.completed_games / elapsed
            remaining_games = self.total_games - self.completed_games
            eta_seconds = remaining_games / games_per_sec if games_per_sec > 0 else 0
            eta_str = self._format_time(eta_seconds)
        else:
            games_per_sec = 0
            eta_str = "--:--:--"
        
        # Build progress bar (ASCII only for Windows compatibility)
        bar_width = 40
        filled = int(bar_width * self.completed_games / self.total_games)
        bar = "#" * filled + "-" * (bar_width - filled)
        
        # Clear screen and print progress
        sys.stdout.write("\033[2J\033[H")  # Clear screen and move to top
        sys.stdout.flush()
        
        print("\n" + "="*80)
        print(f"{'TOURNAMENT PROGRESS':^80}")
        print("="*80)
        print(f"Progress: [{bar}] {progress_pct:.1f}%")
        print(f"Games: {self.completed_games}/{self.total_games} | "
              f"Rate: {games_per_sec:.1f} games/sec | "
              f"ETA: {eta_str}")
        print(f"Elapsed: {self._format_time(elapsed)}")
        print("-"*80)
        
        # Current matchup
        if self.current_matchup:
            print(f"Current: {self.current_matchup}")
            print()
        
        # Agent standings
        print("Current Standings:")
        print(f"{'Rank':<6} {'Agent':<20} {'Wins':<8} {'Games':<8} {'Win %':<10} {'Streak':<8}")
        print("-"*80)
        
        # Sort by win rate
        standings = []
        for agent, stats in self.agent_stats.items():
            win_rate = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
            standings.append((agent, stats['wins'], stats['games'], win_rate, 
                            self.win_streaks[agent]))
        
        standings.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (agent, wins, games, win_rate, streak) in enumerate(standings, 1):
            streak_str = f"{streak}W" if streak > 1 else "-"
            print(f"{rank:<6} {agent:<20} {wins:<8} {games:<8} {win_rate:>6.1f}%   {streak_str:<8}")
        
        print()
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
    def finish(self):
        """Display final summary."""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*80)
        print(f"{'TOURNAMENT COMPLETE':^80}")
        print("="*80)
        print(f"Total Games: {self.completed_games}/{self.total_games}")
        print(f"Total Time: {self._format_time(elapsed)}")
        if elapsed > 0:
            print(f"Average: {self.completed_games/elapsed:.2f} games/sec")
        print()
        

class Tournament:
    """Manages games between multiple agents."""
    
    def __init__(self, agents: List[RummikubAgent], num_players: int = 2):
        """Initialize tournament.
        
        Args:
            agents: List of agents to compete
            num_players: Number of players per game
        """
        self.agents = agents
        self.num_players = num_players
        self.env = RummikubMLEnv(num_players=num_players)
        
        # Statistics
        self.game_results = []
        self.agent_stats = {agent.name: {
            'wins': 0,
            'games': 0,
            'total_reward': 0.0,
            'avg_game_length': 0,
        } for agent in agents}
    
    def run_game(self, agent_indices: List[int], seed: Optional[int] = None,
                 verbose: bool = False) -> Dict:
        """Run a single game with specified agents.
        
        Args:
            agent_indices: Indices of agents to use (length must match num_players)
            seed: Random seed
            verbose: Print game details
            
        Returns:
            Game result dict
        """
        print(f"[DEBUG] Starting run_game with agents: {agent_indices}, seed={seed}")
        
        if len(agent_indices) != self.num_players:
            raise ValueError(f"Need {self.num_players} agents, got {len(agent_indices)}")
        
        game_agents = [self.agents[i] for i in agent_indices]
        print(f"[DEBUG] Game agents: {[a.name for a in game_agents]}")
        
        # Reset environment and agents
        print("[DEBUG] Calling env.reset()...")
        obs = self.env.reset(seed=seed)
        print("[DEBUG] env.reset() completed")
        for agent in game_agents:
            agent.reset()
        
        done = False
        step = 0
        current_player = 0
        game_history = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting game with: {[a.name for a in game_agents]}")
            print(f"{'='*60}")
        
        info = {}
        last_progress_step = 0
        import time
        step_start_time = time.time()
        
        while not done and step < self.env.max_steps:
            agent = game_agents[current_player]
            valid_actions = obs['valid_actions_mask']
            
            # Debug: Check if we're stuck (5 seconds per step)
            elapsed = time.time() - step_start_time
            if elapsed > 5.0:
                print(f"\n[ERROR] Step timeout at step {step}! Breaking.")
                break
            
            # Agent selects action
            action = agent.select_action(obs, valid_actions)
            
            # Execute action
            next_obs, reward, done, info = self.env.step(action)
            
            step_start_time = time.time()  # Reset timer after successful step
            
            # Update Q-learning agents if applicable
            if hasattr(agent, 'update'):
                agent.update(reward, next_obs)
            
            game_history.append({
                'step': step,
                'player': current_player,
                'agent': agent.name,
                'action': action,
                'action_type': info.get('action_taken', 'UNKNOWN'),
                'reward': reward,
            })
            
            if verbose and info.get('action_success'):
                print(f"Step {step}: {agent.name} took {info['action_taken']} "
                      f"(reward: {reward:.2f})")
            
            obs = next_obs
            current_player = obs['current_player'][0]
            step += 1
            
            if done:
                break
        
        # Check if game hit turn limit (timeout)
        if step >= self.env.max_steps:
            print(f"\n[DEBUG] GAME TIMED OUT at turn {step}!")
            print(f"[DEBUG] Printing game state for debugging...")
            
            # Print table melds
            print("\n=== TABLE MELDS ===")
            table_melds = self.env.game_state.table_melds
            if table_melds:
                for i, meld in enumerate(table_melds):
                    tiles_str = ", ".join([str(t) for t in meld.tiles])
                    print(f"  Meld {i}: {tiles_str}")
            else:
                print("  (empty)")
            
            # Print board matrix format
            print("\n=== BOARD MATRIX ===")
            color_names = ['Red', 'Blue', 'Black', 'Orange']
            board_matrix = [[0]*13 for _ in range(4)]
            for meld in table_melds:
                for tile in meld.tiles:
                    if not tile.is_joker and tile.color and tile.number:
                        row = {'red': 0, 'blue': 1, 'black': 2, 'orange': 3}[tile.color]
                        board_matrix[row][tile.number-1] += 1
            
            for row_idx, row in enumerate(board_matrix):
                print(f"  {color_names[row_idx]}: {row}")
            
            # Print player hands
            for pid in range(self.num_players):
                hand = self.env.game_state.player_hands[pid]
                hand_tiles = [str(t) for t in hand]
                print(f"\n=== PLAYER {pid+1} HAND ({len(hand)} tiles) ===")
                print(f"  {hand_tiles}")
                
                # Hand matrix
                hand_matrix = [[0]*13 for _ in range(4)]
                for tile in hand:
                    if not tile.is_joker and tile.color and tile.number:
                        row = {'red': 0, 'blue': 1, 'black': 2, 'orange': 3}[tile.color]
                        hand_matrix[row][tile.number-1] += 1
                
                for row_idx, row in enumerate(hand_matrix):
                    if sum(row) > 0:
                        print(f"  {color_names[row_idx]}: {row}")
            
            print(f"\n=== GAME INFO ===")
            print(f"  Turn count: {self.env.game_state.turn_count}")
            print(f"  Pool remaining: {self.env.game_state.tile_pool.remaining()}")
            for pid in range(self.num_players):
                print(f"  Player {pid+1} initial meld: {self.env.game_state.has_initial_meld[pid]}")
        
        # Determine winner
        winner_idx = info.get('winner')
        if winner_idx is None:
            winner_idx = -1
        
        result = {
            'agents': [a.name for a in game_agents],
            'winner': winner_idx,
            'winner_name': game_agents[winner_idx].name if winner_idx >= 0 else 'None',
            'steps': step,
            'seed': seed,
            'history': game_history if verbose else None,
        }
        
        # Update stats
        for i, agent in enumerate(game_agents):
            self.agent_stats[agent.name]['games'] += 1
            if i == winner_idx:
                self.agent_stats[agent.name]['wins'] += 1

        # Notify agents of game result (enables adaptive agents)
        for i, agent in enumerate(game_agents):
            agent.update_stats(won=(i == winner_idx), reward=0.0, moves=step)
        
        self.game_results.append(result)
        
        if verbose:
            print(f"\nGame ended after {step} steps")
            print(f"Winner: {result['winner_name']}")
        
        return result
    
    def run_round_robin(self, games_per_matchup: int = 10,
                        verbose: bool = False,
                        show_progress: bool = True,
                        progress_interval: float = 2.0) -> Dict:
        """Run round-robin tournament where each agent plays each other.
        
        Args:
            games_per_matchup: Number of games per agent pair
            verbose: Print progress text
            show_progress: Show visual progress tracker
            progress_interval: How often to update progress display (seconds)
            
        Returns:
            Tournament results
        """
        if self.num_players != 2:
            raise ValueError("Round-robin only supported for 2 players")
        
        num_agents = len(self.agents)
        total_games = num_agents * (num_agents - 1) * games_per_matchup
        
        print(f"\nRunning Round-Robin Tournament")
        print(f"Agents: {[a.name for a in self.agents]}")
        print(f"Games per matchup: {games_per_matchup}")
        print(f"Total games: {total_games}")
        print()
        
        # Initialize progress tracker
        tracker = None
        if show_progress:
            tracker = ProgressTracker(total_games, update_interval=progress_interval)
            tracker.start()
        
        games_completed = 0
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                
                for game_num in range(games_per_matchup):
                    seed = random.randint(0, 1000000)
                    result = self.run_game([i, j], seed=seed, verbose=False)
                    games_completed += 1
                    
                    if tracker:
                        tracker.update(result)
                    
                    if verbose and not show_progress and game_num == 0:
                        print(f"{self.agents[i].name} vs {self.agents[j].name}: "
                              f"Winner = {result['winner_name']}")
        
        if tracker:
            tracker.finish()
        
        return self.get_results()
    
    def run_random_matchups(self, num_games: int = 100,
                            verbose: bool = False,
                            show_progress: bool = True,
                            progress_interval: float = 2.0) -> Dict:
        """Run random matchups between agents.
        
        Args:
            num_games: Total number of games to run
            verbose: Print progress text
            show_progress: Show visual progress tracker
            progress_interval: How often to update progress display (seconds)
            
        Returns:
            Tournament results
        """
        print(f"\nRunning Random Matchups")
        print(f"Total games: {num_games}")
        print()
        
        # Initialize progress tracker
        tracker = None
        if show_progress:
            tracker = ProgressTracker(num_games, update_interval=progress_interval)
            tracker.start()
        
        for game_num in range(num_games):
            # Randomly select agents
            agent_indices = random.sample(range(len(self.agents)), self.num_players)
            seed = random.randint(0, 1000000)
            
            result = self.run_game(agent_indices, seed=seed, verbose=False)
            
            if tracker:
                tracker.update(result)
            
            if verbose and not show_progress and game_num % 10 == 0:
                print(f"Game {game_num + 1}/{num_games}: "
                      f"Winner = {result['winner_name']}")
        
        if tracker:
            tracker.finish()
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """Get tournament results and statistics."""
        results = {
            'total_games': len(self.game_results),
            'agent_stats': {},
            'win_matrix': defaultdict(lambda: defaultdict(int)),
        }
        
        for agent_name, stats in self.agent_stats.items():
            games = stats['games']
            wins = stats['wins']
            win_rate = (wins / games * 100) if games > 0 else 0
            
            results['agent_stats'][agent_name] = {
                'games': games,
                'wins': wins,
                'losses': games - wins,
                'win_rate': win_rate,
            }
        
        # Build win matrix
        for game in self.game_results:
            if game['winner'] >= 0:
                winner = game['winner_name']
                loser = game['agents'][1 - game['winner']]
                results['win_matrix'][winner][loser] += 1
        
        return results
    
    def print_results(self):
        """Print tournament results."""
        results = self.get_results()
        
        print("\n" + "="*60)
        print("TOURNAMENT RESULTS")
        print("="*60)
        
        print(f"\nTotal Games: {results['total_games']}")
        print()
        
        # Agent statistics
        print("Agent Performance:")
        print(f"{'Agent':<20} {'Games':<8} {'Wins':<8} {'Losses':<8} {'Win Rate':<10}")
        print("-" * 60)
        
        sorted_agents = sorted(
            results['agent_stats'].items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        for agent_name, stats in sorted_agents:
            print(f"{agent_name:<20} {stats['games']:<8} {stats['wins']:<8} "
                  f"{stats['losses']:<8} {stats['win_rate']:>6.1f}%")
        
        print()
    
    def save_results(self, filename: Optional[str] = None):
        """Save tournament results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tournament_results_{timestamp}.json"
        
        results = self.get_results()
        results['game_results'] = self.game_results
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    from agent import (
        SmartWormAgent, AdaptiveWormAgent,
        create_hoarder, create_aggressive, create_strategic,
        create_greedy, create_cautious, create_adaptive,
    )

    # Create diverse agents with different strategies
    agents = [
        SmartWormAgent("SmartWorm"),
        create_hoarder(),
        create_aggressive(),
        create_strategic(),
        create_greedy(),
        create_cautious(),
        create_adaptive(),
    ]

    print("Rummikub ML Tournament System")
    print("=" * 60)
    print("\nAgent Strategies:")
    print("  - SmartWorm:   Baseline solver (no weights)")
    print("  - Hoarder:     Holds tiles, draws a lot, plays big melds")
    print("  - Aggressive:  Plays everything ASAP, small melds first")
    print("  - Strategic:   Balanced, prefers medium-large melds")
    print("  - Greedy:      Always plays biggest meld available")
    print("  - Cautious:    Conservative, small melds, moderate hoarding")
    print("  - Adaptive:    Learns weight adjustments from wins/losses")

    print("\n" + "=" * 60)
    print("Running Tournament: All Strategies")
    print("=" * 60)
    tournament = Tournament(agents, num_players=2)
    tournament.run_random_matchups(num_games=50, verbose=False, show_progress=True, progress_interval=2.0)
    tournament.print_results()
    
    # Print adaptive agent weight evolution
    for agent in agents:
        if isinstance(agent, AdaptiveWormAgent):
            print()
            print(agent.get_weight_summary())
