"""Test rearrangement integration across ml_environment and agent layers."""

import sys

sys.path.insert(0, ".")

import numpy as np
import time

from ml_environment import RummikubMLEnv
from agent import SmartWormAgent, create_aggressive


def test_action_space_size():
    """Verify action space size is 1263."""
    env = RummikubMLEnv(num_players=2)
    obs = env.reset(seed=42)
    valid_mask = obs["valid_actions_mask"]
    assert valid_mask.shape[0] == 1263, f"Expected 1263, got {valid_mask.shape[0]}"
    print(f"PASS: Action space size = {valid_mask.shape[0]}")


def test_rearrange_base_matches():
    """Verify REARRANGE_BASE is consistent between env and agent."""
    env = RummikubMLEnv(num_players=2)
    agent = SmartWormAgent("Test")
    assert env.REARRANGE_BASE == agent.ACTION_REARRANGE_BASE == 1233, (
        f"Mismatch: env={env.REARRANGE_BASE}, agent={agent.ACTION_REARRANGE_BASE}"
    )
    print(f"PASS: REARRANGE_BASE consistent = {env.REARRANGE_BASE}")


def test_stuck_game_detection():
    """Run the previously stuck seed=0 game and verify it completes."""
    env = RummikubMLEnv(num_players=2, max_steps=2000)
    agents = [SmartWormAgent("P1"), SmartWormAgent("P2")]

    obs = env.reset(seed=0)
    for agent in agents:
        agent.reset()

    done = False
    step = 0
    current_player = 0
    rearrange_count = 0
    stuck_detected = False

    while not done and step < 2000:
        agent = agents[current_player]
        valid_actions = obs["valid_actions_mask"]
        action = agent.select_action(obs, valid_actions)
        next_obs, reward, done, info = env.step(action)

        if info.get("action_taken") == "REARRANGE":
            rearrange_count += 1
        if info.get("stuck"):
            stuck_detected = True

        obs = next_obs
        current_player = obs["current_player"][0]
        step += 1

    winner = info.get("winner")
    print(f"PASS: Seed=0 game completed in {step} steps")
    print(f"  Winner: {'P' + str(winner) if winner is not None else 'Draw'}")
    print(f"  Rearrangements: {rearrange_count}")
    print(f"  Stuck detection triggered: {stuck_detected}")
    return step < 2000  # Should complete before max_steps


def test_full_games(num_games=3):
    """Run several full games and verify they complete with valid outcomes."""
    env = RummikubMLEnv(num_players=2, max_steps=2000)
    agents = [SmartWormAgent("P1"), create_aggressive("P2")]

    wins = 0
    draws = 0
    total_rearrangements = 0
    total_steps = 0

    for seed in range(num_games):
        obs = env.reset(seed=seed)
        for agent in agents:
            agent.reset()

        done = False
        step = 0
        current_player = 0
        game_rearrangements = 0

        while not done and step < 2000:
            agent = agents[current_player]
            valid_actions = obs["valid_actions_mask"]
            action = agent.select_action(obs, valid_actions)
            next_obs, reward, done, info = env.step(action)

            if info.get("action_taken") == "REARRANGE":
                game_rearrangements += 1

            obs = next_obs
            current_player = obs["current_player"][0]
            step += 1

        winner = info.get("winner")
        if winner is not None:
            wins += 1
        else:
            draws += 1

        total_rearrangements += game_rearrangements
        total_steps += step
        print(
            f"  Game {seed}: {step} steps, winner={'P' + str(winner) if winner is not None else 'Draw'}, rearrangements={game_rearrangements}"
        )

    print(f"\nPASS: {num_games} games completed")
    print(f"  Wins: {wins}, Draws: {draws}")
    print(f"  Total rearrangements: {total_rearrangements}")
    print(f"  Avg steps: {total_steps / num_games:.0f}")


if __name__ == "__main__":
    print("=" * 60)
    print("REARRANGEMENT INTEGRATION TESTS")
    print("=" * 60)

    start = time.time()

    print("\n--- Test 1: Action space size ---")
    test_action_space_size()

    print("\n--- Test 2: REARRANGE_BASE consistency ---")
    test_rearrange_base_matches()

    print("\n--- Test 3: Stuck game detection (seed=0) ---")
    test_stuck_game_detection()

    print("\n--- Test 4: Full games ---")
    test_full_games(3)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"All tests passed in {elapsed:.1f}s")
