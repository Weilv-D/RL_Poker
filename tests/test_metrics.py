"""Tests for evaluation metrics: aggregation and Elo rating system."""

import pytest
from collections import defaultdict
from typing import Dict, List

from rl_poker.scripts.evaluate import (
    EvaluationStats,
    EloTracker,
    calculate_expected_score,
    update_elo_ratings,
    DEFAULT_ELO_K,
    DEFAULT_STARTING_ELO,
)
from rl_poker.engine.game_state import NUM_PLAYERS


class TestMetricAggregation:
    """Tests for metric aggregation with deterministic fake results."""

    def test_evaluation_stats_single_episode(self):
        """Test recording a single episode."""
        stats = EvaluationStats()

        # Player 0 wins (rank 1), player 1 second (rank 2), etc.
        scores = {0: 2, 1: 1, 2: -1, 3: -2}
        ranks = {0: 1, 1: 2, 2: 3, 3: 4}

        stats.record_episode(scores, ranks)

        assert stats.total_episodes == 1
        assert stats.get_average_score(0) == 2.0
        assert stats.get_average_score(1) == 1.0
        assert stats.get_average_score(2) == -1.0
        assert stats.get_average_score(3) == -2.0

    def test_evaluation_stats_multiple_episodes(self):
        """Test aggregation over multiple episodes."""
        stats = EvaluationStats()

        # Episode 1: Player 0 wins
        stats.record_episode(
            scores={0: 2, 1: 1, 2: -1, 3: -2},
            ranks={0: 1, 1: 2, 2: 3, 3: 4},
        )
        # Episode 2: Player 1 wins
        stats.record_episode(
            scores={0: -1, 1: 2, 2: 1, 3: -2},
            ranks={0: 3, 1: 1, 2: 2, 3: 4},
        )
        # Episode 3: Player 3 wins
        stats.record_episode(
            scores={0: -2, 1: -1, 2: 1, 3: 2},
            ranks={0: 4, 1: 3, 2: 2, 3: 1},
        )

        assert stats.total_episodes == 3

        # Average scores
        # Player 0: (2 + -1 + -2) / 3 = -1/3
        assert abs(stats.get_average_score(0) - (-1 / 3)) < 0.01
        # Player 1: (1 + 2 + -1) / 3 = 2/3
        assert abs(stats.get_average_score(1) - (2 / 3)) < 0.01
        # Player 2: (-1 + 1 + 1) / 3 = 1/3
        assert abs(stats.get_average_score(2) - (1 / 3)) < 0.01
        # Player 3: (-2 + -2 + 2) / 3 = -2/3
        assert abs(stats.get_average_score(3) - (-2 / 3)) < 0.01

    def test_win_rate_calculation(self):
        """Test win rate (top 2 finish) calculation."""
        stats = EvaluationStats()

        # 4 episodes where player 0 finishes 1st, 2nd, 3rd, 4th
        stats.record_episode({0: 2, 1: 1, 2: -1, 3: -2}, {0: 1, 1: 2, 2: 3, 3: 4})
        stats.record_episode({0: 1, 1: 2, 2: -1, 3: -2}, {0: 2, 1: 1, 2: 3, 3: 4})
        stats.record_episode({0: -1, 1: 2, 2: 1, 3: -2}, {0: 3, 1: 1, 2: 2, 3: 4})
        stats.record_episode({0: -2, 1: 2, 2: 1, 3: -1}, {0: 4, 1: 1, 2: 2, 3: 3})

        # Player 0: 2 wins (1st, 2nd) out of 4 = 50%
        assert stats.get_win_rate(0) == 0.5
        # Player 1: 4 wins (all top 2: 2nd, 1st, 1st, 1st)
        assert stats.get_win_rate(1) == 1.0
        # Player 2: 3rd, 3rd, 2nd, 2nd = 2 wins out of 4 = 50%
        assert stats.get_win_rate(2) == 0.5

    def test_rank_distribution(self):
        """Test rank distribution calculation."""
        stats = EvaluationStats()

        # 4 episodes with known distributions
        stats.record_episode({0: 2, 1: 1, 2: -1, 3: -2}, {0: 1, 1: 2, 2: 3, 3: 4})
        stats.record_episode({0: 2, 1: 1, 2: -1, 3: -2}, {0: 1, 1: 2, 2: 3, 3: 4})
        stats.record_episode({0: 1, 1: 2, 2: -1, 3: -2}, {0: 2, 1: 1, 2: 3, 3: 4})
        stats.record_episode({0: -1, 1: 2, 2: 1, 3: -2}, {0: 3, 1: 1, 2: 2, 3: 4})

        # Player 0: 1st x2, 2nd x1, 3rd x1
        dist_0 = stats.get_rank_distribution(0)
        assert dist_0[1] == 0.5  # 50% first place
        assert dist_0[2] == 0.25  # 25% second place
        assert dist_0[3] == 0.25  # 25% third place
        assert dist_0[4] == 0.0  # 0% fourth place

        # Player 3: always 4th
        dist_3 = stats.get_rank_distribution(3)
        assert dist_3[4] == 1.0
        assert dist_3[1] == 0.0

    def test_empty_stats(self):
        """Test stats with no episodes recorded."""
        stats = EvaluationStats()

        assert stats.total_episodes == 0
        assert stats.get_average_score(0) == 0.0
        assert stats.get_win_rate(0) == 0.0

        dist = stats.get_rank_distribution(0)
        assert dist == {1: 0, 2: 0, 3: 0, 4: 0}

    def test_all_wins(self):
        """Test edge case: player wins all games."""
        stats = EvaluationStats()

        for _ in range(10):
            stats.record_episode(
                {0: 2, 1: 1, 2: -1, 3: -2},
                {0: 1, 1: 2, 2: 3, 3: 4},
            )

        assert stats.get_win_rate(0) == 1.0
        assert stats.get_average_score(0) == 2.0
        assert stats.get_rank_distribution(0)[1] == 1.0

    def test_zero_wins(self):
        """Test edge case: player never wins (always 4th)."""
        stats = EvaluationStats()

        for _ in range(10):
            stats.record_episode(
                {0: -2, 1: 2, 2: 1, 3: -1},
                {0: 4, 1: 1, 2: 2, 3: 3},
            )

        assert stats.get_win_rate(0) == 0.0
        assert stats.get_average_score(0) == -2.0
        assert stats.get_rank_distribution(0)[4] == 1.0


class TestEloRating:
    """Tests for Elo rating calculations."""

    def test_expected_score_equal_ratings(self):
        """Equal ratings should give 0.5 expected score."""
        expected = calculate_expected_score(1000, 1000)
        assert abs(expected - 0.5) < 0.001

    def test_expected_score_higher_rated_favored(self):
        """Higher rated player should have expected score > 0.5."""
        expected = calculate_expected_score(1200, 1000)
        assert expected > 0.5

        # Significantly higher rating
        expected_strong = calculate_expected_score(1600, 1000)
        assert expected_strong > 0.9

    def test_expected_score_lower_rated_disadvantaged(self):
        """Lower rated player should have expected score < 0.5."""
        expected = calculate_expected_score(1000, 1200)
        assert expected < 0.5

    def test_expected_score_symmetry(self):
        """Expected scores should sum to 1."""
        e_a = calculate_expected_score(1200, 1000)
        e_b = calculate_expected_score(1000, 1200)
        assert abs(e_a + e_b - 1.0) < 0.001

    def test_update_elo_winner_gains(self):
        """Winner should gain rating points."""
        ratings = [1000.0, 1000.0, 1000.0, 1000.0]
        ranks = {0: 1, 1: 2, 2: 3, 3: 4}  # Player 0 wins

        new_ratings = update_elo_ratings(ratings, ranks)

        # Winner should gain
        assert new_ratings[0] > 1000.0
        # Last place should lose
        assert new_ratings[3] < 1000.0

    def test_update_elo_loser_loses(self):
        """Loser should lose rating points."""
        ratings = [1000.0, 1000.0, 1000.0, 1000.0]
        ranks = {0: 4, 1: 3, 2: 2, 3: 1}  # Player 0 loses

        new_ratings = update_elo_ratings(ratings, ranks)

        # Last place loses
        assert new_ratings[0] < 1000.0
        # Winner gains
        assert new_ratings[3] > 1000.0

    def test_update_elo_conserves_total(self):
        """Total Elo should be approximately conserved (zero-sum)."""
        ratings = [1000.0, 1000.0, 1000.0, 1000.0]
        ranks = {0: 1, 1: 2, 2: 3, 3: 4}

        new_ratings = update_elo_ratings(ratings, ranks)

        # Total should be approximately conserved
        # Note: In multi-player Elo, this isn't strictly zero-sum
        # but should be close for equally rated players
        initial_total = sum(ratings)
        final_total = sum(new_ratings)
        assert abs(initial_total - final_total) < 1.0

    def test_update_elo_upset_gives_more_points(self):
        """Lower rated player winning should gain more points."""
        # Underdog wins
        ratings_underdog = [800.0, 1000.0, 1000.0, 1000.0]
        ranks_underdog = {0: 1, 1: 2, 2: 3, 3: 4}
        new_underdog = update_elo_ratings(ratings_underdog, ranks_underdog)

        # Favorite wins
        ratings_favorite = [1200.0, 1000.0, 1000.0, 1000.0]
        ranks_favorite = {0: 1, 1: 2, 2: 3, 3: 4}
        new_favorite = update_elo_ratings(ratings_favorite, ranks_favorite)

        # Underdog should gain more than favorite for same result
        underdog_gain = new_underdog[0] - 800.0
        favorite_gain = new_favorite[0] - 1200.0
        assert underdog_gain > favorite_gain

    def test_update_elo_k_factor_affects_magnitude(self):
        """Higher K factor should produce larger rating changes."""
        ratings = [1000.0, 1000.0, 1000.0, 1000.0]
        ranks = {0: 1, 1: 2, 2: 3, 3: 4}

        new_k16 = update_elo_ratings(ratings, ranks, k_factor=16)
        new_k32 = update_elo_ratings(ratings, ranks, k_factor=32)

        # K=32 should have larger changes than K=16
        change_k16 = abs(new_k16[0] - 1000.0)
        change_k32 = abs(new_k32[0] - 1000.0)
        assert change_k32 > change_k16


class TestEloTracker:
    """Tests for EloTracker class."""

    def test_tracker_initial_ratings(self):
        """Tracker should start with default ratings."""
        tracker = EloTracker()
        for i in range(NUM_PLAYERS):
            assert tracker.get_rating(i) == DEFAULT_STARTING_ELO

    def test_tracker_update(self):
        """Tracker should update ratings after episodes."""
        tracker = EloTracker()

        ranks = {0: 1, 1: 2, 2: 3, 3: 4}
        tracker.update(ranks)

        # Winner should gain
        assert tracker.get_rating(0) > DEFAULT_STARTING_ELO
        # Loser should lose
        assert tracker.get_rating(3) < DEFAULT_STARTING_ELO

    def test_tracker_history(self):
        """Tracker should maintain rating history."""
        tracker = EloTracker()

        # Record 3 episodes
        for _ in range(3):
            tracker.update({0: 1, 1: 2, 2: 3, 3: 4})

        assert len(tracker.rating_history) == 3
        # Each history entry should have 4 ratings
        for entry in tracker.rating_history:
            assert len(entry) == NUM_PLAYERS

    def test_tracker_rating_change(self):
        """Should correctly calculate total rating change."""
        tracker = EloTracker()

        # Player 0 wins every game
        for _ in range(5):
            tracker.update({0: 1, 1: 2, 2: 3, 3: 4})

        change = tracker.get_rating_change(0)
        expected_change = tracker.get_rating(0) - DEFAULT_STARTING_ELO
        assert abs(change - expected_change) < 0.001

    def test_tracker_to_dict_from_dict(self):
        """Tracker should serialize and deserialize correctly."""
        tracker = EloTracker()
        tracker.update({0: 1, 1: 2, 2: 3, 3: 4})
        tracker.update({0: 2, 1: 1, 2: 3, 3: 4})

        # Serialize
        data = tracker.to_dict()

        # Deserialize
        restored = EloTracker.from_dict(data)

        assert restored.ratings == tracker.ratings
        assert restored.rating_history == tracker.rating_history
        assert restored.k_factor == tracker.k_factor

    def test_tracker_custom_k_factor(self):
        """Tracker should use custom K factor."""
        tracker = EloTracker()
        tracker.k_factor = 64

        ranks = {0: 1, 1: 2, 2: 3, 3: 4}
        tracker.update(ranks)

        # Changes should be larger with K=64
        change = abs(tracker.get_rating(0) - DEFAULT_STARTING_ELO)

        # Compare with default K
        tracker2 = EloTracker()
        tracker2.update(ranks)
        change2 = abs(tracker2.get_rating(0) - DEFAULT_STARTING_ELO)

        assert change > change2


class TestEvaluationStatsWithElo:
    """Tests for EvaluationStats with integrated Elo tracking."""

    def test_stats_initializes_elo_tracker(self):
        """EvaluationStats should automatically create EloTracker."""
        stats = EvaluationStats()
        assert stats.elo_tracker is not None
        assert isinstance(stats.elo_tracker, EloTracker)

    def test_stats_updates_elo_on_record(self):
        """Recording an episode should update Elo ratings."""
        stats = EvaluationStats()

        scores = {0: 2, 1: 1, 2: -1, 3: -2}
        ranks = {0: 1, 1: 2, 2: 3, 3: 4}
        stats.record_episode(scores, ranks)

        # Winner should have gained rating
        assert stats.get_elo_rating(0) > DEFAULT_STARTING_ELO
        # Loser should have lost rating
        assert stats.get_elo_rating(3) < DEFAULT_STARTING_ELO

    def test_stats_elo_history_grows(self):
        """Elo history should grow with each episode."""
        stats = EvaluationStats()

        for i in range(5):
            stats.record_episode(
                {0: 2, 1: 1, 2: -1, 3: -2},
                {0: 1, 1: 2, 2: 3, 3: 4},
            )

        assert len(stats.elo_tracker.rating_history) == 5

    def test_stats_summary_includes_elo(self):
        """Summary should include Elo ratings."""
        stats = EvaluationStats()
        stats.record_episode(
            {0: 2, 1: 1, 2: -1, 3: -2},
            {0: 1, 1: 2, 2: 3, 3: 4},
        )

        summary = stats.summary(["Player A", "Player B", "Player C", "Player D"])

        assert "Elo Rating" in summary


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_tied_ranks(self):
        """Test handling of tied ranks (same rank for multiple players)."""
        # Note: In actual game this shouldn't happen, but test the math
        ratings = [1000.0, 1000.0, 1000.0, 1000.0]
        ranks = {0: 1, 1: 1, 2: 3, 3: 4}  # Two players tied for 1st

        new_ratings = update_elo_ratings(ratings, ranks)

        # Both tied players should have same change
        assert abs(new_ratings[0] - new_ratings[1]) < 0.001
        # Tied winners should still gain (they beat players 2 and 3)
        assert new_ratings[0] > 1000.0

    def test_single_episode_elo_change(self):
        """Verify specific Elo changes for known scenario."""
        tracker = EloTracker()
        tracker.k_factor = 32

        # All players at 1000, player 0 wins
        ranks = {0: 1, 1: 2, 2: 3, 3: 4}
        tracker.update(ranks)

        # With equal ratings and K=32:
        # Player 0 expected to beat each opponent with 0.5 probability
        # Actual: beat all 3, so +3 * (1 - 0.5) * 32 / 3 = +16 per opponent scaled
        # The exact value depends on formula, but winner should gain ~16-32 points
        gain = tracker.get_rating(0) - DEFAULT_STARTING_ELO
        assert 10 < gain < 50  # Reasonable range

    def test_many_episodes_rating_divergence(self):
        """Over many episodes, ratings should diverge based on performance."""
        tracker = EloTracker()

        # Player 0 always wins, player 3 always loses
        for _ in range(100):
            tracker.update({0: 1, 1: 2, 2: 3, 3: 4})

        # Strong ordering should emerge
        assert tracker.get_rating(0) > tracker.get_rating(1)
        assert tracker.get_rating(1) > tracker.get_rating(2)
        assert tracker.get_rating(2) > tracker.get_rating(3)

        # Difference should be significant
        assert tracker.get_rating(0) - tracker.get_rating(3) > 200
