"""Tests for purged walk-forward cross-validation."""

import pytest

from scalp2.config import WalkForwardConfig
from scalp2.training.walk_forward import PurgedWalkForwardCV


@pytest.fixture
def cv():
    config = WalkForwardConfig(
        train_bars=1000,
        val_bars=200,
        test_bars=200,
        purge_bars=10,
        embargo_bars=5,
        step_bars=200,
    )
    return PurgedWalkForwardCV(config)


class TestPurgedWalkForwardCV:
    def test_no_overlap(self, cv):
        """Train/val/test sets must never overlap."""
        assert cv.validate_no_overlap(5000)

    def test_purge_gap_exists(self, cv):
        """Purge gaps must be at least purge_size bars."""
        for fold in cv.split(5000):
            assert fold.val_start - fold.train_end >= cv.purge_size
            assert fold.test_start - fold.val_end >= cv.purge_size

    def test_fold_count(self, cv):
        """Should produce the expected number of folds."""
        n = cv.n_folds(5000)
        actual = sum(1 for _ in cv.split(5000))
        assert n == actual

    def test_insufficient_data(self, cv):
        """Should produce zero folds for tiny datasets."""
        assert cv.n_folds(100) == 0
        assert list(cv.split(100)) == []

    def test_rolling_not_expanding(self, cv):
        """Train window size must be constant across folds."""
        train_sizes = []
        for fold in cv.split(5000):
            train_sizes.append(fold.train_end - fold.train_start)
        assert len(set(train_sizes)) == 1, "Train size varies — should be rolling!"

    def test_step_size_advance(self, cv):
        """Each fold should advance by step_size."""
        folds = list(cv.split(5000))
        if len(folds) > 1:
            for i in range(1, len(folds)):
                offset = folds[i].train_start - folds[i - 1].train_start
                assert offset == cv.step_size

    def test_default_config(self):
        """Test with actual production config values."""
        config = WalkForwardConfig()  # Uses defaults from config
        cv = PurgedWalkForwardCV(config)
        # 210K bars (approx 6 years of 15m data)
        n_folds = cv.n_folds(210000)
        assert n_folds > 40, f"Expected 40+ folds, got {n_folds}"
        assert cv.validate_no_overlap(210000)
