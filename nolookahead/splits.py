"""Purged, embargoed walk-forward splits for model selection inside a training window.

Use these instead of ``KFold``/``train_test_split`` (which shuffle time) when
tuning hyperparameters. Each split trains strictly on the past; training rows
whose label window overlaps the test block are purged, and an embargo gap
follows the purge to absorb serial correlation.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int,
    horizon: int,
    embargo: int = 0,
    min_train: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` over positions ``0..n_samples-1``.

    The label for row ``i`` uses data up to ``i + horizon``. A training row is
    kept only if ``i + horizon < test_start - embargo``.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    test_size = n_samples // (n_splits + 1)
    if test_size == 0:
        raise ValueError("not enough samples for that many splits")
    for k in range(1, n_splits + 1):
        test_start = k * test_size
        test_end = n_samples if k == n_splits else test_start + test_size
        last_train = test_start - embargo - horizon  # exclusive
        train = np.arange(0, max(last_train, 0))
        if len(train) < min_train:
            continue
        yield train, np.arange(test_start, test_end)
