from __future__ import annotations

import numpy as np
import pytest

from abcurves.judges import cnn_c2st, raw_crops


def test_raw_crops_preserve_counts_activity_and_padding() -> None:
    crops = raw_crops(
        [
            np.asarray([[8, -16], [0, 0], [24, -24]], dtype=np.int16),
            np.zeros((0, 2), dtype=np.int16),
        ],
        crop=4,
    )

    assert crops.shape == (2, 3, 4)
    np.testing.assert_array_equal(crops[0, 0], [1.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(crops[0, 1], [-2.0, 0.0, -2.0, 0.0])
    np.testing.assert_array_equal(crops[0, 2], [1.0, 0.0, 1.0, 0.0])
    assert not np.any(crops[1])


def test_raw_crops_fail_closed_on_invalid_streams() -> None:
    with pytest.raises(ValueError, match="positive"):
        raw_crops([], crop=0)
    with pytest.raises(ValueError, match=r"shape \[ticks, 2\]"):
        raw_crops([np.zeros((8, 3), dtype=np.float32)])
    bad = np.zeros((8, 2), dtype=np.float32)
    bad[3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        raw_crops([bad])


def test_raw_cnn_is_deterministic_and_finds_an_obvious_sequence_difference() -> None:
    rows = 18
    real_streams = []
    fake_streams = []
    for index in range(rows):
        real = np.zeros((64, 2), dtype=np.int16)
        real[index % 8 :: 8, 0] = 1
        fake = np.empty((64, 2), dtype=np.int16)
        fake[:, 0] = 8
        fake[:, 1] = np.where(np.arange(64) % 2 == 0, 8, -8)
        real_streams.append(real)
        fake_streams.append(fake)

    real = raw_crops(real_streams, crop=64)
    fake = raw_crops(fake_streams, crop=64)
    groups = np.arange(rows)
    first = cnn_c2st(real, fake, groups=groups, folds=3, epochs=3, seed=19)
    second = cnn_c2st(real, fake, groups=groups, folds=3, epochs=3, seed=19)

    assert first == second
    assert first > 0.95


def test_raw_cnn_validates_tensor_contract() -> None:
    good = np.zeros((4, 3, 32), dtype=np.float32)
    with pytest.raises(ValueError, match=r"matching \[3, ticks\]"):
        cnn_c2st(good, np.zeros((4, 2, 32), dtype=np.float32), epochs=1)
    with pytest.raises(ValueError, match="finite"):
        bad = good.copy()
        bad[0, 0, 0] = np.inf
        cnn_c2st(good, bad, epochs=1)
    with pytest.raises(ValueError, match="epochs"):
        cnn_c2st(good, good, epochs=0)
