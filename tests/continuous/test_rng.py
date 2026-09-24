"""Reproducible CPU RNG parity against the frozen reference's Torch draws."""
import numpy as np
import pytest

from abcurves._continuous.rng import RandomStream


torch = pytest.importorskip("torch", reason="Reference RNG verification requires optional Torch")


@pytest.mark.parametrize("seed", [-(1 << 63), -7, 0, 7, 23, 109, 701, (1 << 40) + 17, (1 << 64) - 1])
def test_uniform_stream_matches_torch(seed):
    reference = torch.Generator().manual_seed(seed)
    actual = RandomStream(seed)
    # Initial scalar hazard draw followed by pairs differs from bulk-only tests.
    for count in (1, 2, 2, 2, 16, 31, 1024):
        expected = torch.rand(count, generator=reference).numpy()
        np.testing.assert_array_equal(actual.uniform(count), expected)


@pytest.mark.parametrize("seed", [0, 7, 23, (1 << 40) + 17, (1 << 64) - 1])
def test_exponential_stream_matches_torch(seed):
    reference = torch.Generator().manual_seed(seed)
    actual = RandomStream(seed)
    for _ in range(256):
        expected = torch.empty(16).exponential_(generator=reference).numpy()
        np.testing.assert_array_equal(actual.exponential(16), expected)
    np.testing.assert_array_equal(actual.uniform(64), torch.rand(64, generator=reference).numpy())


@pytest.mark.parametrize("seed", [7, 23, 109, 701])
def test_categorical_choices_and_consumption_match_torch(seed):
    reference = torch.Generator().manual_seed(seed)
    actual = RandomStream(seed)
    data_rng = np.random.default_rng(1402)
    for index in range(1024):
        logits = torch.from_numpy(data_rng.normal(size=16).astype(np.float32)) * (0.1 if index % 2 else 5.0)
        probabilities = logits.softmax(0)
        expected = torch.multinomial(probabilities, 1, generator=reference).item()
        assert actual.categorical(probabilities.numpy()) == expected
    np.testing.assert_array_equal(actual.uniform(128), torch.rand(128, generator=reference).numpy())


def test_discarded_proposals_preserve_later_draws():
    complete = RandomStream(23)
    skipped = RandomStream(23)
    weights = np.linspace(0.1, 1.0, 16, dtype=np.float32)
    weights /= weights.sum()
    for index in range(1000):
        head = complete.categorical(weights)
        if index % 11:
            skipped.discard_categorical(16)
        else:
            assert skipped.categorical(weights) == head
    np.testing.assert_array_equal(complete.uniform(1024), skipped.uniform(1024))


@pytest.mark.parametrize("seed", [7, -7, (1 << 40) + 17, (1 << 64) - 1])
def test_mixed_draws_cross_prefetch_boundaries(seed):
    reference = torch.Generator().manual_seed(seed)
    actual = RandomStream(seed)
    weights = torch.linspace(.1, 1., 16)
    weights /= weights.sum()
    for _ in range(12):
        # Odd offsets, exact boundaries, empty draws, and draws larger than the
        # cache exercise both pair parities and preservation of leftover words.
        for count in (1, 509, 2, 0, 1027):
            np.testing.assert_array_equal(actual.uniform(count), torch.rand(count, generator=reference).numpy())
            for size in (16, 255, 0, 257):
                np.testing.assert_array_equal(actual.exponential(size), torch.empty(size).exponential_(generator=reference).numpy())
        torch.multinomial(weights, 1, generator=reference)
        actual.discard_categorical()
        expected = torch.multinomial(weights, 1, generator=reference).item()
        assert actual.categorical(weights.numpy()) == expected
        np.testing.assert_array_equal(actual.uniform((3, 2)), torch.rand((3, 2), generator=reference).numpy())


def test_prefetch_storage_is_bounded_and_public_arrays_are_independent():
    actual = RandomStream(23)
    reference = RandomStream(23)
    first = actual.uniform(17)
    expected_first = reference.uniform(17)
    noise = actual.exponential(16)
    expected_noise = reference.exponential(16)
    for _ in range(40):
        np.testing.assert_array_equal(actual.uniform(13), reference.uniform(13))
        np.testing.assert_array_equal(actual.exponential(16), reference.exponential(16))
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(noise, expected_noise)
    first[:] = -100
    noise[:] = -100
    np.testing.assert_array_equal(actual.uniform(5000), reference.uniform(5000))
    np.testing.assert_array_equal(actual.exponential(2048), reference.exponential(2048))
    assert actual._raw.size == 512
    assert actual._uniforms is None or actual._uniforms.size == 512
    assert actual._exponentials is None or actual._exponentials.size == 511
