"""Small, explicit CPU sampling contract compatible with the audited Torch RNG.

NumPy supplies MT19937. Uniform conversion and the exponential race mirror the
audited PyTorch CPU float32 rand / one-sample multinomial draw consumption.
No Torch import, global random state, or lookup table of reference choices.
"""
from __future__ import annotations

import numpy as np
from operator import index as integer_index


class RandomStream:
    """Reference-compatible draws with a fixed 512-word prefetch window.

    The logical stream is unchanged by prefetching. Uniforms and adjacent-pair
    exponentials are computed lazily for one window; mixed calls and refills
    retain every unused word. Cached draw arrays use at most 6 KiB per stream,
    in addition to NumPy's MT19937 state. Returned arrays never alias caches.
    """
    _BLOCK_WORDS = 512

    def __init__(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("Seed must be an integer")
        if not -(1 << 63) <= int(seed) < (1 << 64):
            raise ValueError("Seed must fit the reference signed/unsigned 64-bit range")
        self.seed = int(seed) % (1 << 64)
        self._rng = np.random.RandomState(self.seed & 0xffffffff)
        self._raw = np.empty(self._BLOCK_WORDS, np.uint32)
        self._at = self._BLOCK_WORDS
        self._uniforms = None
        self._exponentials = None

    def _reserve(self, count):
        """Make a small draw contiguous, carrying any unused prefix forward."""
        if self._at + count <= self._BLOCK_WORDS:
            return
        remaining = self._BLOCK_WORDS - self._at
        if remaining:
            self._raw[:remaining] = self._raw[self._at:]
        self._raw[remaining:] = self._rng.randint(
            0, 1 << 32, size=self._BLOCK_WORDS-remaining, dtype=np.uint32)
        self._at = 0
        self._uniforms = self._exponentials = None

    def _words(self, count):
        """Consume arbitrary raw word counts; large public draws stay bounded."""
        if count <= self._BLOCK_WORDS:
            self._reserve(count)
            start = self._at
            self._at += count
            return self._raw[start:self._at]
        result = np.empty(count, np.uint32)
        written = 0
        while written < count:
            take = min(self._BLOCK_WORDS, count-written)
            result[written:written+take] = self._words(take)
            written += take
        return result

    @staticmethod
    def _exponential_from_words(raw):
        bits = (raw[::2].astype(np.uint64) << np.uint64(32)) | raw[1::2]
        uniform = (bits & np.uint64((1 << 53)-1)).astype(np.float64) * (2. ** -53)
        return (-np.log1p(-uniform)).astype(np.float32)

    def _exponential_view(self, count):
        if count == 0:
            return np.empty(0, np.float32)
        words = count*2
        if words > self._BLOCK_WORDS:
            return self._exponential_from_words(self._words(words))
        self._reserve(words)
        if self._exponentials is None:
            # Both starting parities are needed: brake draws follow one uniform
            # budget word, while motor draws start on the even word boundary.
            bits = (self._raw[:-1].astype(np.uint64) << np.uint64(32)) | self._raw[1:]
            uniform = (bits & np.uint64((1 << 53)-1)).astype(np.float64) * (2. ** -53)
            self._exponentials = (-np.log1p(-uniform)).astype(np.float32)
        start = self._at
        self._at += words
        return self._exponentials[start:self._at:2]

    def uniform(self, size):
        shape = () if size is None else ((integer_index(size),) if np.isscalar(size)
                                        else tuple(integer_index(x) for x in size))
        if any(x < 0 for x in shape):
            raise ValueError("negative dimensions are not allowed")
        count = 1
        for extent in shape:
            count *= extent
        if count == 0:
            return np.empty(shape, np.float32)
        if count > self._BLOCK_WORDS:
            raw = self._words(count)
            return ((raw & np.uint32(0xffffff)).astype(np.float32)
                    * np.float32(2. ** -24)).reshape(shape)
        self._reserve(count)
        if self._uniforms is None:
            self._uniforms = ((self._raw & np.uint32(0xffffff)).astype(np.float32)
                              * np.float32(2. ** -24))
        start = self._at
        self._at += count
        values = self._uniforms[start:self._at].copy().reshape(shape)
        return values[()] if size is None else values

    def exponential(self, size=16):
        count = integer_index(size)
        if count < 0:
            raise ValueError("negative dimensions are not allowed")
        return self._exponential_view(count).copy()

    def categorical(self, probabilities):
        weights = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        exponential = self._exponential_view(weights.size)
        # A zero exponential has the same +inf race score as the CPU reference.
        with np.errstate(divide="ignore", invalid="ignore"):
            return int(np.argmax(weights / exponential))

    def discard_categorical(self, size=16):
        count = 2 * integer_index(size)
        if count < 0:
            raise ValueError("negative dimensions are not allowed")
        while count:
            take = min(self._BLOCK_WORDS, count)
            self._reserve(take)
            self._at += take
            count -= take


def softmax(logits):
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    values = np.exp(x - x.max())
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Invalid categorical probabilities")
    return values / total
