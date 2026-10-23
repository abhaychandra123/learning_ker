"""Section 5.1 Bernoulli-Gaussian sparse codes and independent RNG streams."""
import math

import numpy as np

from alista_weights import validate_dictionary


def make_rng(seed, split):
    if not isinstance(seed, int) or seed < 0 or split not in ("train", "validation", "test"):
        raise ValueError("seed must be nonnegative and split train/validation/test")
    # Independent streams even when the same user seed is used in every CLI.
    return np.random.default_rng(np.random.SeedSequence(
        [seed, 0x414C4953, {"train": 0, "validation": 1, "test": 2}[split]]))


class SparseProblem:
    def __init__(self, dictionary, *, probability=.1, snr_db=math.inf):
        self.D = validate_dictionary(dictionary)
        if not math.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("probability must be in (0, 1]")
        if math.isnan(snr_db) or snr_db == -math.inf:
            raise ValueError("SNR must be finite or positive infinity")
        self.probability, self.snr_db = probability, snr_db

    def sample(self, count, rng):
        if not isinstance(count, int) or count < 1:
            raise ValueError("sample count must be a positive integer")
        shape = (count, self.D.shape[1])
        support = rng.random(shape) < self.probability
        x = support * rng.standard_normal(shape)
        b = x @ self.D.T
        if self.snr_db != math.inf:
            # Match authors' utils/train.py: per-example measurement std,
            # not RMS, and Gaussian noise with this expected standard deviation.
            sigma = b.std(axis=1, keepdims=True) * 10.**(-self.snr_db/20.)
            b = b + sigma*rng.standard_normal(b.shape)
        if not np.isfinite(b).all():
            raise FloatingPointError("nonfinite measurements; check SNR")
        return b, x

    def configuration(self):
        return dict(probability=self.probability,
                    snr_db=None if self.snr_db == math.inf else self.snr_db,
                    noise_scaling="per-example population standard deviation of clean b")
