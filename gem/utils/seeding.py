"""Seeding the random number generator."""

import numpy as np


def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise Exception(f"Seed must be a python integer, actual type: {type(seed)}")
        else:
            raise Exception(
                f"Seed must be greater or equal to zero, actual value: {seed}"
            )

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
    return rng, np_seed


RNG = RandomNumberGenerator = np.random.Generator
