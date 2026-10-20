"""Reproducible holdout splits that keep duplicate groups together."""

import numpy as np


def grouped_holdout_indices(groups, holdout_size, seed=42):
    if not 0 < holdout_size < len(groups):
        raise ValueError("holdout_size must leave nonempty training and holdout sets")
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    bundles = list(members.values())
    rng = np.random.default_rng(seed)
    # Each bit marks a reachable sample count using whole groups.
    reachable = 1
    mask = (1 << (holdout_size + 1)) - 1
    predecessors = {}
    for group_index in rng.permutation(len(bundles)):
        size = len(bundles[group_index])
        shifted = (reachable << size) & mask
        added = shifted & ~reachable
        while added:
            bit = added & -added
            total = bit.bit_length() - 1
            predecessors[total] = (total - size, group_index)
            added ^= bit
        reachable |= shifted
        if (reachable >> holdout_size) & 1:
            break
    if not (reachable >> holdout_size) & 1:
        raise ValueError("Requested holdout size cannot keep duplicate groups disjoint")
    chosen, remaining = set(), holdout_size
    while remaining:
        remaining, group_index = predecessors[remaining]
        chosen.update(bundles[group_index])
    order = rng.permutation(len(groups))
    holdout = np.asarray([index for index in order if index in chosen], dtype=np.int64)
    training = np.asarray([index for index in order if index not in chosen], dtype=np.int64)
    return training, holdout
