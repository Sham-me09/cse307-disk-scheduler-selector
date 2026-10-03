"""Synthetic request-window generators and feature extraction."""
import numpy as np

NCYL = 200
WIN = 32          # requests per window
HEAD0 = 100       # head position at the start of every window


def seq_window(rng):
    # long ascending runs: stride of 1-3 tracks, a couple of run restarts
    reqs, t = [], []
    pos = int(rng.integers(0, NCYL))
    clock = 0.0
    for i in range(WIN):
        if rng.random() < 0.06:
            pos = int(rng.integers(0, NCYL))
        pos = (pos + int(rng.integers(1, 4))) % NCYL
        clock += rng.exponential(1.0)
        reqs.append(pos); t.append(clock)
    return reqs, t


def random_window(rng):
    reqs = [int(x) for x in rng.integers(0, NCYL, WIN)]
    t = list(np.cumsum(rng.exponential(1.0, WIN)))
    return reqs, t


def bursty_window(rng):
    # a few hot spots, requests arrive in tight bursts separated by long gaps
    k = int(rng.integers(2, 4))
    centres = rng.integers(10, NCYL - 10, k)
    reqs, t, clock = [], [], 0.0
    while len(reqs) < WIN:
        c = int(rng.choice(centres))
        clock += rng.exponential(8.0)                    # quiet gap
        for _ in range(int(rng.integers(4, 10))):
            if len(reqs) == WIN:
                break
            reqs.append(int(np.clip(rng.normal(c, 6), 0, NCYL - 1)))
            clock += rng.exponential(0.1)
            t.append(clock)
    return reqs, t


GEN = {"sequential": seq_window, "random": random_window, "bursty": bursty_window}


def features(reqs, t, head=HEAD0):
    r = np.array(reqs, dtype=float)
    gaps = np.diff(np.array(t))
    d = np.abs(np.diff(r))
    return [
        r.var(),                                   # variance of track positions
        d.mean(),                                  # mean jump between consecutive requests
        float(np.mean(np.diff(r) > 0)),            # fraction of ascending steps
        gaps.std() / (gaps.mean() + 1e-9),         # burstiness (CV of inter-arrival gaps)
        abs(r.mean() - head),                      # how far the cluster is from the head
    ]

FEATURE_NAMES = ["pos_variance", "mean_jump", "frac_ascending",
                 "arrival_cv", "centre_dist"]
