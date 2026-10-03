"""Step 1: generate the training data.

Each row is one window of 32 disk requests. The columns are the 5 features
and the seek distance of every scheduler on that window; the label ("best")
is the scheduler with the smallest seek distance.
"""
import numpy as np
import pandas as pd
from schedulers import SCHEDULERS, NAMES
from workloads import GEN, features, FEATURE_NAMES, HEAD0


def build(per_type=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ty in GEN:
        for _ in range(per_type):
            reqs, t = GEN[ty](rng)
            seeks = [SCHEDULERS[n](reqs, HEAD0) for n in NAMES]
            row = dict(zip(FEATURE_NAMES, features(reqs, t)))
            row.update({f"seek_{n}": s for n, s in zip(NAMES, seeks)})
            row["best"] = NAMES[int(np.argmin(seeks))]
            row["workload"] = ty
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv("results/dataset.csv", index=False)
    print(df.shape)
    print(df["best"].value_counts())
    print(df.groupby("workload")["best"].value_counts().unstack(fill_value=0))
