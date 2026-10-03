"""Step 2: train the classifier on the dataset and compare a few simple models.

Reads results/dataset.csv, trains on 70% of the windows, tests on the other 30%,
and saves the chosen model (random forest) to results/model.joblib.
The comparison is repeated on 10 fresh datasets to see how stable it is.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from make_dataset import build
from schedulers import NAMES
from workloads import FEATURE_NAMES

MODELS = {
    "Always SSTF (no model)": None,
    "Decision tree (depth 4)": lambda s: DecisionTreeClassifier(max_depth=4, random_state=s),
    "Logistic regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random forest (60 trees)": lambda s: RandomForestClassifier(n_estimators=60, max_depth=5, random_state=s),
}


def score(df_test, pred):
    seek = df_test[[f"seek_{n}" for n in NAMES]].to_numpy()
    chosen = seek[np.arange(len(seek)), [NAMES.index(p) for p in pred]]
    oracle = seek.min(1)
    return float((np.array(pred) == df_test["best"].to_numpy()).mean()), float((chosen - oracle).mean())


if __name__ == "__main__":
    out = {m: {"acc": [], "regret": []} for m in MODELS}
    for rep in range(10):
        df = build(400, seed=rep)
        tr, te = train_test_split(df, test_size=0.3, random_state=rep, stratify=df["workload"])
        for name, mk in MODELS.items():
            if mk is None:
                pred = ["SSTF"] * len(te)
            else:
                m = mk(rep).fit(tr[FEATURE_NAMES], tr["best"])
                pred = m.predict(te[FEATURE_NAMES])
            a, r = score(te, pred)
            out[name]["acc"].append(a); out[name]["regret"].append(r)
    summary = {k: dict(acc=[float(np.mean(v["acc"])), float(np.std(v["acc"]))],
                       regret=[float(np.mean(v["regret"])), float(np.std(v["regret"]))]) for k, v in out.items()}
    print(f"{'model':28s} {'accuracy':>16s} {'extra seek/window':>20s}")
    for k, v in summary.items():
        print(f"{k:28s} {v['acc'][0]:.3f} +- {v['acc'][1]:.3f}   {v['regret'][0]:6.1f} +- {v['regret'][1]:.1f}")
    json.dump(summary, open("results/model_comparison.json", "w"), indent=1)

    # the model that is used in the main experiment: train once on dataset.csv and save it
    df = pd.read_csv("results/dataset.csv")
    final = MODELS["Random forest (60 trees)"](0).fit(df[FEATURE_NAMES], df["best"])
    joblib.dump(final, "results/model.joblib")
    imp = dict(zip(FEATURE_NAMES, map(float, final.feature_importances_)))
    json.dump(imp, open("results/feature_importance.json", "w"), indent=1)
    print("feature importance:", {k: round(v, 3) for k, v in imp.items()})
    print("saved results/model.joblib")
