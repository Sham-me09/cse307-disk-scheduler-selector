"""Learned scheduler selector: training, shifted-timeline test, calibration."""
import json, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from schedulers import SCHEDULERS, NAMES
from workloads import GEN, features, HEAD0

TRAIN_PER_TYPE = 400
PHASE_LEN = 20                       # windows per phase in the test timeline
PHASES = ["sequential", "random", "bursty"]
N_RUNS = 30                          # independent timelines for the averages


def seeks(reqs):
    return [SCHEDULERS[n](reqs, HEAD0) for n in NAMES]


def make(types, n_each, rng):
    X, Y, S, T = [], [], [], []
    for ty in types:
        for _ in range(n_each):
            reqs, t = GEN[ty](rng)
            s = seeks(reqs)
            X.append(features(reqs, t)); S.append(s)
            Y.append(int(np.argmin(s))); T.append(ty)
    return np.array(X), np.array(Y), np.array(S), T


def timeline(rng):
    return make([p for p in PHASES for _ in range(1)], PHASE_LEN, rng)


def fit(types, seed):
    rng = np.random.default_rng(seed)
    X, Y, _, _ = make(types, TRAIN_PER_TYPE, rng)
    clf = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=seed)
    return clf.fit(X, Y)


def evaluate(train_types, label):
    out = {"fixed": {n: [] for n in NAMES}, "learned": [], "oracle": [],
           "phase": {p: {n: [] for n in NAMES + ["Learned", "Oracle"]} for p in PHASES},
           "acc": [], "conf": [], "correct": [], "pred": [], "true": [], "ptype": []}
    for run in range(N_RUNS):
        clf = fit(train_types, 1000 + run)
        rng = np.random.default_rng(5000 + run)
        X, Y, S, T = timeline(rng)
        proba = clf.predict_proba(X)
        pred = clf.classes_[proba.argmax(1)]
        conf = proba.max(1)
        picked = S[np.arange(len(S)), pred]
        for j, n in enumerate(NAMES):
            out["fixed"][n].append(int(S[:, j].sum()))
        out["learned"].append(int(picked.sum()))
        out["oracle"].append(int(S.min(1).sum()))
        for p in PHASES:
            m = np.array([t == p for t in T])
            for j, n in enumerate(NAMES):
                out["phase"][p][n].append(int(S[m, j].sum()))
            out["phase"][p]["Learned"].append(int(picked[m].sum()))
            out["phase"][p]["Oracle"].append(int(S[m].min(1).sum()))
        out["acc"].append(float((pred == Y).mean()))
        out["conf"] += conf.tolist(); out["correct"] += (pred == Y).tolist()
        out["pred"] += pred.tolist(); out["true"] += Y.tolist(); out["ptype"] += T
    return out


def calibration(conf, correct, nb=5):
    conf, correct = np.array(conf), np.array(correct, dtype=float)
    edges = np.linspace(0.2, 1.0, nb + 1)
    rows, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & ((conf < b) if b < 1.0 else (conf <= b))
        if m.sum() == 0:
            continue
        rows.append(dict(lo=round(a, 2), hi=round(b, 2), n=int(m.sum()),
                         conf=float(conf[m].mean()), acc=float(correct[m].mean())))
        ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return rows, float(ece)


def ms(v):
    return float(np.mean(v)), float(np.std(v))


if __name__ == "__main__":
    res = {}
    for label, types in [("all_types", PHASES), ("no_bursty_in_training", ["sequential", "random"])]:
        o = evaluate(types, label)
        rows, ece = calibration(o["conf"], o["correct"])
        conf, corr = np.array(o["conf"]), np.array(o["correct"])
        res[label] = {
            "total_seek": {**{n: ms(o["fixed"][n]) for n in NAMES},
                           "Learned": ms(o["learned"]), "Oracle": ms(o["oracle"])},
            "phase_seek": {p: {k: ms(v) for k, v in d.items()} for p, d in o["phase"].items()},
            "accuracy": ms(o["acc"]),
            "conf_correct": float(conf[corr].mean()), "conf_wrong": float(conf[~corr].mean()),
            "n_wrong": int((~corr).sum()), "n": int(len(corr)),
            "calibration": rows, "ece": ece,
            "acc_by_phase": {p: float(np.mean([c for c, t in zip(o["correct"], o["ptype"]) if t == p])) for p in PHASES},
            "conf_by_phase": {p: float(np.mean([c for c, t in zip(o["conf"], o["ptype"]) if t == p])) for p in PHASES},
        }
        if label == "all_types":
            keep = o
    json.dump(res, open("results/results.json", "w"), indent=1)

    # ---------- figures (black and white only) ----------
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    # 1. per-phase seek totals
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    keys = NAMES + ["Learned", "Oracle"]
    hatches = ["", "///", "\\\\\\", "xxx", "...", "ooo"]
    fills = ["white", "white", "white", "white", "0.6", "0.25"]
    w = 0.13
    for i, k in enumerate(keys):
        vals = [res["all_types"]["phase_seek"][p][k][0] for p in PHASES]
        errs = [res["all_types"]["phase_seek"][p][k][1] for p in PHASES]
        ax.bar(np.arange(3) + (i - 2.5) * w, vals, w, yerr=errs, color=fills[i],
               edgecolor="black", hatch=hatches[i], label=k, capsize=1.5, linewidth=0.7,
               error_kw={"elinewidth": 0.6})
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{p}\n(windows {i*PHASE_LEN+1}-{(i+1)*PHASE_LEN})" for i, p in enumerate(PHASES)])
    ax.set_ylabel("total seek distance (tracks)"); ax.legend(ncol=3, fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("results/fig_seek_by_phase.pdf"); fig.savefig("results/fig_seek_by_phase.png", dpi=200)

    # 2. reliability diagram
    rows = res["all_types"]["calibration"]
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot([0.2, 1], [0.2, 1], "k--", lw=0.8, label="perfect calibration")
    ax.plot([r["conf"] for r in rows], [r["acc"] for r in rows], "ks-", ms=4, lw=1, label="classifier")
    for r in rows:
        ax.annotate(f"n={r['n']}", (r["conf"], r["acc"]), textcoords="offset points", xytext=(4, -9), fontsize=6)
    ax.set_xlabel("mean confidence"); ax.set_ylabel("fraction correct")
    ax.set_xlim(0.2, 1.02); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("results/fig_calibration.pdf"); fig.savefig("results/fig_calibration.png", dpi=200)

    # 3. confidence histogram right vs wrong
    conf, corr = np.array(keep["conf"]), np.array(keep["correct"])
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    bins = np.linspace(0.2, 1, 17)
    ax.hist(conf[corr], bins, color="0.75", edgecolor="black", linewidth=0.5, label="correct")
    ax.hist(conf[~corr], bins, color="white", edgecolor="black", hatch="///", linewidth=0.5, label="wrong")
    ax.set_xlabel("confidence"); ax.set_ylabel("count"); ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("results/fig_conf_hist.pdf"); fig.savefig("results/fig_conf_hist.png", dpi=200)

    # 4. best-scheduler counts per workload (ground truth)
    cnt = {p: np.zeros(4, int) for p in PHASES}
    for t, y in zip(keep["ptype"], keep["true"]):
        cnt[t][y] += 1
    res["best_counts"] = {p: dict(zip(NAMES, map(int, cnt[p]))) for p in PHASES}
    json.dump(res, open("results/results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
