"""Pool level: a learned attacker on text embeddings of "question + options" (no video).

prequential (default, the protocol of the study)
    The items of a benchmark are visited in a random order; item i is predicted by a model trained on the i - 1 items
    before it, then its gold letter is revealed and the model takes one update step. 50 orders (seeds 1..50).
    Learner: SGD logistic regression (L2 1e-4, constant learning rate 0.03, averaged), classes = option letters,
    features = the embedding standardized with the running mean and variance of the seen items, divided by sqrt(dim),
    plus a one-hot code of the option count; the prediction is the most probable of the item's k letters.
    eps_last20 = mean accuracy over the last 20 % of the positions (accuracy smoothed over 20 positions) - chance.
five_fold
    Logistic regression on the embeddings, five-fold held-out prediction within the benchmark (stratified by gold
    letter); eps_last20 holds the held-out accuracy - chance over all items.
Output  tables/pool_attack.csv: benchmark, attacker, mode, n_items, n_mcq, chance_export, eps_last20, acc_last20, ci_lo,
        ci_hi, n20, n_perms, embedding_model. The levels stage re-bases the margin on its own chance value.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

from ..runner import log
from ..scoring import mcq_letter
from .items import item_text, load_samples, select
from .sample import removed_items

N_LETTERS = 26
ALPHA, ETA0, AVERAGE, UNIT_SCALE = 1e-4, 3e-2, True, True
SMOOTH, STEPS_MARGIN = 20, 0.05
COLS = ["benchmark", "attacker", "mode", "n_items", "n_mcq", "chance_export", "eps_last20", "acc_last20", "ci_lo", "ci_hi",
        "n20", "n_perms", "embedding_model"]


class Learner:
    def __init__(self, dim, seed, alpha=ALPHA, eta0=ETA0, average=AVERAGE, learning_rate="constant", unit_scale=UNIT_SCALE):
        from sklearn.linear_model import SGDClassifier
        self.dim, self.unit_scale, self.n = dim, unit_scale, 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)          # Welford: sum of squared deviations
        self.rng = np.random.default_rng(seed)
        self.classes = np.arange(N_LETTERS)
        self.clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, learning_rate=learning_rate, eta0=eta0,
                                 power_t=0.5, average=average, fit_intercept=True, class_weight=None, shuffle=False,
                                 random_state=int(seed), tol=None, max_iter=1, n_jobs=1, warm_start=False)
        self.fitted = False

    def _feat(self, emb, k):
        var = self.m2 / self.n if self.n >= 2 else np.zeros(self.dim)
        z = (np.asarray(emb, dtype=np.float64) - self.mean) / np.sqrt(var + 1.0 / self.dim)
        if self.unit_scale:
            z = z / math.sqrt(self.dim)
        oh = np.zeros(N_LETTERS)
        oh[k - 1] = 1.0
        return np.concatenate([z, oh])[None, :]

    def predict_proba(self, emb, k):
        if not self.fitted:
            return np.full(k, 1.0 / k)
        p = self.clf.predict_proba(self._feat(emb, k))[0][:k]
        s = p.sum()
        return p / s if s > 0 else np.full(k, 1.0 / k)

    def predict(self, emb, k):
        if not self.fitted:
            return int(self.rng.integers(k))
        return int(np.argmax(self.predict_proba(emb, k)))

    def update(self, emb, k, y):
        x = self._feat(emb, k)                             # standardized with the statistics of the items seen before
        emb = np.asarray(emb, dtype=np.float64)
        self.n += 1
        d = emb - self.mean
        self.mean += d / self.n
        self.m2 += d * (emb - self.mean)
        self.clf.partial_fit(x, np.array([int(y)]), classes=self.classes)
        self.fitted = True


def curve_metrics(raw_mean, chance, smooth=SMOOTH, margin=STEPS_MARGIN):
    n = raw_mean.size
    acc = np.array([raw_mean[max(0, i - smooth + 1):i + 1].mean() for i in range(n)])
    alc = float(np.mean((acc - chance) / max(1.0 - chance, 1e-12)))
    n20 = max(1, math.ceil(0.2 * n))
    hit = np.nonzero(acc > chance + margin)[0]
    return dict(ALC=alc, eps_last20=float(acc[-n20:].mean() - chance), steps=int(hit[0]) + 1 if hit.size else n + 1,
                censored=not hit.size, acc=acc)


def run_permutation(emb, ks, ys, seed, **hp):
    """One order -> correctness per position; an item without a valid option count is wrong and gives no update."""
    n = len(ks)
    perm = np.random.default_rng(seed).permutation(n)
    correct = np.zeros(n)
    m = Learner(emb.shape[1], seed, **hp)
    for pos, idx in enumerate(perm):
        k, y = ks[idx], ys[idx]
        if k is None:
            continue
        correct[pos] = float(m.predict(emb[idx], k) == y)
        m.update(emb[idx], k, y)
    return correct


def targets(samples):
    """(ks, ys): option count and 0-based gold index per item, None for items the attacker cannot answer."""
    ks, ys = [], []
    for r in samples:
        opts, k, y = r.get("options"), None, None
        if isinstance(opts, list) and 1 <= len(opts) <= N_LETTERS:
            L = mcq_letter(r.get("answer"), opts)
            if L is not None and ord(L) - 65 < len(opts):
                k, y = len(opts), ord(L) - 65
        ks.append(k)
        ys.append(y)
    return ks, ys


def prequential(emb, ks, ys, seeds):
    scored = [k for k in ks if k is not None]
    chance = float(np.mean([1.0 / k for k in scored])) if scored else 0.0
    raws = np.stack([run_permutation(emb, ks, ys, s) for s in seeds])
    m = curve_metrics(raws.mean(0), chance)
    per = np.array([curve_metrics(r, chance)["eps_last20"] for r in raws])
    boot = np.random.default_rng(0).choice(per, size=(1000, per.size), replace=True).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(chance_export=chance, eps_last20=m["eps_last20"], ci_lo=float(lo), ci_hi=float(hi),
                n20=max(1, math.ceil(0.2 * len(ks))), n_mcq=len(scored))


def five_fold(emb, ks, ys, seed=42, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold
    idx = np.array([i for i, k in enumerate(ks) if k is not None])
    chance = float(np.mean([1.0 / ks[i] for i in idx])) if idx.size else 0.0
    if idx.size < folds * 2:
        return dict(chance_export=chance, eps_last20=None, ci_lo=None, ci_hi=None, n20=int(idx.size), n_mcq=int(idx.size))
    X, y = emb[idx].astype(np.float64), np.array([ys[i] for i in idx])
    strat = min(np.bincount(y)[np.unique(y)]) >= folds
    split = (StratifiedKFold(folds, shuffle=True, random_state=seed) if strat
             else KFold(folds, shuffle=True, random_state=seed)).split(X, y)
    correct = np.zeros(idx.size)
    for tr, te in split:
        if len(set(y[tr])) < 2:
            correct[te] = (y[te] == y[tr][0])
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
        P = clf.predict_proba(X[te])
        for j, t in enumerate(te):
            k = ks[idx[t]]
            p = np.zeros(k)
            for c, v in zip(clf.classes_, P[j]):
                if c < k:
                    p[c] = v
            correct[t] = float(int(np.argmax(p)) == y[t])
    acc = float(correct.sum() / len(ks))                         # items without a valid option count are wrong
    se = math.sqrt(max(acc * (1 - acc), 0.0) / len(ks))
    return dict(chance_export=chance, eps_last20=acc - chance, ci_lo=acc - chance - 1.96 * se,
                ci_hi=acc - chance + 1.96 * se, n20=len(ks), n_mcq=int(idx.size))


def run(ctx, benchmarks=None, limit=None):
    from .embeddings import DEFAULT_MODEL, embed_cached
    mode = ctx.get("pool_attack_mode", "prequential")
    n_perms = int(ctx.get("pool_attack_permutations", 50))
    rows, benches = [], set()
    for bench in select(ctx, benchmarks):
        samples = load_samples(ctx, bench, limit)
        removed = removed_items(ctx, bench)
        items = [r for q, r in samples.items() if q not in removed]
        if len(items) < 2:
            continue
        benches.add(bench)
        emb = embed_cached(ctx, bench, "text", [r["qid"] for r in items],
                           [item_text(r.get("question"), r.get("options")) for r in items])
        ks, ys = targets(items)
        if mode == "five_fold":
            d = five_fold(emb, ks, ys, int(ctx.get("seed", 42)))
        else:
            d = prequential(emb, ks, ys, list(range(1, n_perms + 1)))
        f = lambda v: "" if v is None else f"{v:.4f}"
        rows.append(dict(benchmark=bench, attacker="learned", mode=mode, n_items=len(items), n_mcq=d["n_mcq"],
                         chance_export=f(d["chance_export"]), eps_last20=f(d["eps_last20"]),
                         acc_last20=f(None if d["eps_last20"] is None else d["eps_last20"] + d["chance_export"]),
                         ci_lo=f(d["ci_lo"]), ci_hi=f(d["ci_hi"]), n20=d["n20"],
                         n_perms=n_perms if mode != "five_fold" else "", embedding_model=ctx.get("embedding_model", DEFAULT_MODEL)))
        log(f"pool_attack [{mode}] {bench}: margin over chance {rows[-1]['eps_last20'] or 'n/a'} "
            f"(chance {rows[-1]['chance_export']}, {len(items)} items)", "audit")
    path = ctx.path("table", name="pool_attack.csv")
    old = [r for r in csv.DictReader(open(path)) if not (r["benchmark"] in benches and r["attacker"] == "learned")] \
        if os.path.exists(path) else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(old + rows, key=lambda r: (r["benchmark"], r["attacker"])))
    return rows
