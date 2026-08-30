#!/usr/bin/env python
"""Session aggregation over locally regenerated event-PAD score cells.

Event scores are working data; frozen FRR=5% cuts are loaded independently
from the detector checkpoint tree.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- config
REPO = Path(__file__).resolve().parents[3]
EVENT_SCORE_ROOT = Path(os.environ.get(
    "ACTREAL_EVENT_SCORE_ROOT",
    REPO / ".actreal-work" / "event_level" / "actreal" / "cells",
))
if not EVENT_SCORE_ROOT.is_absolute():
    EVENT_SCORE_ROOT = REPO / EVENT_SCORE_ROOT
THRESHOLD_ROOT = REPO / "checkpoints" / "detectors"
OUTDIR = str(REPO / ".actreal-work" / "session_level")

ACTIONS = ["tap", "scroll", "swipe", "pinch", "keystroke"]
MODALITIES = ["trajectory_xytime", "imu_only", "imu_trajectory_xytime"]
DETECTORS = ["hmog_style_svm", "hmog_style_rf", "paper_svm", "paper_xgboost",
             "behaveformer_stdat", "authconformer"]
STATS = ["S1_count", "S2_mean", "S3_max", "S4_trimmed", "S5_logodds", "S5R_ranklogodds"]
HEADLINE_STATS = ["S1_count", "S2_mean", "S3_max", "S4_trimmed", "S5_logodds"]

R_REPS = 20
N_BOOT = 2000
FRR_TARGETS = [0.05, 0.01]
CAUGHT_TARGETS = [0.50, 0.80, 0.95]
EPS_LOGIT = 1e-6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------- IO
def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cell_dir(root, action, modality, detector):
    return os.path.join(root, f"{action}__{modality}__{detector}")


def scores_file(cdir):
    for nm in ("test_scores.jsonl.gz", "test_scores.jsonl"):
        p = os.path.join(cdir, nm)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(cdir)


# ------------------------------------------------------------------- roster + loading
class Release:
    """Canonical event roster + per-cell score matrices for one results tree."""

    def __init__(self, root, name, threshold_root):
        self.root = root
        self.name = name
        self.threshold_root = threshold_root
        self.thresholds = {}      # (action, modality, detector) -> frr5
        self.load()

    def load(self):
        log(f"loading release '{self.name}' from {self.root}")
        # ---- pass 1: canonical roster from a reference cell per action
        gen_rows, fake_rows = [], []
        ref_mod, ref_det = MODALITIES[0], DETECTORS[0]
        for a in ACTIONS:
            rows = read_jsonl(scores_file(cell_dir(self.root, a, ref_mod, ref_det)))
            for r in rows:
                (gen_rows if r["label"] == 0 else fake_rows).append(
                    (r["event_id"], a, r["user_id"], r["session_id"]))
        gen_rows.sort(key=lambda t: (t[1], t[0]))
        fake_rows.sort(key=lambda t: (t[1], t[2], t[0]))

        self.gen_ids = [t[0] for t in gen_rows]
        self.gen_action = np.array([ACTIONS.index(t[1]) for t in gen_rows], np.int8)
        self.gen_user = [t[2] for t in gen_rows]
        self.gen_session = [t[3] for t in gen_rows]
        self.fake_ids = [t[0] for t in fake_rows]
        self.fake_action = np.array([ACTIONS.index(t[1]) for t in fake_rows], np.int8)
        self.fake_user = [t[2] for t in fake_rows]

        self.users = sorted(set(self.gen_user))
        uix = {u: i for i, u in enumerate(self.users)}
        self.gen_uidx = np.array([uix[u] for u in self.gen_user], np.int32)
        self.fake_uidx = np.array([uix[u] for u in self.fake_user], np.int32)
        self.n_gen, self.n_fake = len(self.gen_ids), len(self.fake_ids)
        gpos = {e: i for i, e in enumerate(self.gen_ids)}
        fpos = {e: i for i, e in enumerate(self.fake_ids)}

        # ---- pass 2: every cell, aligned to the canonical roster
        self.gen_scores = {}    # (mod, det) -> float64[n_gen]
        self.fake_scores = {}
        self.thr = {}           # (action, mod, det) -> frr5
        self.event_far = {}     # (action, mod, det) -> FAR@frr5
        self.event_frr = {}
        for m in MODALITIES:
            for d in DETECTORS:
                g = np.full(self.n_gen, np.nan)
                f = np.full(self.n_fake, np.nan)
                for a in ACTIONS:
                    cdir = cell_dir(self.root, a, m, d)
                    threshold_file = os.path.join(
                        self.threshold_root,
                        f"{a}__{m}__{d}",
                        "thresholds.json",
                    )
                    with open(threshold_file) as threshold_stream:
                        th = json.load(threshold_stream)
                    assert th["score_direction"] == "larger_is_more_fake", (a, m, d)
                    assert th["selection_split"] == "development" and th["target_frr"] == 0.05
                    self.thr[(a, m, d)] = float(th["frr5"])
                    ga, fa = [], []
                    for r in read_jsonl(scores_file(cdir)):
                        s = float(r["fake_high_score"])
                        if r["label"] == 0:
                            g[gpos[r["event_id"]]] = s
                            ga.append(s)
                        else:
                            f[fpos[r["event_id"]]] = s
                            fa.append(s)
                    ga = np.asarray(ga); fa = np.asarray(fa)
                    cut = self.thr[(a, m, d)]
                    # accept rule D3: accepted iff score < cut ; caught iff score >= cut
                    self.event_far[(a, m, d)] = float(np.mean(fa < cut))
                    self.event_frr[(a, m, d)] = float(np.mean(ga >= cut))
                assert not np.isnan(g).any() and not np.isnan(f).any(), (m, d)
                self.gen_scores[(m, d)] = g
                self.fake_scores[(m, d)] = f
        log(f"  {self.n_gen} genuine / {self.n_fake} fake events, "
            f"{len(self.users)} users, 18 cells loaded")

    # ------------------------------------------------------------- session structure
    def build_sessions(self):
        by_sess = defaultdict(list)
        for i, s in enumerate(self.gen_session):
            by_sess[s].append(i)
        self.session_ids = sorted(by_sess)
        self.sess_events = [np.array(sorted(by_sess[s]), np.int32) for s in self.session_ids]
        self.n_sess = len(self.session_ids)
        self.sess_user = np.array([self.gen_uidx[ev[0]] for ev in self.sess_events], np.int32)
        for si, ev in enumerate(self.sess_events):
            assert len(set(self.gen_uidx[ev].tolist())) == 1
        self.sess_len = np.array([len(e) for e in self.sess_events], np.int32)
        # per session action counts
        self.sess_acount = np.zeros((self.n_sess, len(ACTIONS)), np.int32)
        for si, ev in enumerate(self.sess_events):
            for a in self.gen_action[ev]:
                self.sess_acount[si, a] += 1
        # flat segment layout shared by both arms
        self.seg_start = np.concatenate([[0], np.cumsum(self.sess_len)[:-1]]).astype(np.int64)
        self.seg_end = np.cumsum(self.sess_len).astype(np.int64)
        self.seg_id = np.repeat(np.arange(self.n_sess), self.sess_len)
        self.gen_flat = np.concatenate(self.sess_events).astype(np.int32)
        # fake pools per (user, action)
        self.fake_pool = {}
        for u in range(len(self.users)):
            for a in range(len(ACTIONS)):
                self.fake_pool[(u, a)] = np.flatnonzero(
                    (self.fake_uidx == u) & (self.fake_action == a)).astype(np.int32)
        # shortfall audit: max single-session demand and cumulative demand vs pool
        self.shortfalls = []
        need_cum = defaultdict(int)
        for si in range(self.n_sess):
            u = int(self.sess_user[si])
            for a in range(len(ACTIONS)):
                c = int(self.sess_acount[si, a])
                if c == 0:
                    continue
                need_cum[(u, a)] += c
                if c > len(self.fake_pool[(u, a)]):
                    self.shortfalls.append(
                        dict(kind="within_session", session=self.session_ids[si],
                             user=self.users[u], action=ACTIONS[a], need=c,
                             pool=int(len(self.fake_pool[(u, a)]))))
        self.pool_headroom = {}
        for (u, a), need in need_cum.items():
            self.pool_headroom[f"{self.users[u]}|{ACTIONS[a]}"] = dict(
                cumulative_need=int(need), pool=int(len(self.fake_pool[(u, a)])),
                ratio=float(need) / len(self.fake_pool[(u, a)]))
        log(f"  {self.n_sess} genuine sessions, len min {self.sess_len.min()} "
            f"median {np.median(self.sess_len):.0f} mean {self.sess_len.mean():.2f} "
            f"max {self.sess_len.max()}; within-session shortfalls: {len(self.shortfalls)}")

    def draw_fake_sessions(self, seed):
        """Mirror every genuine session; returns flat fake event indices in seg layout."""
        rng = np.random.default_rng(seed)
        flat = np.empty(int(self.seg_end[-1]), np.int32)
        for si in range(self.n_sess):
            u = int(self.sess_user[si])
            pos = int(self.seg_start[si])
            for a in range(len(ACTIONS)):
                c = int(self.sess_acount[si, a])
                if c == 0:
                    continue
                pool = self.fake_pool[(u, a)]
                take = rng.choice(pool, size=c, replace=False)  # no replacement in-session
                flat[pos:pos + c] = take
                pos += c
            assert pos == int(self.seg_end[si])
        return flat


# --------------------------------------------------------------- session statistics
def logit_clipped(s):
    c = np.clip(s, EPS_LOGIT, 1.0 - EPS_LOGIT)
    return np.log(c / (1.0 - c))


def rank_logit(all_scores, s):
    """Monotone in-cell calibration: ECDF rank of s within all_scores -> logit."""
    order = np.argsort(all_scores, kind="mergesort")
    srt = all_scores[order]
    n = srt.size
    # mid-rank ECDF, mapped into (0,1) with the (r-0.5)/n plotting position
    lo = np.searchsorted(srt, s, "left")
    hi = np.searchsorted(srt, s, "right")
    p = ((lo + hi) * 0.5) / n
    p = np.clip(p, 0.5 / n, 1.0 - 0.5 / n)
    return np.log(p / (1.0 - p))


def session_stats(flat_scores, flat_caught, flat_logit, flat_ranklogit,
                  seg_start, seg_end, seg_id, seg_len):
    """All statistics for one arm, fully vectorised over sessions."""
    out = {}
    csum = np.concatenate([[0.0], np.cumsum(flat_scores)])
    out["S2_mean"] = (csum[seg_end] - csum[seg_start]) / seg_len
    ccnt = np.concatenate([[0], np.cumsum(flat_caught.astype(np.int64))])
    out["S1_count"] = (ccnt[seg_end] - ccnt[seg_start]).astype(np.float64)
    out["S3_max"] = np.maximum.reduceat(flat_scores, seg_start)
    clog = np.concatenate([[0.0], np.cumsum(flat_logit)])
    out["S5_logodds"] = clog[seg_end] - clog[seg_start]
    crl = np.concatenate([[0.0], np.cumsum(flat_ranklogit)])
    out["S5R_ranklogodds"] = crl[seg_end] - crl[seg_start]
    # S4: mean of the top ceil(n/3) scores, via within-segment sort
    order = np.lexsort((flat_scores, seg_id))
    ssorted = flat_scores[order]
    cs = np.concatenate([[0.0], np.cumsum(ssorted)])
    k = np.ceil(seg_len / 3.0).astype(np.int64)
    top_start = seg_end - k
    out["S4_trimmed"] = (cs[seg_end] - cs[top_start]) / k
    return out


# -------------------------------------------------------------------- ROC / operating
def auc_mw(gen, fake):
    """P(fake > gen) + 0.5 P(fake == gen), tie corrected, via vectorised mid-ranks."""
    n_g, n_f = gen.size, fake.size
    if n_g == 0 or n_f == 0:
        return np.nan
    allv = np.concatenate([gen, fake])
    order = np.argsort(allv, kind="mergesort")
    srt = allv[order]
    n = srt.size
    new = np.empty(n, bool)
    new[0] = True
    np.not_equal(srt[1:], srt[:-1], out=new[1:])
    dense = np.cumsum(new)                      # 1-based group id per sorted position
    grp_start = np.flatnonzero(new)             # first index of each group
    bounds = np.concatenate([grp_start, [n]])
    midrank = 0.5 * (bounds[dense - 1] + bounds[dense] + 1)   # average of 1-based ranks
    ranks = np.empty(n, np.float64)
    ranks[order] = midrank
    u = ranks[n_g:].sum() - n_f * (n_f + 1) / 2.0
    return float(u / (n_g * n_f))


def cut_at_target(cal_gen, q):
    """Cut c such that flagging {x > c} gives calibration FRR <= q. Returns c."""
    n = cal_gen.size
    if n == 0:
        return np.inf
    k = int(math.floor(q * n))
    if k >= n:
        return -np.inf
    g = np.sort(cal_gen)[::-1]
    return float(g[k])


def apply_cut(vals, cut):
    return vals > cut


def roc_curve(gen, fake):
    """Threshold sweep with rule flag iff x >= v. Returns (v, frr, caught) descending v."""
    vals = np.unique(np.concatenate([gen, fake]))
    gs = np.sort(gen); fs = np.sort(fake)
    frr = 1.0 - np.searchsorted(gs, vals, "left") / gs.size
    caught = 1.0 - np.searchsorted(fs, vals, "left") / fs.size
    # prepend the "flag everything" point
    vals = np.concatenate([[-np.inf], vals])
    frr = np.concatenate([[1.0], frr])
    caught = np.concatenate([[1.0], caught])
    # append "flag nothing"
    vals = np.concatenate([vals, [np.inf]])
    frr = np.concatenate([frr, [0.0]])
    caught = np.concatenate([caught, [0.0]])
    return vals, frr, caught


def price_of_detection(gen, fake, targets):
    vals, frr, caught = roc_curve(gen, fake)
    out = {}
    for t in targets:
        ok = caught >= t - 1e-12
        if not ok.any():
            out[str(t)] = dict(session_frr=None, caught=None, reachable=False)
        else:
            j = int(np.argmin(np.where(ok, frr, np.inf)))
            out[str(t)] = dict(session_frr=float(frr[j]), caught=float(caught[j]),
                               reachable=True)
    return out


def oracle_point(gen, fake, q):
    c = cut_at_target(gen, q)
    return dict(cut=None if not np.isfinite(c) else float(c),
                caught=float(np.mean(apply_cut(fake, c))),
                frr=float(np.mean(apply_cut(gen, c))))


def disjoint_point(gen, fake, inA, q):
    """Calibrate on A eval on B and vice versa; pool the two evaluation folds.
    `inA` is a boolean mask over sessions (True = user in calibration half A)."""
    inB = ~inA
    n_flag_g = n_g = n_flag_f = n_f = 0
    cuts = []
    for cal, ev in ((inA, inB), (inB, inA)):
        if cal.sum() == 0 or ev.sum() == 0:
            continue
        c = cut_at_target(gen[cal], q)
        cuts.append(c)
        n_flag_g += int(apply_cut(gen[ev], c).sum()); n_g += int(ev.sum())
        n_flag_f += int(apply_cut(fake[ev], c).sum()); n_f += int(ev.sum())
    if n_g == 0 or n_f == 0:
        return dict(caught=np.nan, frr=np.nan, cuts=[])
    return dict(caught=n_flag_f / n_f, frr=n_flag_g / n_g,
                cuts=[None if not np.isfinite(c) else float(c) for c in cuts])


def disjoint_price(gen, fake, inA, targets):
    """Sweep the calibration FRR target q; report min honest eval FRR reaching each T."""
    nA, nB = int(inA.sum()), int((~inA).sum())
    qs = np.unique(np.concatenate([
        np.arange(0, nA + 1) / max(nA, 1), np.arange(0, nB + 1) / max(nB, 1)]))
    qs = np.clip(qs + 1e-12, 0.0, 1.0)
    caught = np.empty(qs.size); frr = np.empty(qs.size)
    for i, q in enumerate(qs):
        r = disjoint_point(gen, fake, inA, q)
        caught[i] = r["caught"]; frr[i] = r["frr"]
    out = {}
    for t in targets:
        ok = caught >= t - 1e-12
        if not ok.any():
            out[str(t)] = dict(session_frr=None, caught=None, reachable=False, q=None)
        else:
            j = int(np.argmin(np.where(ok, frr, np.inf)))
            out[str(t)] = dict(session_frr=float(frr[j]), caught=float(caught[j]),
                               reachable=True, q=float(qs[j]))
    return out, qs, frr, caught


# ------------------------------------------------------------------------ statistics
def mean_ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fmt(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


# ------------------------------------------------------------------------------ main
def run(root, name, outprefix, n_boot=N_BOOT, reps=R_REPS,
        threshold_root=THRESHOLD_ROOT):
    rel = Release(root, name, threshold_root)
    rel.build_sessions()
    U = len(rel.users)

    # ------------------------------------------------ session structure descriptives
    prof = Counter()
    for si in range(rel.n_sess):
        prof[tuple(ACTIONS[a] for a in range(5) if rel.sess_acount[si, a] > 0)] += 1
    structure = dict(
        n_users=U, users=rel.users, n_sessions=int(rel.n_sess),
        n_genuine_events=int(rel.n_gen), n_fake_events=int(rel.n_fake),
        session_len=dict(min=int(rel.sess_len.min()), p5=float(np.percentile(rel.sess_len, 5)),
                         p25=float(np.percentile(rel.sess_len, 25)),
                         median=float(np.median(rel.sess_len)), mean=float(rel.sess_len.mean()),
                         p75=float(np.percentile(rel.sess_len, 75)),
                         p95=float(np.percentile(rel.sess_len, 95)),
                         max=int(rel.sess_len.max())),
        action_profiles={"+".join(k) if k else "(empty)": v for k, v in
                         sorted(prof.items(), key=lambda kv: -kv[1])},
        within_session_shortfalls=rel.shortfalls,
        n_within_session_shortfalls=len(rel.shortfalls),
        max_cumulative_need_over_pool=max(v["ratio"] for v in rel.pool_headroom.values()),
    )

    # ------------------------------------------------------------------- user splits
    splits = {
        "primary_first10_last10": np.array([rel.users.index(u) for u in rel.users[:10]]),
        "sensitivity_alternating": np.array([i for i in range(U) if i % 2 == 0]),
    }
    split_doc = {k: [rel.users[i] for i in v] for k, v in splits.items()}

    # -------------------------------------------------- fake session draws (shared)
    log(f"drawing {reps} mirrored fake-session replicates (seeds 0..{reps-1})")
    fake_flats = [rel.draw_fake_sessions(seed) for seed in range(reps)]
    # audit: uniqueness within session for replicate 0
    f0 = fake_flats[0]
    for si in range(rel.n_sess):
        seg = f0[rel.seg_start[si]:rel.seg_end[si]]
        assert len(set(seg.tolist())) == seg.size, si
        assert (rel.fake_uidx[seg] == rel.sess_user[si]).all()
        assert Counter(rel.fake_action[seg].tolist()) == Counter(
            {a: int(rel.sess_acount[si, a]) for a in range(5) if rel.sess_acount[si, a] > 0})
    log("  mirror audit passed: same user, same action multiset, no in-session repeats")

    # ---------------------------------------------------------- per-cell statistics
    cells = [(m, d) for m in MODALITIES for d in DETECTORS]
    # gen_stat[(m,d)][stat] -> (n_sess,)   fake_stat[(m,d)][stat] -> (reps, n_sess)
    gen_stat, fake_stat = {}, {}
    event_far_cell, event_frr_cell = {}, {}
    for (m, d) in cells:
        gs = rel.gen_scores[(m, d)]
        fs = rel.fake_scores[(m, d)]
        thr_vec_g = np.array([rel.thr[(ACTIONS[a], m, d)] for a in rel.gen_action])
        thr_vec_f = np.array([rel.thr[(ACTIONS[a], m, d)] for a in rel.fake_action])
        caught_g = gs >= thr_vec_g          # D3
        caught_f = fs >= thr_vec_f
        pool_all = np.concatenate([gs, fs])
        lg_g, lg_f = logit_clipped(gs), logit_clipped(fs)
        rl_g, rl_f = rank_logit(pool_all, gs), rank_logit(pool_all, fs)
        idx = rel.gen_flat
        gen_stat[(m, d)] = session_stats(gs[idx], caught_g[idx], lg_g[idx], rl_g[idx],
                                         rel.seg_start, rel.seg_end, rel.seg_id, rel.sess_len)
        per_rep = {s: np.empty((reps, rel.n_sess)) for s in STATS}
        for r in range(reps):
            fi = fake_flats[r]
            st = session_stats(fs[fi], caught_f[fi], lg_f[fi], rl_f[fi],
                               rel.seg_start, rel.seg_end, rel.seg_id, rel.sess_len)
            for s in STATS:
                per_rep[s][r] = st[s]
        fake_stat[(m, d)] = per_rep
        event_far_cell[(m, d)] = float(np.mean([rel.event_far[(a, m, d)] for a in ACTIONS]))
        event_frr_cell[(m, d)] = float(np.mean([rel.event_frr[(a, m, d)] for a in ACTIONS]))
    log("session statistics computed for all 18 cells x 6 statistics x (1 + R) arms")

    # ---------------------------------------------------------------- point estimates
    halfA = splits["primary_first10_last10"]
    su = rel.sess_user
    maskA = np.isin(su, halfA)
    per_cell = {}
    for (m, d) in cells:
        for s in STATS:
            g = gen_stat[(m, d)][s]
            recs = dict(auc=[], oracle={f"{q}": dict(caught=[], frr=[]) for q in FRR_TARGETS},
                        disjoint={f"{q}": dict(caught=[], frr=[]) for q in FRR_TARGETS},
                        oracle_price={f"{t}": [] for t in CAUGHT_TARGETS},
                        disjoint_price={f"{t}": [] for t in CAUGHT_TARGETS},
                        disjoint_price_caught={f"{t}": [] for t in CAUGHT_TARGETS})
            for r in range(reps):
                f = fake_stat[(m, d)][s][r]
                recs["auc"].append(auc_mw(g, f))
                for q in FRR_TARGETS:
                    o = oracle_point(g, f, q)
                    recs["oracle"][f"{q}"]["caught"].append(o["caught"])
                    recs["oracle"][f"{q}"]["frr"].append(o["frr"])
                    dj = disjoint_point(g, f, maskA, q)
                    recs["disjoint"][f"{q}"]["caught"].append(dj["caught"])
                    recs["disjoint"][f"{q}"]["frr"].append(dj["frr"])
                op = price_of_detection(g, f, CAUGHT_TARGETS)
                for t in CAUGHT_TARGETS:
                    recs["oracle_price"][f"{t}"].append(op[str(t)]["session_frr"])
                dp, _, _, _ = disjoint_price(g, f, maskA, CAUGHT_TARGETS)
                for t in CAUGHT_TARGETS:
                    recs["disjoint_price"][f"{t}"].append(dp[str(t)]["session_frr"])
                    recs["disjoint_price_caught"][f"{t}"].append(dp[str(t)]["caught"])
            per_cell[f"{m}|{d}|{s}"] = recs
        log(f"  point estimates done: {m} / {d}")

    # ------------------------------------------------------------------- bootstrap
    log(f"user-cluster bootstrap: {n_boot} replicates, shared user multisets")
    brng = np.random.default_rng(20260809)
    boot_users = brng.integers(0, U, size=(n_boot, U))
    sess_of_user = [np.flatnonzero(su == u) for u in range(U)]
    boot_idx = []
    boot_maskA = []
    for b in range(n_boot):
        pick = np.concatenate([sess_of_user[u] for u in boot_users[b]])
        boot_idx.append(pick)
        boot_maskA.append(np.isin(su[pick], halfA))

    boot_metrics = {}   # key -> {metric: array(n_boot)}
    for (m, d) in cells:
        for s in STATS:
            gfull = gen_stat[(m, d)][s]
            arrs = dict(auc=np.full(n_boot, np.nan))
            for q in FRR_TARGETS:
                arrs[f"oracle_caught@{q}"] = np.full(n_boot, np.nan)
                arrs[f"disjoint_caught@{q}"] = np.full(n_boot, np.nan)
                arrs[f"disjoint_frr@{q}"] = np.full(n_boot, np.nan)
            for t in CAUGHT_TARGETS:
                arrs[f"oracle_price@{t}"] = np.full(n_boot, np.nan)
            for b in range(n_boot):
                pick = boot_idx[b]
                g = gfull[pick]
                f = fake_stat[(m, d)][s][b % reps][pick]
                mA = boot_maskA[b]
                arrs["auc"][b] = auc_mw(g, f)
                for q in FRR_TARGETS:
                    arrs[f"oracle_caught@{q}"][b] = oracle_point(g, f, q)["caught"]
                    dj = disjoint_point(g, f, mA, q)
                    arrs[f"disjoint_caught@{q}"][b] = dj["caught"]
                    arrs[f"disjoint_frr@{q}"][b] = dj["frr"]
                pp = price_of_detection(g, f, CAUGHT_TARGETS)
                for t in CAUGHT_TARGETS:
                    v = pp[str(t)]["session_frr"]
                    arrs[f"oracle_price@{t}"][b] = np.nan if v is None else v
            boot_metrics[f"{m}|{d}|{s}"] = arrs
        log(f"  bootstrap done: {m} / {d}")

    # ------------------------------------------------------- assemble per-cell output
    metric_names = (["auc"]
                    + [f"oracle_caught@{q}" for q in FRR_TARGETS]
                    + [f"disjoint_caught@{q}" for q in FRR_TARGETS]
                    + [f"disjoint_frr@{q}" for q in FRR_TARGETS]
                    + [f"oracle_price@{t}" for t in CAUGHT_TARGETS])
    cell_out = {}
    for (m, d) in cells:
        for s in STATS:
            key = f"{m}|{d}|{s}"
            recs = per_cell[key]
            bm = boot_metrics[key]
            e = {}
            e["event_far_at_frr5"] = fmt(event_far_cell[(m, d)])
            e["event_frr_at_frr5"] = fmt(event_frr_cell[(m, d)])
            e["auc"] = dict(mean=fmt(np.mean(recs["auc"])),
                            rep_min=fmt(np.min(recs["auc"])), rep_max=fmt(np.max(recs["auc"])),
                            ci=[fmt(x) for x in mean_ci(bm["auc"])])
            for q in FRR_TARGETS:
                for kind in ("oracle", "disjoint"):
                    c = recs[kind][f"{q}"]["caught"]; fr = recs[kind][f"{q}"]["frr"]
                    ent = dict(caught_mean=fmt(np.nanmean(c)),
                               caught_rep_min=fmt(np.nanmin(c)), caught_rep_max=fmt(np.nanmax(c)),
                               realised_session_frr=fmt(np.nanmean(fr)))
                    bk = f"{kind}_caught@{q}"
                    if bk in bm:
                        ent["caught_ci"] = [fmt(x) for x in mean_ci(bm[bk])]
                    e[f"{kind}_at_frr{q}"] = ent
            e["oracle_price"] = {}
            e["disjoint_price"] = {}
            for t in CAUGHT_TARGETS:
                v = [x for x in recs["oracle_price"][f"{t}"] if x is not None]
                e["oracle_price"][f"{t}"] = dict(
                    session_frr_mean=fmt(np.mean(v)) if v else None,
                    rep_min=fmt(np.min(v)) if v else None, rep_max=fmt(np.max(v)) if v else None,
                    ci=[fmt(x) for x in mean_ci(bm[f"oracle_price@{t}"])])
                w = [x for x in recs["disjoint_price"][f"{t}"] if x is not None]
                wc = [x for x in recs["disjoint_price_caught"][f"{t}"] if x is not None]
                e["disjoint_price"][f"{t}"] = dict(
                    session_frr_mean=fmt(np.mean(w)) if w else None,
                    caught_mean=fmt(np.mean(wc)) if wc else None,
                    reachable_reps=len(w), n_reps=reps)
            cell_out[key] = e

    # ------------------------------------------------------------------- aggregates
    def aggregate(keys, label):
        agg = {}
        for s in STATS:
            sel = [f"{m}|{d}|{s}" for (m, d) in keys]
            a = {}
            a["n_cells"] = len(sel)
            a["event_far_at_frr5"] = fmt(np.mean([cell_out[k]["event_far_at_frr5"] for k in sel]))
            a["auc"] = fmt(np.mean([cell_out[k]["auc"]["mean"] for k in sel]))
            a["auc_ci"] = [fmt(x) for x in mean_ci(
                np.nanmean(np.vstack([boot_metrics[k]["auc"] for k in sel]), axis=0))]
            for q in FRR_TARGETS:
                for kind in ("oracle", "disjoint"):
                    a[f"{kind}_caught_at_frr{q}"] = fmt(np.mean(
                        [cell_out[k][f"{kind}_at_frr{q}"]["caught_mean"] for k in sel]))
                    a[f"{kind}_realised_frr_at_frr{q}"] = fmt(np.mean(
                        [cell_out[k][f"{kind}_at_frr{q}"]["realised_session_frr"] for k in sel]))
                    bk = f"{kind}_caught@{q}"
                    a[f"{kind}_caught_at_frr{q}_ci"] = [fmt(x) for x in mean_ci(
                        np.nanmean(np.vstack([boot_metrics[k][bk] for k in sel]), axis=0))]
            for t in CAUGHT_TARGETS:
                vals = [cell_out[k]["oracle_price"][f"{t}"]["session_frr_mean"] for k in sel]
                vals = [v for v in vals if v is not None]
                a[f"oracle_price_at_caught{t}"] = fmt(np.mean(vals)) if vals else None
                a[f"oracle_price_at_caught{t}_ci"] = [fmt(x) for x in mean_ci(
                    np.nanmean(np.vstack([boot_metrics[k][f"oracle_price@{t}"] for k in sel]),
                               axis=0))]
                dv = [cell_out[k]["disjoint_price"][f"{t}"]["session_frr_mean"] for k in sel]
                dv = [v for v in dv if v is not None]
                a[f"disjoint_price_at_caught{t}"] = fmt(np.mean(dv)) if dv else None
                dc = [cell_out[k]["disjoint_price"][f"{t}"]["caught_mean"] for k in sel]
                dc = [v for v in dc if v is not None]
                a[f"disjoint_price_at_caught{t}_realised_caught"] = fmt(np.mean(dc)) if dc else None
            # calibration gap
            for q in FRR_TARGETS:
                a[f"calibration_gap_at_frr{q}"] = fmt(
                    a[f"oracle_caught_at_frr{q}"] - a[f"disjoint_caught_at_frr{q}"])
            agg[s] = a
        return {label: agg}

    aggregates = {}
    aggregates.update(aggregate(cells, "all_18"))
    for m in MODALITIES:
        aggregates.update(aggregate([(m, d) for d in DETECTORS], f"modality::{m}"))
    for d in DETECTORS:
        aggregates.update(aggregate([(m, d) for m in MODALITIES], f"detector::{d}"))

    # ---------------------------------------------------- S1 integer count-rule table
    log("building the S1 integer count-rule table")
    kmax = int(rel.sess_len.max())
    s1_rule = {}
    kgrid = list(range(1, kmax + 1))
    frr_k = np.zeros(len(kgrid)); caught_k = np.zeros(len(kgrid))
    per_cell_k = {}
    for (m, d) in cells:
        g = gen_stat[(m, d)]["S1_count"]
        fr = np.array([np.mean(g >= k) for k in kgrid])
        ca = np.mean([[np.mean(fake_stat[(m, d)]["S1_count"][r] >= k) for k in kgrid]
                      for r in range(reps)], axis=0)
        per_cell_k[f"{m}|{d}"] = dict(frr=[fmt(x) for x in fr], caught=[fmt(x) for x in ca])
        frr_k += fr / len(cells); caught_k += ca / len(cells)
    ok = np.flatnonzero(frr_k <= 0.05)
    best_deploy = int(kgrid[int(ok[0])]) if ok.size else None
    j = caught_k - frr_k
    best_j = int(kgrid[int(np.argmax(j))])
    s1_rule = dict(
        k_grid=kgrid,
        aggregate_session_frr=[fmt(x) for x in frr_k],
        aggregate_caught=[fmt(x) for x in caught_k],
        best_k_under_frr5=best_deploy,
        best_k_under_frr5_point=None if best_deploy is None else dict(
            k=best_deploy, session_frr=fmt(frr_k[kgrid.index(best_deploy)]),
            caught=fmt(caught_k[kgrid.index(best_deploy)])),
        best_k_youden=best_j,
        best_k_youden_point=dict(k=best_j, session_frr=fmt(frr_k[kgrid.index(best_j)]),
                                 caught=fmt(caught_k[kgrid.index(best_j)]),
                                 youden_j=fmt(j[kgrid.index(best_j)])),
        per_cell=per_cell_k)

    # ------------------------------------------------------------- sensitivity split
    log("sensitivity: alternating user split")
    maskA2 = np.isin(su, splits["sensitivity_alternating"])
    sens = {}
    for s in STATS:
        rows = []
        for (m, d) in cells:
            g = gen_stat[(m, d)][s]
            c = np.mean([disjoint_point(g, fake_stat[(m, d)][s][r], maskA2, 0.05)["caught"]
                         for r in range(reps)])
            rows.append(c)
        sens[s] = fmt(np.mean(rows))
    # ------------------------------------------------------------------- ROC curves
    log("dumping ROC curves (replicate 0, 201-point FRR grid)")
    grid = np.linspace(0, 1, 201)
    curves = {}
    for (m, d) in cells:
        for s in STATS:
            g = gen_stat[(m, d)][s]; f = fake_stat[(m, d)][s][0]
            vals, frr, caught = roc_curve(g, f)
            # step-interpolate caught at each grid FRR (max caught with frr <= x)
            o = np.argsort(frr)
            fs_, cs_ = frr[o], np.maximum.accumulate(caught[o])
            cg = np.interp(grid, fs_, cs_)
            curves[f"{m}|{d}|{s}"] = [round(float(x), 5) for x in cg]
    curve_blob = dict(frr_grid=[round(float(x), 5) for x in grid], curves=curves,
                      note="caught-rate as a function of session FRR, replicate 0, all 20 users")

    # ------------------------------------------------------------------------ output
    result = dict(
        meta=dict(
            release=name, root=str(root), score_root=str(root),
            threshold_root=str(threshold_root),
            python=sys.version.split()[0], numpy=np.__version__,
            n_boot=n_boot, n_fake_replicates=reps,
            threshold_field="frr5 (thresholds.json, selection_split=development, target_frr=0.05)",
            accept_rule="ACCEPTED iff score < threshold; CAUGHT iff score >= threshold "
                        "(score_direction=larger_is_more_fake, verified in thresholds.json 90/90)",
            session_flag_rule="session FLAGGED iff stat > cut (strict, see D6)",
            user_splits=split_doc,
            s5_note="S5 is the literal clip-to-[1e-6,1-1e-6] logit sum; S5R is a secondary "
                    "rank-calibrated variant for the two SVM detectors whose raw decision "
                    "values leave [0,1]",
        ),
        structure=structure,
        event_level=dict(
            far_at_frr5_per_cell={f"{a}|{m}|{d}": fmt(rel.event_far[(a, m, d)], 6)
                                  for a in ACTIONS for m in MODALITIES for d in DETECTORS},
            frr_at_frr5_per_cell={f"{a}|{m}|{d}": fmt(rel.event_frr[(a, m, d)], 6)
                                  for a in ACTIONS for m in MODALITIES for d in DETECTORS},
            far_at_frr5_mean_over_90=fmt(np.mean([rel.event_far[(a, m, d)] for a in ACTIONS
                                                  for m in MODALITIES for d in DETECTORS]), 6),
            frr_at_frr5_mean_over_90=fmt(np.mean([rel.event_frr[(a, m, d)] for a in ACTIONS
                                                  for m in MODALITIES for d in DETECTORS]), 6),
        ),
        per_cell=cell_out,
        aggregates=aggregates,
        s1_count_rule=s1_rule,
        sensitivity_alternating_split_disjoint_caught_at_frr5=sens,
    )
    op = os.path.join(OUTDIR, f"{outprefix}_results.json")
    json.dump(result, open(op, "w"), indent=1, sort_keys=False)
    json.dump(curve_blob, open(os.path.join(OUTDIR, f"{outprefix}_curves.json"), "w"))
    log(f"wrote {op}")

    # ------------------------------------------------------------------ console table
    print("\n================ AGGREGATE OVER ALL 18 (modality, detector) ================")
    print(f"{'stat':<16}{'AUC':>8}{'D-c@5%':>9}{'D-c@1%':>9}{'O-c@5%':>9}{'O-c@1%':>9}"
          f"{'p@50':>8}{'p@80':>8}{'p@95':>8}")
    for s in STATS:
        a = aggregates["all_18"][s]
        print(f"{s:<16}{a['auc']:>8.4f}{a['disjoint_caught_at_frr0.05']:>9.4f}"
              f"{a['disjoint_caught_at_frr0.01']:>9.4f}{a['oracle_caught_at_frr0.05']:>9.4f}"
              f"{a['oracle_caught_at_frr0.01']:>9.4f}"
              f"{a['oracle_price_at_caught0.5']:>8.4f}{a['oracle_price_at_caught0.8']:>8.4f}"
              f"{a['oracle_price_at_caught0.95']:>8.4f}")
    print("\nS1 count rule (aggregate over 18 cells):", json.dumps(
        dict(best_k_under_frr5=s1_rule["best_k_under_frr5_point"],
             best_k_youden=s1_rule["best_k_youden_point"])))
    for lab in [f"modality::{m}" for m in MODALITIES] + [f"detector::{d}" for d in DETECTORS]:
        a = aggregates[lab]["S1_count"]
        b = aggregates[lab]["S2_mean"]
        print(f"{lab:<34} S1 auc {a['auc']:.4f} d@5% {a['disjoint_caught_at_frr0.05']:.4f} | "
              f"S2 auc {b['auc']:.4f} d@5% {b['disjoint_caught_at_frr0.05']:.4f}")
    return result


def main():
    global OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--reps", type=int, default=R_REPS)
    ap.add_argument(
        "--cells", type=Path, default=EVENT_SCORE_ROOT,
        help=("regenerated event-score cells; default: ACTREAL_EVENT_SCORE_ROOT "
              "or .actreal-work/event_level/actreal/cells"),
    )
    ap.add_argument(
        "--threshold-root", type=Path, default=THRESHOLD_ROOT,
        help="frozen detector checkpoint/threshold root",
    )
    ap.add_argument("--output-dir", default=OUTDIR)
    args = ap.parse_args()
    OUTDIR = args.output_dir
    os.makedirs(OUTDIR, exist_ok=True)
    run(args.cells, "paper_release", "paper", args.boot, args.reps,
        args.threshold_root)


if __name__ == "__main__":
    main()
