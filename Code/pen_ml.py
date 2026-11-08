#!/usr/bin/env python3
"""
pen_ml.py - handwriting / gesture RECOGNITION core for the two-IMU pen.

The pen's motion sensors can't trace ink accurately, but every word, letter or
shape you write leaves a distinctive motion "signature". This module turns the
live sensor stream into those signatures and learns to recognise them.

Used by:
    pen_trainer.py   build a vocabulary, record labelled samples, train, test
    pen_live.py      live recognition -> text

It can also be used from the command line:
    python pen_ml.py train        train from pen_dataset/ and save pen_model.pkl
    python pen_ml.py info         show what's in the dataset / model

Install:  pip install pygame numpy scikit-learn      (pyserial only for --serial,
          bleak only for --ble, wordfreq for the big letter-mode dictionary)
Keep pen_ml.py, pen_trainer.py, pen_live.py and pen_viewer.py in the same folder:
the mount calibration (pen_calibration.json) made in pen_viewer.py is used here too.

Pipeline
    raw samples -> FeatureStream (12 channels in the PEN's own frame, so they don't
    depend on which way the page faces) -> Segmenter (one item = burst of writing
    followed by a short pause) -> Model (cross-validated: extra-trees on summary
    features, or DTW nearest-neighbour; whichever scores better) -> Transcript.
"""
import json
import math
import os
import pickle
import queue
import random
import sys
import threading
import time
from collections import deque

import numpy as np

import pen_viewer as pv

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "pen_dataset")
VOCAB_FILE = os.path.join(HERE, "vocab.txt")
MODEL_FILE = os.path.join(HERE, "pen_model.pkl")
SETTINGS_FILE = os.path.join(HERE, "pen_ml_settings.json")

FEATURE_VERSION = 1
CHANNELS = ["acc_x", "acc_y", "acc_z",          # linear acceleration, pen frame (m/s^2)
            "dacc_x", "dacc_y", "dacc_z",       # tip sensor minus rear sensor (rotation signature)
            "grav_x", "grav_y", "grav_z",       # which way is up, in the pen frame (tilt / roll)
            "gyro_x", "gyro_y", "gyro_z"]       # rotation rate, pen frame (rad/s)
N_CH = len(CHANNELS)

DEFAULT_VOCAB = ["hello", "star", "circle", "help", "yes", "no"]

DEFAULT_SETTINGS = {
    "gyro_on": 25.0,      # deg/s - writing starts when rotation goes above this...
    "acc_on": 0.6,        # m/s^2 - ...or acceleration goes above this
    "gyro_off": 10.0,     # deg/s - quiet when both are below the "off" levels
    "acc_off": 0.25,      # m/s^2
    "end_quiet_s": 0.6,   # s     - a pause this long ends the item
    "min_len_s": 0.25,    # s     - ignore shorter blips
    "max_len_s": 8.0,     # s     - force an item to end after this long
    "word_gap_s": 1.5,    # s     - for single-letter labels: a gap this long starts a new word
    "reject_below": 0.35, # show "?" when the best guess is less sure than this
    "hold_mode": 0.0,     # 0 = items end with a pause; 1 = hold the pen's BOOT button (or
                          #     Space on the PC) while writing each item
}


def use_demo_files():
    """--demo uses its own dataset / vocabulary / model, so fake data never mixes with yours."""
    global DATASET_DIR, VOCAB_FILE, MODEL_FILE
    DATASET_DIR = os.path.join(HERE, "pen_dataset_demo")
    VOCAB_FILE = os.path.join(HERE, "vocab_demo.txt")
    MODEL_FILE = os.path.join(HERE, "pen_model_demo.pkl")
    if not os.path.exists(VOCAB_FILE):
        save_vocab(list(DEMO_SHAPES))


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE) as f:
            s.update({k: float(v) for k, v in json.load(f).items() if k in s})
    except (OSError, ValueError, TypeError):
        pass
    return s


def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f, indent=2)
        return True
    except OSError:
        return False


# =============================================================================
# Live features
# =============================================================================
class FeatureStream:
    """Feeds raw pen samples into the viewer's Tracker (which applies the mount and
    accelerometer calibration) and produces one 12-channel feature vector per sample."""

    def __init__(self):
        self.tracker = pv.Tracker()
        self.tracker.track_position = False
        self._R_prev = None
        self._gyro = np.zeros(3)
        self.t = 0.0
        self.button = False                # BOOT button on the pen (needs the updated sketch)

    def push(self, s):
        tr = self.tracker
        prev_us = tr.last_us
        tr.feed(s)
        self.button = bool(s["flags"] & 64)
        idx = [i for i in (0, 1) if tr.valid[i] and tr.a_raw[i] is not None]
        if not idx:
            return None
        dt = None
        if prev_us is not None:
            d = ((s["us"] - prev_us) & 0xFFFFFFFF) / 1e6
            if 0.0 < d <= 0.1:
                dt = d
        self.t += dt if dt is not None else 0.01

        R = tr.R_fused
        acc_w = [pv.rotz(tr.psi[i]) @ tr.a_raw[i] for i in idx]
        acc_p = R.T @ np.mean(acc_w, axis=0)
        dacc_p = R.T @ (acc_w[-1] - acc_w[0]) if len(idx) == 2 else np.zeros(3)
        grav_p = R.T @ np.array([0.0, 0.0, 1.0])
        if self._R_prev is not None and dt is not None:
            dR = self._R_prev.T @ R
            self._gyro = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * dt)
        self._R_prev = R.copy()
        return self.t, np.concatenate([acc_p, dacc_p, grav_p, self._gyro]).astype(np.float32)


# =============================================================================
# Segmentation: one item = a burst of writing followed by a short pause
# =============================================================================
class Segmenter:
    def __init__(self, settings):
        self.s = settings
        self.buf_t = deque(maxlen=30)      # pre-roll (~0.3 s)
        self.buf_x = deque(maxlen=30)
        self.seg_t, self.seg_x = [], []
        self.state = "idle"                # idle | writing
        self.gyro_lvl = 0.0
        self.acc_lvl = 0.0
        self.last_active_t = 0.0
        self.start_t = 0.0
        self.last_end_t = None
        self.forced = False

    def activity(self):
        """0..1-ish activity for meters."""
        return max(self.gyro_lvl / max(self.s["gyro_on"], 1e-6), self.acc_lvl / max(self.s["acc_on"], 1e-6))

    def force_end(self):
        self.forced = True

    def reset(self):
        self.state = "idle"
        self.seg_t, self.seg_x = [], []

    def push(self, t, x, hold=False):
        """Returns a finished segment dict, or None. `hold` = button held (hold mode)."""
        k = 0.2   # ~50 ms smoothing at 100 Hz
        self.gyro_lvl += k * (math.degrees(float(np.linalg.norm(x[9:12]))) - self.gyro_lvl)
        self.acc_lvl += k * (float(np.linalg.norm(x[0:3])) - self.acc_lvl)
        s = self.s
        if s.get("hold_mode", 0) >= 0.5:
            return self._push_hold(t, x, hold)
        active = self.gyro_lvl > s["gyro_on"] or self.acc_lvl > s["acc_on"]
        quiet = self.gyro_lvl < s["gyro_off"] and self.acc_lvl < s["acc_off"]

        if self.state == "idle":
            self.buf_t.append(t)
            self.buf_x.append(x)
            if active and not self.forced:
                self.state = "writing"
                self.seg_t, self.seg_x = list(self.buf_t), list(self.buf_x)
                self.start_t = self.seg_t[0]
                self.last_active_t = t
            self.forced = False
            return None

        self.seg_t.append(t)
        self.seg_x.append(x)
        if not quiet:
            self.last_active_t = t
        long_enough_quiet = quiet and (t - self.last_active_t) >= s["end_quiet_s"]
        too_long = (t - self.start_t) >= s["max_len_s"]
        if not (long_enough_quiet or too_long or self.forced):
            return None

        self.forced = False
        self.state = "idle"
        self.buf_t.clear()
        self.buf_x.clear()
        end = self.last_active_t + 0.1
        tt = np.array(self.seg_t)
        keep = tt <= end
        return self._finish(tt[keep], np.array(self.seg_x)[keep])

    def _finish(self, tt, xx):
        self.seg_t, self.seg_x = [], []
        if len(tt) < 5 or tt[-1] - tt[0] < self.s["min_len_s"]:
            return None
        seg = {"t": tt - tt[0], "X": np.asarray(xx, np.float32), "t_start": float(tt[0]), "t_end": float(tt[-1]),
               "gap_before": None if self.last_end_t is None else float(tt[0] - self.last_end_t)}
        self.last_end_t = float(tt[-1])
        return seg

    def _push_hold(self, t, x, hold):
        if self.state == "idle":
            self.buf_t.append(t)
            self.buf_x.append(x)
            if hold:
                self.state = "writing"
                self.seg_t, self.seg_x = list(self.buf_t)[-10:], list(self.buf_x)[-10:]
                self.start_t = self.seg_t[0]
            return None
        self.seg_t.append(t)
        self.seg_x.append(x)
        if hold and (t - self.start_t) < self.s["max_len_s"] and not self.forced:
            return None
        self.forced = False
        self.state = "idle"
        self.buf_t.clear()
        self.buf_x.clear()
        return self._finish(np.array(self.seg_t), np.array(self.seg_x))


def calibrate_quiet(levels_gyro, levels_acc, settings):
    """Set the segmenter thresholds from ~2 s of the pen held still."""
    g95 = float(np.percentile(levels_gyro, 95))
    a95 = float(np.percentile(levels_acc, 95))
    s = dict(settings)
    s["gyro_off"] = max(6.0, 4.0 * g95)
    s["gyro_on"] = max(15.0, 2.2 * s["gyro_off"])
    s["acc_off"] = max(0.15, 4.0 * a95)
    s["acc_on"] = max(0.4, 2.2 * s["acc_off"])
    return s


# =============================================================================
# Dataset on disk
# =============================================================================
def safe_name(label):
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return keep[:40] or "label"


def load_vocab():
    try:
        with open(VOCAB_FILE, encoding="utf-8") as f:
            v = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        return v or list(DEFAULT_VOCAB)
    except OSError:
        return list(DEFAULT_VOCAB)


def save_vocab(vocab):
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        f.write("# one label per line: words, single letters, shapes, or commands like <space> <back> <enter>\n")
        for v in vocab:
            f.write(v + "\n")


def save_sample(seg, label, source="trainer"):
    os.makedirs(DATASET_DIR, exist_ok=True)
    name = f"{safe_name(label)}_{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 99999):05d}.npz"
    path = os.path.join(DATASET_DIR, name)
    np.savez_compressed(path, X=seg["X"], t=seg["t"], label=np.array(label),
                        version=np.array(FEATURE_VERSION), source=np.array(source))
    return path


def load_dataset():
    samples = []
    if not os.path.isdir(DATASET_DIR):
        return samples
    for fn in sorted(os.listdir(DATASET_DIR)):
        if not fn.endswith(".npz"):
            continue
        p = os.path.join(DATASET_DIR, fn)
        try:
            z = np.load(p, allow_pickle=False)
            if int(z["version"]) != FEATURE_VERSION:
                continue
            samples.append({"X": z["X"], "t": z["t"], "label": str(z["label"]), "path": p})
        except Exception:
            continue
    return samples


def dataset_counts(samples=None):
    samples = load_dataset() if samples is None else samples
    c = {}
    for s in samples:
        c[s["label"]] = c.get(s["label"], 0) + 1
    return c


def delete_sample(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


# =============================================================================
# Features for the model
# =============================================================================
N_SEQ = 48


def resample(t, X, n=N_SEQ):
    t = np.asarray(t, float)
    if t[-1] <= t[0]:
        t = np.arange(len(t), dtype=float)
    tt = np.linspace(t[0], t[-1], n)
    return np.stack([np.interp(tt, t, X[:, c]) for c in range(X.shape[1])], axis=1)


def augment(t, X, rng):
    """Small realistic variations: crop, speed, amplitude, noise, time warp."""
    n = len(t)
    a = int(rng.integers(0, max(1, n // 20) + 1))
    b = n - int(rng.integers(0, max(1, n // 20) + 1))
    t, X = t[a:b], X[a:b].copy()
    if len(t) < 5:
        return None
    u = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    warp = u + rng.normal(0, 0.04) * np.sin(math.pi * u)
    t = t[0] + np.maximum.accumulate(warp) * (t[-1] - t[0]) * rng.uniform(0.85, 1.15)
    X[:, 0:6] *= rng.uniform(0.85, 1.15)
    X[:, 9:12] *= rng.uniform(0.85, 1.15)
    X += rng.normal(0, 1, X.shape) * np.array([0.05] * 6 + [0.01] * 3 + [0.02] * 3)
    return t, X


def summary_features(t, X, mu, sd):
    """Fixed-length vector from one segment."""
    R = (resample(t, X) - mu) / sd                     # N_SEQ x 12, standardised
    dur = float(t[-1] - t[0])
    f = [R[::2].ravel()]                               # coarse shape of every channel
    f.append(np.concatenate([R.mean(0), R.std(0), R.min(0), R.max(0),
                             np.abs(np.diff(R, axis=0)).mean(0)]))
    thirds = np.array_split(R, 3)
    f.append(np.concatenate([p.mean(0) for p in thirds]))
    # motion integrals (in physical units): total rotation and a crude path of the pen
    Rp = resample(t, X)
    dt = dur / (N_SEQ - 1)
    rot = Rp[:, 9:12].sum(0) * dt
    v = np.cumsum(Rp[:, 0:3] - Rp[:, 0:3].mean(0), axis=0) * dt
    v -= np.linspace(0, 1, N_SEQ)[:, None] * v[-1]
    path = np.cumsum(v, axis=0) * dt
    path_n = path / (np.abs(path).max() + 1e-6)
    f.append(rot)
    f.append(path_n[::4].ravel())
    f.append([math.log(dur + 1e-3)])
    return np.concatenate([np.ravel(x) for x in f]).astype(np.float32)


def dtw_dist(A, B, band=8):
    """Multichannel DTW with a Sakoe-Chiba band (A, B: n x c, same n)."""
    n = len(A)
    C = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        row = D[i]
        prev = D[i - 1]
        for j in range(lo, hi + 1):
            row[j] = C[i - 1, j - 1] + min(prev[j], prev[j - 1], row[j - 1])
    return D[n, n] / n


# =============================================================================
# Model
# =============================================================================
class Model:
    def __init__(self):
        self.kind = None
        self.labels = []
        self.mu = np.zeros(N_CH)
        self.sd = np.ones(N_CH)
        self.forest = None
        self.templates = []        # for DTW: (label, seq)
        self.temp = 1.0
        self.report = {}

    # ---- prediction ------------------------------------------------------
    def _dtw_seq(self, t, X):
        return (resample(t, X, 32) - self.mu) / self.sd

    def predict(self, seg, top=5):
        t, X = seg["t"], seg["X"]
        if self.kind == "forest":
            p = self.forest.predict_proba(summary_features(t, X, self.mu, self.sd)[None])[0]
            order = np.argsort(-p)[:top]
            return [(str(self.forest.classes_[i]), float(p[i])) for i in order]
        q = self._dtw_seq(t, X)
        best = {}
        for lab, ref in self.templates:
            d = dtw_dist(q, ref, band=6)
            if d < best.get(lab, np.inf):
                best[lab] = d
        labs = list(best)
        d = np.array([best[l] for l in labs])
        w = np.exp(-(d - d.min()) / self.temp)
        w /= w.sum()
        order = np.argsort(-w)[:top]
        return [(str(labs[i]), float(w[i])) for i in order]

    # ---- save / load -----------------------------------------------------
    def save(self, path=None):
        with open(path or MODEL_FILE, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(path=None):
        m = Model()
        with open(path or MODEL_FILE, "rb") as f:
            m.__dict__.update(pickle.load(f))
        return m


def _fit_scaler(samples):
    allx = np.vstack([s["X"] for s in samples]).astype(float)
    mu = allx.mean(0)
    sd = allx.std(0)
    sd[sd < 1e-3] = 1e-3
    return mu, sd


def _fit_forest(samples, mu, sd, rng, n_aug=6):
    from sklearn.ensemble import ExtraTreesClassifier
    F, y = [], []
    for s in samples:
        F.append(summary_features(s["t"], s["X"], mu, sd))
        y.append(s["label"])
        for _ in range(n_aug):
            a = augment(s["t"], s["X"], rng)
            if a is not None:
                F.append(summary_features(a[0], a[1], mu, sd))
                y.append(s["label"])
    clf = ExtraTreesClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=1,
                               class_weight="balanced", n_jobs=-1, random_state=0)
    clf.fit(np.array(F), np.array(y))
    return clf


def _dtw_model(samples, mu, sd):
    m = Model()
    m.kind, m.mu, m.sd = "dtw", mu, sd
    m.templates = [(s["label"], m._dtw_seq(s["t"], s["X"])) for s in samples]
    # temperature from typical nearest-neighbour distances
    ds = []
    for i, (lab, a) in enumerate(m.templates[:60]):
        d = [dtw_dist(a, b, 6) for j, (l2, b) in enumerate(m.templates) if j != i and l2 == lab]
        if d:
            ds.append(min(d))
    m.temp = max(0.05, 0.5 * float(np.median(ds))) if ds else 1.0
    return m


def train(samples=None, progress=print):
    """Cross-validates both model types, keeps the better one, fits it on all data."""
    from sklearn.model_selection import StratifiedKFold
    samples = load_dataset() if samples is None else samples
    counts = dataset_counts(samples)
    labels = sorted(counts)
    if len(labels) < 2:
        raise ValueError("Need samples of at least 2 different labels.")
    if min(counts.values()) < 3:
        few = ", ".join(f"{l} ({n})" for l, n in counts.items() if n < 3)
        raise ValueError(f"Need at least 3 samples of every label - too few: {few}")
    rng = np.random.default_rng(0)
    y = np.array([s["label"] for s in samples])
    k = int(min(5, min(counts.values())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    scores = {"forest": [], "dtw": []}
    confusion = {}
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        progress(f"Cross-validation fold {fold + 1}/{k}...")
        trs = [samples[i] for i in tr]
        mu, sd = _fit_scaler(trs)
        mf = Model()
        mf.kind, mf.mu, mf.sd = "forest", mu, sd
        mf.forest = _fit_forest(trs, mu, sd, rng)
        md = _dtw_model(trs, mu, sd)
        for i in te:
            truth = samples[i]["label"]
            pf = mf.predict(samples[i], top=1)[0][0]
            pd_ = md.predict(samples[i], top=1)[0][0]
            scores["forest"].append(pf == truth)
            scores["dtw"].append(pd_ == truth)
            confusion.setdefault("forest", []).append((truth, pf))
            confusion.setdefault("dtw", []).append((truth, pd_))
    acc = {k_: float(np.mean(v)) for k_, v in scores.items()}
    kind = "forest" if acc["forest"] >= acc["dtw"] else "dtw"
    progress(f"Accuracy - extra-trees {acc['forest']:.0%}, DTW {acc['dtw']:.0%}. Training final {kind} model...")
    mu, sd = _fit_scaler(samples)
    if kind == "forest":
        model = Model()
        model.kind, model.mu, model.sd = "forest", mu, sd
        model.forest = _fit_forest(samples, mu, sd, rng)
    else:
        model = _dtw_model(samples, mu, sd)
    model.labels = labels
    per_label, confused = {}, {}
    for truth, pred in confusion[kind]:
        per_label.setdefault(truth, []).append(truth == pred)
        if truth != pred:
            confused[(truth, pred)] = confused.get((truth, pred), 0) + 1
    model.report = {
        "kind": kind, "cv_accuracy": acc[kind], "cv_all": acc, "folds": k,
        "n_samples": len(samples), "counts": counts,
        "per_label": {l: float(np.mean(v)) for l, v in per_label.items()},
        "confused": sorted(((a, b, n) for (a, b), n in confused.items()), key=lambda r: -r[2])[:8],
        "trained": time.strftime("%Y-%m-%d %H:%M"),
    }
    return model


# =============================================================================
# Turning recognised items into text
# =============================================================================
class Transcript:
    """Items -> text. Whole-word labels become words. Single-letter labels are grouped
    into words (a gap longer than word_gap_s, or a <space> item, starts a new word) and,
    when a Dictionary is given, each group is decoded to the most likely real word.
    Labels <space>, <back> and <enter> act as commands."""

    def __init__(self, settings, decoder=None):
        self.s = settings
        self.decoder = decoder
        self.items = []          # dict(probs, seg, alt, gap, walt)
        self._cache = {}

    def add(self, probs, seg):
        it = {"probs": probs, "seg": seg, "alt": 0, "walt": 0, "gap": seg.get("gap_before") if seg else None}
        self.items.append(it)
        return it

    @staticmethod
    def label_of(it):
        return it["probs"][it["alt"]][0]

    def is_unsure(self, it):
        return it["alt"] == 0 and it["probs"][0][1] < self.s["reject_below"]

    # ---- grouping ---------------------------------------------------------
    def groups(self):
        out = []
        for it in self.items:
            lab = self.label_of(it)
            if lab == "<back>":
                if out and out[-1]["kind"] == "letters":
                    out[-1]["items"].pop()
                    if not out[-1]["items"]:
                        out.pop()
                elif out and out[-1]["kind"] == "word":
                    out.pop()
                continue
            if lab == "<enter>":
                out.append({"kind": "nl", "items": [it]})
                continue
            if lab == "<space>":
                out.append({"kind": "break", "items": [it]})
                continue
            gap = it["gap"] if it["gap"] is not None else 0.0
            if len(lab) == 1:
                if out and out[-1]["kind"] == "letters" and gap < self.s["word_gap_s"]:
                    out[-1]["items"].append(it)
                else:
                    out.append({"kind": "letters", "items": [it]})
            else:
                out.append({"kind": "word", "items": [it]})
        return [g for g in out if g["kind"] != "break"]

    def candidates(self, g):
        """Word candidates [(word, prob)] for a letters group."""
        letters = "".join(self.label_of(it) for it in g["items"])
        if self.decoder is None:
            return [(letters, min(it["probs"][it["alt"]][1] for it in g["items"]))]
        key = tuple(id(it) for it in g["items"])
        if key not in self._cache:
            self._cache[key] = self.decoder.decode([dict(it["probs"]) for it in g["items"]]) or [(letters, 0.0)]
        return self._cache[key]

    def group_text(self, g):
        if g["kind"] == "nl":
            return "\n", False
        if g["kind"] == "word":
            it = g["items"][0]
            return self.label_of(it), self.is_unsure(it)
        c = self.candidates(g)
        k = g["items"][0]["walt"] % len(c)
        return c[k][0], (k == 0 and c[k][1] < self.s["reject_below"])

    # ---- corrections on the last word ---------------------------------------
    def last_group(self):
        gs = self.groups()
        return gs[-1] if gs else None

    def cycle(self):
        self.pick(None)

    def pick(self, n):
        """Choose alternative n (0-based) for the last word; None = next one."""
        g = self.last_group()
        if g is None or g["kind"] == "nl":
            return
        if g["kind"] == "letters" and self.decoder is not None:
            m = len(self.candidates(g))
            it = g["items"][0]
            it["walt"] = (it["walt"] + 1) % m if n is None else min(n, m - 1)
        else:
            it = g["items"][-1]
            m = len(it["probs"])
            it["alt"] = (it["alt"] + 1) % m if n is None else min(n, m - 1)

    def last_info(self):
        """For display: (letters or None, [(option, prob)], chosen index)."""
        g = self.last_group()
        if g is None or g["kind"] == "nl":
            return None
        if g["kind"] == "letters" and self.decoder is not None:
            c = self.candidates(g)
            letters = " ".join(f"{self.label_of(it)}" for it in g["items"])
            return letters, c, g["items"][0]["walt"] % len(c)
        it = g["items"][-1]
        return None, it["probs"], it["alt"]

    def undo(self):
        if self.items:
            self.items.pop()

    def clear(self):
        self.items = []
        self._cache = {}

    def tokens(self):
        """Words and line breaks as [(text, unsure)]; '\n' is a line break."""
        return [self.group_text(g) for g in self.groups()]

    def text(self, mark_unsure=True):
        out = ""
        for w, u in self.tokens():
            if w == "\n":
                out += "\n"
                continue
            if out and not out.endswith("\n"):
                out += " "
            out += ("?" if u and mark_unsure else "") + w
        return out


# =============================================================================
# Dictionary decoding for letter-by-letter writing
# =============================================================================
WORDS_FILE = os.path.join(HERE, "words.txt")
BUILTIN_WORDS = """the be to of and a in that have i it for not on with he as you do at this but his by
from they we say her she or an will my one all would there their what so up out if about who get which go
me when make can like time no just him know take people into year your good some could them see other than
then now look only come its over think also back after use two how our work first well way even new want
because any these give day most us is was are has had were been am hello hi help yes ok okay please thank
thanks sorry name write pen pencil paper word letter test power tower python star circle where why find
here more many very much tell call still try ask need feel become leave put mean keep let begin seem show
hear play run move live believe hold bring happen stand lose pay meet include continue set learn change
lead understand watch follow stop create speak read allow add spend grow open walk win offer remember love
consider appear buy wait serve die send expect build stay fall cut reach kill remain""".split()


class Dictionary:
    """Picks the real word that best explains a sequence of letter guesses.
    Uses the wordfreq package if installed (pip install wordfreq: ~50k English words with
    frequencies), else a small built-in list. Your own words (names, terms) go in words.txt,
    one per line - they're always included."""

    INS = math.log(0.03)     # a segment that isn't a letter of the word (e.g. a stray stroke)
    DEL = math.log(0.03)     # a letter that was missed (merged into its neighbour)
    PRIOR = 0.3              # weight of word frequency vs. the letter evidence
    OOV = math.log(1e-4)     # cost of accepting the raw letters as a non-dictionary word

    def __init__(self, max_words=50000):
        words = {}
        try:
            from wordfreq import top_n_list, word_frequency
            for w in top_n_list("en", max_words):
                if w.isascii() and w.isalpha():
                    words[w] = math.log(word_frequency(w, "en") + 1e-9)
            self.source = f"wordfreq, {len(words)} words"
        except ImportError:
            for r, w in enumerate(BUILTIN_WORDS):
                words.setdefault(w, -5.0 - 0.02 * r)
            self.source = f"built-in, {len(words)} words (pip install wordfreq for 50k)"
        n_user = 0
        try:
            with open(WORDS_FILE, encoding="utf-8") as f:
                for ln in f:
                    w = ln.strip().lower()
                    if w and not w.startswith("#") and w.isalpha():
                        words[w] = max(words.get(w, -99.0), -6.0)
                        n_user += 1
        except OSError:
            pass
        if n_user:
            self.source += f" + {n_user} of yours"
        self.words = words
        self._index = {}

    def _by_len(self, alphabet):
        key = tuple(alphabet)
        if key not in self._index:
            pos = {c: i for i, c in enumerate(alphabet)}
            groups = {}
            for w, pr in self.words.items():
                if all(c in pos for c in w):
                    groups.setdefault(len(w), []).append((w, pr))
            self._index[key] = {m: ([w for w, _ in g], np.array([[pos[c] for c in w] for w, _ in g]),
                                    np.array([p for _, p in g]))
                                for m, g in groups.items()}
        return self._index[key]

    def decode(self, dists, top=5):
        """dists: one {label: prob} per written letter. Returns [(word, prob)]."""
        alphabet = sorted({k.lower() for d in dists for k in d if len(k) == 1 and k.isalpha()})
        if not alphabet or not dists:
            return []
        n = len(dists)
        L = np.full((n, len(alphabet)), math.log(1e-3))
        for i, d in enumerate(dists):
            tot = sum(p for k, p in d.items() if len(k) == 1 and k.isalpha()) or 1.0
            for k, p in d.items():
                if len(k) == 1 and k.isalpha():
                    j = alphabet.index(k.lower())
                    L[i, j] = max(L[i, j], math.log(p / tot + 1e-3))
        idx = self._by_len(alphabet)
        cands, scores = [], []
        for m in range(max(1, n - 1), n + 2):
            if m not in idx:
                continue
            ws, W, pri = idx[m]
            E = np.transpose(L[:, W], (1, 0, 2))          # k x n x m
            k = len(ws)
            D = np.full((k, n + 1, m + 1), -np.inf)
            D[:, 0, 0] = 0.0
            for i in range(n + 1):
                for j in range(m + 1):
                    if i == 0 and j == 0:
                        continue
                    best = np.full(k, -np.inf)
                    if i and j:
                        best = np.maximum(best, D[:, i - 1, j - 1] + E[:, i - 1, j - 1])
                    if i:
                        best = np.maximum(best, D[:, i - 1, j] + self.INS)
                    if j:
                        best = np.maximum(best, D[:, i, j - 1] + self.DEL)
                    D[:, i, j] = best
            sc = D[:, n, m] + self.PRIOR * pri
            keep = np.argsort(-sc)[:top]
            cands += [ws[i] for i in keep]
            scores += [float(sc[i]) for i in keep]
        raw = "".join(alphabet[int(np.argmax(L[i]))] for i in range(n))
        if raw not in cands:
            cands.append(raw)
            scores.append(float(L.max(axis=1).sum()) + self.OOV)
        sc = np.array(scores)
        p = np.exp(sc - sc.max())
        p /= p.sum()
        order = np.argsort(-p)[:top]
        return [(cands[i], float(p[i])) for i in order]


# =============================================================================
# Synthetic writer for --demo (no hardware)
# =============================================================================
def _min_jerk(n):
    s = np.linspace(0, 1, n)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


DEMO_SHAPES = {
    "circle": lambda: [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)],
    "star": lambda: [(math.cos(math.pi / 2 + k * 4 * math.pi / 5), math.sin(math.pi / 2 + k * 4 * math.pi / 5))
                     for k in range(6)],
    "zigzag": lambda: [(-1, 1), (-0.5, -1), (0, 1), (0.5, -1), (1, 1)],
    "L": lambda: [(0, 1), (0, -1), (0.7, -1)],
    "hello": lambda: [(-1, 0), (-0.8, 1), (-0.8, -0.5), (-0.5, 0), (-0.3, -0.5), (0, 0.3), (0.1, -0.5),
                      (0.3, 1), (0.35, -0.5), (0.55, 1), (0.6, -0.5), (0.8, 0), (1, -0.4), (0.8, -0.5), (0.8, 0)],
}


class DemoWriter:
    """Writes synthetic items with pauses. Call set_next(label) to choose what comes next."""

    def __init__(self, out_q, stop, labels=None):
        self.q, self.stop = out_q, stop
        self.labels = [l for l in (labels or DEMO_SHAPES) if self.knows(l)] or list(DEMO_SHAPES)
        self.next = None
        self.rng = np.random.default_rng()
        self.seq = 0
        self.us = 0
        self.size = 0.02           # half-size of a demo item (m)

    @staticmethod
    def knows(label):
        return label in DEMO_SHAPES or (len(label) == 1 and label.isalpha())

    def set_next(self, label):
        self.next = label if self.knows(label) else None

    @staticmethod
    def _letter_path(c):
        # not real letter shapes: a fixed, distinct little stroke pattern per letter
        r = np.random.default_rng(ord(c.lower()))
        return [tuple(p) for p in r.uniform(-1, 1, (r.integers(3, 6), 2))]

    def _sample(self, acc, yaw, pitch, roll):
        q = pv.q_from_euler(yaw, pitch, roll)
        a = np.asarray(acc, float)
        rot = self.rng.normal(0, 0.03, 3)
        s = {"seq": self.seq, "us": self.us & 0xFFFFFFFF, "flags": 3, "src": "demo",
             "q": [q, q], "a": [a + self.rng.normal(0, 0.02, 3), a + rot]}
        self.seq += 1
        self.us += 10000
        return s

    def run(self):
        while not self.stop.is_set():
            label = self.next or self.rng.choice(self.labels)
            self.next = None
            for s in self.item(label):
                if self.stop.is_set():
                    return
                pv._push(self.q, s)
                time.sleep(0.01)

    def item(self, label, rest=None):
        """Rest, then one written item: a list of raw sample dicts."""
        base_yaw, base_pitch = 0.3, 0.9
        if not self.knows(label):
            label = str(self.rng.choice(self.labels))
        n_rest = int(rest * 100) if rest is not None else int(self.rng.uniform(100, 150))
        out = [self._sample(np.zeros(3), base_yaw, base_pitch, 0.0) for _ in range(n_rest)]
        if True:
            pts = np.array(DEMO_SHAPES[label]() if label in DEMO_SHAPES else self._letter_path(label), float)
            ang = self.rng.normal(0, 0.15)
            Rm = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            pts = pts @ Rm.T * self.rng.uniform(0.8, 1.2) * self.size
            path = []
            for p0, p1 in zip(pts[:-1], pts[1:]):
                n = max(6, int(np.linalg.norm(p1 - p0) / 0.02 * 25 * self.rng.uniform(0.8, 1.2)))
                s = _min_jerk(n)[:, None]
                path.extend(p0 + (p1 - p0) * s)
            path = np.array(path)
            vel = np.gradient(path, 0.01, axis=0)
            acc = np.gradient(vel, 0.01, axis=0)
            for k in range(len(path)):
                a3 = np.array([acc[k, 0], acc[k, 1], 0.0])
                out.append(self._sample(a3, base_yaw + 3 * vel[k, 1], base_pitch - 3 * vel[k, 0], 2 * vel[k, 0]))
        return out


def start_source(args, rx, stop, demo_labels=None):
    """Starts the UDP / serial / demo reader thread. Returns (description, demo_writer or None)."""
    if getattr(args, "demo", False):
        use_demo_files()
        w = DemoWriter(rx, stop, demo_labels)
        threading.Thread(target=w.run, daemon=True).start()
        return "DEMO (synthetic writing)", w
    if getattr(args, "ble", None):
        pv.check_ble_available()
        threading.Thread(target=pv.ble_thread, args=(args.ble, rx, stop), daemon=True).start()
        return f"Bluetooth '{args.ble}'", None
    if getattr(args, "serial", None):
        threading.Thread(target=pv.serial_thread, args=(args.serial, args.baud, rx, stop), daemon=True).start()
        return f"USB serial {args.serial}", None
    try:
        import socket
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("", args.port))
        probe.close()
    except OSError as exc:
        sys.exit(f"Cannot listen on UDP port {args.port}: {exc}")
    threading.Thread(target=pv.udp_thread, args=(args.port, rx, stop), daemon=True).start()
    return f"WiFi UDP port {args.port}", None


def add_source_args(ap):
    ap.add_argument("--port", type=int, default=pv.UDP_PORT, help="UDP port (default %(default)s)")
    ap.add_argument("--serial", metavar="COMx", help="read from a USB serial port instead of WiFi")
    ap.add_argument("--ble", nargs="?", const="PenTracker", metavar="NAME",
                    help="connect over Bluetooth LE (default name PenTracker; or give a name/address)")
    ap.add_argument("--baud", type=int, default=pv.SERIAL_BAUD)
    ap.add_argument("--demo", action="store_true", help="synthetic writing, no hardware")


# =============================================================================
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    if cmd == "train":
        m = train()
        m.save()
        r = m.report
        print(f"Saved {MODEL_FILE}: {r['kind']}, cross-validated accuracy {r['cv_accuracy']:.0%} "
              f"on {r['n_samples']} samples")
        for a, b, n in r["confused"]:
            print(f"  '{a}' mistaken for '{b}' x{n}")
    else:
        c = dataset_counts()
        print("Vocabulary:", ", ".join(load_vocab()))
        print("Samples:", ", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "none")
        if os.path.exists(MODEL_FILE):
            r = Model.load().report
            print(f"Model: {r.get('kind')} {r.get('cv_accuracy', 0):.0%} (trained {r.get('trained')})")


if __name__ == "__main__":
    main()
