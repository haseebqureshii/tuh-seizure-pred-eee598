# # TUSZ Preprocessing Pipeline – Smart Canonical Montage (Reference Alignment)
# 
# ## High-level
# This notebook builds a seizure vs background window dataset from TUH Seizure Corpus (TUSZ) tcp-bipolar EEG.
# 
# It does:
# - Recursively walks `train/`, `dev/`, or `eval` under `ROOT_DIR`.
# - For each EDF:
#   - Load EEG + its seizure annotations (.csv).
#   - Bandpass 0.5–40 Hz using a Kaiser FIR, zero-phase (`filtfilt`).
#   - Downsample to 250 Hz *if* original fs is higher; never upsample.
#   - Robust z-score per channel (median/IQR).
#   - Slice into 10 s windows with 5 s hop.
#   - Label each window seizure(1) or background(0), pooling all seizure subtypes.
# - Skips recordings that are too short for `filtfilt` padding.
# - Saves:
#   - `split_windows.npz` with windows, labels, and metadata.
#   - `qc_window.png` plotting an informative example window.
#   - `class_balance.png` showing seizure/background counts.
# 
# ## Smart Canonical Montage
# We do **not** force a giant hard-coded canonical bipolar channel list upfront.
# Instead we learn a canonical layout from *inside the split*:
# 
# 1. The **first usable recording** becomes our **reference layout**.
#    - We store its channel names/ordering. Call this `ref_ch_names`.
# 2. Every later recording is aligned to that same reference layout:
#    - For each reference channel name (like `FP1-F7`), we try to find the best-matching channel name in the new recording using a robust normalizer.
#    - If found, we copy that channel.
#    - If missing, we fill zeros in that row.
# 3. We compute a **coverage ratio** = fraction of reference channels we could actually fill with real data from this recording.
#    - If coverage is too low (default <30%), we SKIP that recording to avoid mostly-zero data.
# 
# So we get:
# - Consistent `[C, T]` across all recordings (good for training).
# - Mostly real EEG, not all zeros.
# - QC plot that reflects the aligned tensors you’ll actually train on.
# 
# ## Prediction Labeling
# 
# - **Positive (y = 1)**: **pre-ictal** windows only — the interval *before onset* where a predictor should alarm.  
#   Defined per seizure as **[onset − SOP_MIN, onset − SPH_MIN]**.
# - **Negative (y = 0)**: **interictal** background **plus** a short **early-ictal** portion immediately after onset:  
#   **[onset, min(onset + EARLY_ICTAL_KEPT_S, offset)]**.
# - **Ignored** (not used for training): the **prediction horizon** just before onset **(onset − SPH_MIN, onset)**, the **remainder of ictal** after the early-ictal portion, and an optional **post-ictal** buffer **(offset, offset + POSTICTAL_BUFFER_S]** when enabled.
# - **Artifacts**: windows overlapping artifact intervals are **dropped**; if `KEEP_PREICTAL_EVEN_IF_ARTIFACT = True`, pre-ictal windows are **kept** even when overlapping artifacts (positives are scarce).
# 
# ### Timing variables (all in seconds)
# - `SOP_MIN` — Seizure Occurrence Period look-back (e.g., 30*60).  
# - `SPH_MIN` — Seizure Prediction Horizon “no-alarm” band (e.g., 5*60).  
# - `EARLY_ICTAL_KEPT_S` — early-ictal kept as **negative** (e.g., 10).  
# - `POSTICTAL_BUFFER_S` — optional ignore after offset (0 disables).  
# - `KEEP_PREICTAL_EVEN_IF_ARTIFACT` — keep/don’t keep pre-ictal windows that overlap artifacts.
# 
# ### Note on normalization (robust z)
# Pre-ictal windows can be very quiet (IQR≈0). Robust z-score uses a **safe denominator**  
# `denom = max(IQR, 0.1*STD, 1e-6)` to prevent extreme amplitudes while preserving behavior elsewhere.
# 
# ## Runtime knobs
# - `MAX_SUBJECTS`: cap how many subjects to include (fast debug).
# - `MAX_RECORDINGS`: cap recordings per subject (or overall).
# - `NUM_TAPS`: FIR length. Shorter => fewer clips skipped for being "too short".
# - `GROUP_MODE`: iterate by `"subject"` (recommended) or just all EDFs (`"recording"`).
# - `min_coverage`: skip a recording if < this fraction of reference channels could be mapped.

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from scipy.signal import firwin, filtfilt, resample_poly
from math import gcd
import io, re
import random
import os
import torch

# ------------------ USER CONFIG ------------------
# Point this at the root of the TUSZ split folders that contain train/dev/eval
ROOT_DIR = Path(r"../../../../../../../../../../../../TUH-Seizure-Corpus/edf").expanduser().resolve()

# Output directory for NPZ + plots
SPLIT = "train"  # "train", "dev", or "eval"

OUT_ROOT = Path(r"./tusz_windows").expanduser().resolve()
OUT_DIR  = (OUT_ROOT / SPLIT).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Windowing params
WIN_LEN_S = 30   # seconds per window
HOP_S     = 2.5 * 60    # seconds hop (50% overlap)

# --- Prediction labeling policy (all times in SECONDS) ---
SOP_MIN = 20 * 60          # 20 minutes before seizure onset (pre-ictal start)
SPH_MIN = 10 * 60          # 10 minutes before onset (prediction horizon, ignored)
EARLY_ICTAL_KEPT_S = 0    # every ictal treated as NEGATIVE (y=0)
POSTICTAL_BUFFER_S = 120     # ignore after offset (2min disables)
KEEP_PREICTAL_EVEN_IF_ARTIFACT = False  # keep pre-ictal even if overlapping artifacts

# DSP params
NUM_TAPS  = 1001    # Kaiser FIR taps for bandpass 0.5-40Hz (shorter -> fewer skips)
TARGET_FS = 250.0  # We'll downsample to 250 Hz if fs_orig > 250 Hz; never upsample

# Runtime limiting params
MAX_SUBJECTS   = 203       # limit number of subjects used (debug speed). None = no explicit cap
MAX_RECORDINGS = 20      # max recordings per subject / overall. None = no explicit cap
GROUP_MODE     = "subject" # "subject" or "recording"

# Seizure subtype labels we consider "seizure" (label=1)
# Everything else becomes background (label=0)
SEIZURE_LABELS = {
    'seiz','fnsz','gnsz','spsz','cpsz','absz',
    'tnsz','cnsz','tcsz','atsz','mysz'
}

# Artifact / noise labels in TUSZ annotations.
# Based on TUSZ event map, includes muscle, eye movement, chewing, shiver,
# electrode pops, and combos.
ARTIFACT_LABELS = {
    'artf',        # generic artifact
    'eyem',        # eye movement / blink
    'chew',        # chewing
    'shiv',        # shiver / tremor
    'musc',        # muscle artifact
    'elec',        # electrode pop / line noise
    # common multi-label combos found in TUSZ maps:
    'eyem_chew','eyem_shiv','eyem_musc','eyem_elec',
    'chew_shiv','chew_musc','chew_elec',
    'shiv_musc','shiv_elec',
    'musc_elec'
}

print("ROOT_DIR:", ROOT_DIR)
print("OUT_DIR:", OUT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ## Filtering / resampling / normalization helpers
# 
# ### Bandpass 0.5–40 Hz (Kaiser FIR)
# - We design a linear-phase FIR using `firwin(..., window=('kaiser', beta))`.
# - We run `filtfilt` per channel to get zero-phase.
# - `filtfilt` needs enough padding, so we skip recordings that are too short.
# 
# ### Resampling
# - If original fs > 250 Hz, downsample to 250 using polyphase (`resample_poly`).
# - If fs <= 250 Hz, leave it (no upsampling).
# 
# ### Robust z-score
# - Per channel: subtract median, divide by IQR.
# - Flat channels (IQR ≈ 0) become ~0, which is fine.


def _kaiser_beta_from_ripple(ripple_db: float) -> float:
    """
    Approximate Kaiser beta as a function of desired stopband attenuation in dB.
    """
    A = ripple_db
    if A > 50:
        return 0.1102 * (A - 8.7)
    elif A >= 21:
        return 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
    else:
        return 0.0

def apply_kaiser_bandpass(data, fs_orig, f_lo=0.5, f_hi=40.0, num_taps=101, ripple_db=60.0):

    n_ch, n_samp = data.shape
    padlen_needed = 3 * (num_taps - 1)
    if n_samp <= padlen_needed:
        raise ValueError(
            f"too short for filtfilt len={n_samp} need>{padlen_needed}"
        )

    beta = _kaiser_beta_from_ripple(ripple_db)
    taps = firwin(
        numtaps=num_taps,
        cutoff=[f_lo, f_hi],
        pass_zero=False,
        window=("kaiser", beta),
        fs=fs_orig,
        scale=True,
    )

    # zero-phase per channel
    filtered = np.stack([filtfilt(taps, [1.0], ch, axis=0) for ch in data], axis=0)
    return filtered

def _downsample_for_nonlinear(x, target_len=400):
    """
    Cheap decimation to limit length for O(N^2) nonlinear features.
    Keeps shape/scale roughly similar but makes ApEn/Higuchi tractable.
    """
    x = np.asarray(x)
    N = len(x)
    if N <= target_len:
        return x
    factor = int(np.floor(N / target_len))
    return x[::factor]

def maybe_resample_to_250(data, fs_orig, target_fs=TARGET_FS):

    if abs(fs_orig - target_fs) < 1e-6:
        return data, fs_orig
    if fs_orig < target_fs:
        print(f"[WARN] fs {fs_orig} Hz < {target_fs} Hz, not upsampling.")
        return data, fs_orig

    # integer ratio for resample_poly
    from math import gcd
    up = int(target_fs * 1000)
    down = int(fs_orig * 1000)
    g = gcd(up, down)
    up //= g
    down //= g

    resampled = []
    for ch in data:
        ch_rs = resample_poly(ch, up, down)
        resampled.append(ch_rs)
    resampled = np.stack(resampled, axis=0)
    return resampled, target_fs

def robust_zscore(x, axis=1, eps_floor=1e-6, std_frac=0.1):

    med = np.median(x, axis=axis, keepdims=True)
    q1  = np.percentile(x, 25, axis=axis, keepdims=True)
    q3  = np.percentile(x, 75, axis=axis, keepdims=True)
    iqr = (q3 - q1)
    std = np.std(x, axis=axis, keepdims=True)
    denom = np.maximum(iqr, std_frac * std)
    denom = np.maximum(denom, eps_floor)
    return (x - med) / denom


# ## Annotation loader (CSV)
# 
# TUSZ seizure annotations are in `.csv` alongside the EDF. We:
# 1. Open the `.csv` and try to locate the header line that contains `start` and `stop`.
# 2. Read from that line forward using pandas.
# 3. Extract time intervals for any label in a target label set.
# 
# We create two label sets:
# - `SEIZURE_LABELS`: all events we consider “seizure” for model supervision.
# - `ARTIFACT_LABELS`: all events we consider “artifact / unusable background”, including muscle noise, eye movement, chewing, electrode pops, etc.
# 
# We then define:
# - `load_seizure_intervals(csv_path)` → merged `(start_sec, stop_sec)` intervals containing seizure activity.
# - `load_artifact_intervals(csv_path)` → merged `(start_sec, stop_sec)` intervals containing artifact.
# 
# Merging step:
# - If two intervals overlap or touch, we merge them into a single continuous interval.  
#   This prevents double-counting when annotations are dense.
# 
# 
# These intervals are later used to decide if each 10 s window is seizure, clean background, or should be thrown away due to artifact.
# 


def _parse_annotation_csv(csv_path: Path):

    csv_path = Path(csv_path)
    try:
        raw_text = csv_path.read_text(errors="replace")
    except Exception as e:
        print(f"[WARN] Could not open {csv_path}: {e}")
        return None

    lines = raw_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if ("start" in low) and ("stop" in low):
            header_idx = i
            break

    if header_idx is None:
        # try naive pandas read
        try:
            ann = pd.read_csv(
                csv_path,
                sep=",",
                engine="python",
                on_bad_lines="skip"
            )
        except Exception as e:
            print(f"WARNING: Could not parse {csv_path}: {e}")
            return None
    else:
        main_body = "\n".join(lines[header_idx:])
        try:
            ann = pd.read_csv(
                io.StringIO(main_body),
                sep=",",
                engine="python",
                on_bad_lines="skip"
            )
        except Exception as e:
            print(f"WARNING: Could not smart-parse {csv_path}: {e}")
            return None

    if ann is None or ann.empty:
        return None

    # normalize column names
    ann.columns = [c.strip().lower() for c in ann.columns]
    return ann


def _extract_intervals_from_ann_df(ann_df: pd.DataFrame,
                                   target_label_set: set[str]):

    if ann_df is None or ann_df.empty:
        return []

    # figure out column names for start/stop and label
    if 'start_time' in ann_df.columns:
        start_col = 'start_time'
    elif 'start' in ann_df.columns:
        start_col = 'start'
    else:
        return []

    if 'stop_time' in ann_df.columns:
        stop_col = 'stop_time'
    elif 'stop' in ann_df.columns:
        stop_col = 'stop'
    else:
        return []

    if 'label' in ann_df.columns:
        label_col = 'label'
    elif 'type' in ann_df.columns:
        label_col = 'type'
    else:
        return []

    # normalize the label column
    tmp = ann_df.copy()
    tmp[label_col] = tmp[label_col].astype(str).str.lower().str.strip()

    use_rows = tmp[tmp[label_col].isin(target_label_set)]
    if use_rows.empty:
        return []

    raw_intervals = []
    for _, r in use_rows.iterrows():
        try:
            s = float(r[start_col])
            e = float(r[stop_col])
        except Exception:
            continue
        if not np.isfinite(s) or not np.isfinite(e):
            continue
        if e <= s:
            continue
        raw_intervals.append([s, e])

    if not raw_intervals:
        return []

    # merge overlaps
    raw_intervals.sort(key=lambda x: x[0])
    merged = [raw_intervals[0]]
    for s, e in raw_intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1][1] = max(le, e)
        else:
            merged.append([s, e])

    return merged


def load_seizure_intervals(csv_path: Path):
    ann_df = _parse_annotation_csv(csv_path)
    return _extract_intervals_from_ann_df(ann_df, SEIZURE_LABELS)


def load_artifact_intervals(csv_path: Path):
    ann_df = _parse_annotation_csv(csv_path)
    return _extract_intervals_from_ann_df(ann_df, ARTIFACT_LABELS)


# ## Sliding window extraction and labeling
# 
# We cut the filtered, resampled, z-scored EEG into 10 s windows with 5 s hop.
# 
# For each window `[w_start, w_end]`:
# 1. We label windows for **prediction** as follows, per annotated seizure `(onset, offset)`:
# 
# - **y = 1 (pre-ictal)**: `[onset − SOP_MIN, onset − SPH_MIN]`
# - **ignored (SPH)**: `(onset − SPH_MIN, onset)`
# - **y = 0 (early-ictal)**: `[onset, min(onset + EARLY_ICTAL_KEPT_S, offset)]`
# - **ignored (rest ictal)**: `(onset + EARLY_ICTAL_KEPT_S, offset]`
# - **ignored (post-ictal optional)**: `(offset, offset + POSTICTAL_BUFFER_S]` if `POSTICTAL_BUFFER_S > 0`
# 
# Artifact rule: windows overlapping artifact intervals are dropped, except pre-ictal windows may be kept if `KEEP_PREICTAL_EVEN_IF_ARTIFACT = True`.
# 
# We return:
# - `X_arr`: `[N_win, C, T]`
# - `y_arr`: `[N_win]` seizure/background labels
# - `t0_arr`: `[N_win]` start time (sec from file start)
# 

def window_data_with_artifact(data,
                              fs,
                              seizure_intervals,
                              artifact_intervals,
                              win_len_s,
                              hop_s):
    def _merge(iv):
        if not iv:
            return []
        iv = sorted(iv, key=lambda x: x[0])
        # Ensure the first element is a list for mutability
        merged = [[iv[0][0], iv[0][1]]]
        for s, e in iv[1:]:
            ls, le = merged[-1]
            if s <= le:
                # Modify the list element
                merged[-1][1] = max(le, e)
            else:
                # Append a new list
                merged.append([s, e])
        # Return as a list of tuples as originally intended by the function's usage later
        return [(s, e) for s, e in merged]


    def _regions_prediction(seiz_iv, sop_s, sph_s, early_s, post_s):
        """
        Build regions per seizure:
        pre  -> y=1 (pre-ictal): [onset - SOP, onset - SPH]
        igp  -> ignore: (onset - SPH, onset)
        e0   -> y=0: early ictal [onset, onset + early_s]
        igi  -> ignore: rest ictal
        igpo -> ignore: optional post-ictal buffer
        """
        pre, igp, e0, igi, igpo = [], [], [], [], []
        for on, off in seiz_iv:
            if off <= on:
                continue
            pre.append((max(0.0, on - sop_s), max(0.0, on - sph_s)))
            igp.append((max(0.0, on - sph_s), on))
            ei = min(on + early_s, off)
            if ei > on:
                e0.append((on, ei))
            if off > ei:
                igi.append((ei, off))
            if post_s > 0:
                igpo.append((off, off + post_s))
        return {
            "pre":  _merge([x for x in pre  if x[1] > x[0]]),
            "igp":  _merge([x for x in igp  if x[1] > x[0]]),
            "e0":   _merge([x for x in e0   if x[1] > x[0]]),
            "igi":  _merge([x for x in igi  if x[1] > x[0]]),
            "igpo": _merge([x for x in igpo if x[1] > x[0]]),
        }


    def _overlaps(w, ivs):
        ws, we = w
        for s, e in ivs:
            if ws < e and we > s:
                return True
        return False


    def window_prediction_with_artifacts(data, fs, seizure_iv, artifact_iv,
                                         win_s, hop_s,
                                         sop_s, sph_s, early_s, post_s,
                                         keep_pre_artf=True):
        """
        Prediction labels only (minimal change):
        - y=1 for pre-ictal windows in [onset - SOP, onset - SPH]
        - y=0 for interictal and EARLY ictal [onset, onset + early_s]
        - ignore SPH band, the rest of ictal, and optional post-ictal
        - drop artifact-overlapping windows, except keep pre-ictal if keep_pre_artf=True
        """
        regs = _regions_prediction(seizure_iv, sop_s, sph_s, early_s, post_s)
        pre, igp, e0, igi, igpo = regs["pre"], regs["igp"], regs["e0"], regs["igi"], regs["igpo"]

        C, N = data.shape
        W = int(round(win_s * fs))
        H = int(round(hop_s * fs))
        Xs, Ys, T0 = [], [], []
        drop_a = drop_i = 0

        i = 0
        while i + W <= N:
            seg = data[:, i:i+W]
            ws, we = i / fs, (i + W) / fs
            w = (ws, we)

            # ignore zones (prediction horizon, rest ictal, post-ictal)
            if _overlaps(w, igp) or _overlaps(w, igi) or _overlaps(w, igpo):
                drop_i += 1
                i += H
                continue

            # positive: pre-ictal
            if _overlaps(w, pre):
                if (not keep_pre_artf) and _overlaps(w, artifact_iv):
                    drop_a += 1
                    i += H
                    continue
                Xs.append(seg); Ys.append(1); T0.append(ws)
                i += H
                continue

            # negative: early ictal
            if _overlaps(w, e0):
                if _overlaps(w, artifact_iv):
                    drop_a += 1
                    i += H
                    continue
                Xs.append(seg); Ys.append(0); T0.append(ws)
                i += H
                continue

            # interictal negative (unless artifact)
            if _overlaps(w, artifact_iv):
                drop_a += 1
                i += H
                continue

            Xs.append(seg); Ys.append(0); T0.append(ws)
            i += H

        if not Xs:
            return (np.empty((0, C, W), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.float32),
                    drop_a, drop_i)

        X = np.stack(Xs, axis=0).astype(np.float32)
        y = np.array(Ys, dtype=np.int64)
        t0 = np.array(T0, dtype=np.float32)
        return X, y, t0, drop_a, drop_i

    X, y, t0s, dropped_art, dropped_ignored = window_prediction_with_artifacts(
        data, fs, seizure_intervals, artifact_intervals,
        win_s=win_len_s,
        hop_s=hop_s,
        sop_s=SOP_MIN,
        sph_s=SPH_MIN,
        early_s=EARLY_ICTAL_KEPT_S,
        post_s=POSTICTAL_BUFFER_S,
        keep_pre_artf=KEEP_PREICTAL_EVEN_IF_ARTIFACT,
    )
    # Old code only tracked artifact drops; ignore-count is still logged upstream if you want.
    return X, y, t0s, int(dropped_art)

# %% [markdown]
# ## Matching EDF ↔ CSV
# 
# We try several strategies to find the annotation `.csv` that goes with an EDF:
# 1. Exact stem match: `file.edf` → `file.csv`
# 2. A `.csv` whose stem starts with the EDF stem
# 3. If there's exactly one `.csv` in the folder, assume it's the match
# 

# %%
def find_best_annotation_for_edf(edf_path: Path):
    edf_path = Path(edf_path)
    d = edf_path.parent
    stem = edf_path.stem

    exact = d / f"{stem}.csv"
    if exact.exists():
        return exact

    csvs = list(d.glob('*.csv'))
    pref = [c for c in csvs if c.stem.startswith(stem)]
    if pref:
        return pref[0]

    if len(csvs) == 1:
        return csvs[0]

    return None


# %% [markdown]
# ## Smart Canonical Montage: channel normalization + alignment
# 
# ### `_normalize_ch_name`
# We normalize channel names so variants like:
# - `EEG FP1-F7`
# - `fp1_f7-ref`
# - `FP1–F7`
# map to the same key: `FP1-F7`.
# This handles prefixes (`EEG `), suffixes (`-REF`, `-AVG`, etc.), underscores, fancy dashes.
# 
# ### `align_to_reference`
# For each processed recording *after* filtering/z-scoring/windowing:
# - We know its own channel names (`dbg['channel_names']`).
# - We know the **reference channel layout** picked from the *first* good recording `ref_ch_names`.
# - We build a lookup from normalized channel name → row index in this recording.
# - For each ref channel name, if we find a match, we copy that row; if not, we insert zeros.
# - We track coverage_ratio = fraction of ref channels we successfully filled.
# If coverage is too low (<30% by default), we skip that recording so we don't poison the dataset with mostly zeros.
# 


def _normalize_ch_name(name: str) -> str:
    """
    Normalize a bipolar channel label so that variants like
    'Fp1-F7', 'EEG FP1-F7', 'fp1_f7-ref', 'FP1–F7', 'FP1-F7-AVG'
    all map to 'FP1-F7'.
    We are **not** constructing new bipolar pairs here. We're just
    making naming consistent so we can match channels across recordings.
    """
    n = name.upper().strip()

    # drop leading 'EEG '
    n = re.sub(r'^EEG\s+', '', n)

    # normalize fancy dashes to plain '-'
    n = n.replace('–', '-').replace('—', '-')

    # collapse spaces/underscores into '-'
    n = re.sub(r'[\s_]+', '-', n)

    # remove common reference suffixes like -REF, -AVG, -AV, -LE, A1/A2, M1/M2
    n = re.sub(r'-(REF|AVG|AV|LE|A1|A2|M1|M2)$', '', n)

    # collapse multiple dashes
    n = re.sub(r'-+', '-', n)

    # trim stray '-'
    n = n.strip('- ')

    return n

def align_to_reference(data_curr, ch_names_curr, ref_ch_names):

    norm_lookup = {}
    for i, nm in enumerate(ch_names_curr):
        norm_lookup[_normalize_ch_name(nm)] = i

    T = data_curr.shape[1]
    out = np.zeros((len(ref_ch_names), T), dtype=data_curr.dtype)

    filled = 0
    for i, ref_nm in enumerate(ref_ch_names):
        idx = norm_lookup.get(_normalize_ch_name(ref_nm))
        if idx is not None:
            out[i, :] = data_curr[idx, :]
            filled += 1

    coverage_ratio = filled / max(1, len(ref_ch_names))
    return out, coverage_ratio


# ## Preprocess a single EDF
# 
# Steps:
# 1. Load EDF with MNE.
# 2. 0.5–40 Hz bandpass via Kaiser FIR + zero-phase `filtfilt`.
# 3. Downsample to 250 Hz if fs > 250 (never upsample).
# 4. Robust z-score per channel.
# 5. Load seizure intervals from .csv, merge overlaps.
# 6. Slice into 10s windows / 5s hop.
# 7. Label seizure(1)/background(0).
# 8. We call `window_data_with_artifact(...)`, which slides 10 s / 5 s hop windows across the normalized EEG and returns only the windows we want to train on.
#    - Rules:
#      - Overlap seizure: KEEP (label=1).
#      - Overlap artifact (but no seizure) → DROP.
#      - Otherwise: KEEP as background (label=0).
#    - We count:
#      - how many windows we kept,
#      - how many were labeled seizure vs background,
#      - how many artifact-only windows we dropped.
# 
# Returns:
# - `X` shape `[N_win, C_curr, T]` in THE RECORDING'S OWN channel order (not aligned yet)
# - `y` shape `[N_win]`
# - `meta` dict with bookkeeping
# - `debug_preview` dict with windows, labels, fs, channel_names (used later for alignment & QC)
# 


def preprocess_single_recording_raw(
    edf_path: Path,
    ann_csv_path: Path,
    win_len_s=WIN_LEN_S,
    hop_s=HOP_S,
    num_taps=NUM_TAPS
):

    print(f"INFO ONLY: EDF: {edf_path.as_posix()}")
    print(f"       CSV: {ann_csv_path.as_posix()}")

    # -------------------------------------------------
    # Load raw EDF with MNE
    # -------------------------------------------------
    raw = mne.io.read_raw_edf(
        edf_path.as_posix(),
        preload=True,
        verbose='ERROR'
    )

    data = raw.get_data()      # shape [C_raw, N_samples]
    ch_names = raw.ch_names    # list of length C_raw
    fs_orig = float(raw.info['sfreq'])
    n_samp = data.shape[1]

    print(f"       fs_orig={fs_orig} Hz  ch={data.shape[0]}  n_samples={n_samp}")

    # -------------------------------------------------
    # Check if clip is long enough for filtfilt padding
    # filtfilt needs ~3*(num_taps-1) samples of margin.
    # If not long enough, we cannot safely apply the Kaiser filter.
    # We just skip this recording entirely.
    # -------------------------------------------------
    padlen_needed = 3 * (num_taps - 1)
    if n_samp <= padlen_needed:
        print(f"       [SKIP SHORT] len={n_samp/fs_orig:.1f}s "
              f"(need >{padlen_needed/fs_orig:.1f}s for filtfilt with {num_taps} taps)")
        return (
            np.empty((0,0,0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {
                'recording_id': edf_path.stem,
                'fs': fs_orig,
                'skipped_short': True,
                'num_windows': 0,
                'num_Pre_Ictal_windows': 0,
                'num_Negative_windows': 0,
                'num_artifact_windows_dropped': 0,
                'total_seizure_seconds_in_file': 0.0,
                'total_artifact_seconds_in_file': 0.0,
                'raw_ch_names': ch_names,
                'num_channels_raw': len(ch_names),
            },
            None,
        )

    # -------------------------------------------------
    # 0.5–40 Hz bandpass using Kaiser FIR with zero-phase filtfilt
    # -------------------------------------------------
    data_bp = apply_kaiser_bandpass(
        data,
        fs_orig,
        f_lo=0.5,
        f_hi=40.0,
        num_taps=num_taps,
        ripple_db=60.0,
    )
    print(f"       after bandpass: {data_bp.shape}")

    # -------------------------------------------------
    # Downsample if needed
    # If fs_orig > TARGET_FS (e.g. 500 Hz), we polyphase resample down to 250 Hz
    # for anti-aliasing + speed.
    # If fs_orig <= TARGET_FS, we keep as is (no upsampling).
    # -------------------------------------------------
    data_ds, fs_proc = maybe_resample_to_250(
        data_bp,
        fs_orig,
        target_fs=TARGET_FS
    )
    print(f"       after resample check: {data_ds.shape}, fs_proc={fs_proc}")

    # -------------------------------------------------
    # Robust z-score (median / IQR) per channel
    # -------------------------------------------------
    data_norm = robust_zscore(data_ds)
    print(f"       after robust z-score: {data_norm.shape}")

    # -------------------------------------------------
    # Load seizure + artifact intervals from CSV
    # We will use both to decide which windows to keep.
    # - seizure windows: always kept, labeled 1
    # - artifact-only windows: dropped entirely
    # - clean windows: kept, labeled 0
    # -------------------------------------------------
    seizure_intervals = load_seizure_intervals(ann_csv_path)
    artifact_intervals = load_artifact_intervals(ann_csv_path)

    total_sz_sec = (
        sum(e - s for (s, e) in seizure_intervals)
        if seizure_intervals else 0.0
    )
    total_artf_sec = (
        sum(e - s for (s, e) in artifact_intervals)
        if artifact_intervals else 0.0
    )
     # --- PATCH 2: explicit first/last seizure time from CSV ---
    if seizure_intervals:
        first_seizure_onset_sec = min(s for (s, e) in seizure_intervals)
        last_seizure_offset_sec = max(e for (s, e) in seizure_intervals)
    else:
        first_seizure_onset_sec = None
        last_seizure_offset_sec = None

    print(f"       seizure_intervals={seizure_intervals}")
    print(f"       total seizure sec in file: {total_sz_sec:.2f}")
    print(f"       artifact_intervals={artifact_intervals}")
    print(f"       total artifact sec in file: {total_artf_sec:.2f}")

    # -------------------------------------------------
    # Sliding windows with artifact-aware policy
    # window_data_with_artifact() returns:
    #   X  -> kept windows [N_keep, C_curr, T]
    #   y  -> labels [N_keep] (1=seizure, 0=clean background)
    #   t0s -> start times of windows (sec from start)
    #   dropped_art -> how many windows we removed due to artifact-only contamination
    # -------------------------------------------------
    X, y, t0s, dropped_art = window_data_with_artifact(
        data_norm,
        fs_proc,
        seizure_intervals,
        artifact_intervals,
        win_len_s=win_len_s,
        hop_s=hop_s,
    )

    print(
        f"       => {X.shape[0]} KEPT windows | "
        f"PI={int(np.sum(y==1))} | "
        f"Ng={int(np.sum(y==0))} | "
        f"dropped_artifact={dropped_art}"
    )

    # -------------------------------------------------
    # Build debug_preview for downstream reference alignment and QC plotting.
    # NOTE: channel_names here are the ORIGINAL channel order for THIS EDF.
    # We'll align them to the canonical montage later in build_split().
    # -------------------------------------------------
    debug_preview = None
    if X.shape[0] > 0:
        debug_preview = {
            'recording_id': edf_path.stem,
            'all_windows': X,        # [N_keep, C_curr, T]
            'all_labels': y,         # [N_keep]
            'fs': fs_proc,
            'channel_names': ch_names,
        }

    # -------------------------------------------------
    # Metadata for analysis and auditing
    # We'll also store:
    #   - total_artifact_seconds_in_file
    #   - num_artifact_windows_dropped
    # -------------------------------------------------
    meta = {
        'recording_id': edf_path.stem,
        'fs': fs_proc,
        'num_windows': int(len(y)),
        'num_Pre_Ictal_windows': int(np.sum(y == 1)),
        'num_Negative_windows': int(np.sum(y == 0)),
        'num_artifact_windows_dropped': int(dropped_art),
        'total_seizure_seconds_in_file': total_sz_sec,
        'total_artifact_seconds_in_file': total_artf_sec,
        'first_seizure_onset_sec': first_seizure_onset_sec,
        'last_seizure_offset_sec': last_seizure_offset_sec,
        'skipped_short': False,
        'raw_ch_names': ch_names,
        'num_channels_raw': data_norm.shape[0],
    }

    return X, y, meta, debug_preview


# ## Build a split (`train`, `dev`, or `eval`)
# 
# Algorithm:
# 1. Collect all EDF files under the chosen split dir.
# 2. If `GROUP_MODE == 'subject'`, group EDFs by subject folder and cap at `MAX_SUBJECTS`.
# 3. For each EDF:
#    - Preprocess.
#    - The first usable recording defines `ref_ch_names` (our learned canonical montage order).
#    - For each window in that recording (and all later recordings), align channel order to `ref_ch_names` using `align_to_reference`.
#    - Compute coverage_ratio for that recording (fraction of ref channels actually present).
#    - If coverage_ratio < `min_coverage` (default 0.3 = 30%), skip that recording so we don't add mostly-zero tensors.
# 4. Concatenate aligned windows across recordings -> one big `X_split`, `y_split`.
# 5. Pick the best debug candidate for QC plotting (prefer seizure-heavy recording).
# 


def _extract_subject_id(split_root: Path, edf_path: Path) -> str:
    rel = edf_path.relative_to(split_root)
    parts = rel.parts
    if len(parts) < 2:
        return "unknown"
    return parts[0]

def build_split(split_root: Path,
                group_mode=GROUP_MODE,
                max_subjects=MAX_SUBJECTS,
                max_recordings=MAX_RECORDINGS,
                win_len_s=WIN_LEN_S,
                hop_s=HOP_S,
                num_taps=NUM_TAPS,
                min_coverage=0.3):
    """
    min_coverage:
      If a recording can't map at least this fraction of the reference channels,
      we skip it (to avoid mostly-zero data).
    """

    split_root = Path(split_root)
    print(f"\n INFO ONLY ==== Processing split: {split_root.as_posix()} ====")

    edf_files = list(split_root.rglob("*.edf"))
    print(f"INFO ONLY: Found {len(edf_files)} EDF files under {split_root.as_posix()}")

    X_all_list = []
    y_all_list = []
    metas = []
    debug_candidates = []

    ref_ch_names = None  # learned canonical order from the first usable recording

    #keep per-window recording index
    rec_idx_all_list = []

    def handle_edf_list(edf_iter, subj_id=None):
        nonlocal ref_ch_names
        count_used = 0
        for edf_path in edf_iter:
            if max_recordings is not None and count_used >= max_recordings:
                break

            ann_path = find_best_annotation_for_edf(edf_path)
            if ann_path is None:
                print("[SKIP NO CSV]", edf_path.as_posix(),
                      "CSV candidates:", [c.name for c in edf_path.parent.glob("*.csv")])
                continue

            X_rec, y_rec, meta_rec, dbg = preprocess_single_recording_raw(
                edf_path,
                ann_path,
                win_len_s=win_len_s,
                hop_s=hop_s,
                num_taps=num_taps,
            )

            if X_rec.shape[0] == 0:
                continue

            # first usable recording defines reference channel layout
            if ref_ch_names is None:
                if dbg is None:
                    continue
                ref_ch_names = list(dbg['channel_names'])
                print(f"INFO ONLY: reference channel layout set from {edf_path.stem}:")
                print(ref_ch_names)

            # align every window in this recording to ref_ch_names
            aligned_list = []
            coverages = []
            for w in X_rec:  # w: [C_curr, T]
                aligned_w, cov = align_to_reference(
                    w,
                    dbg['channel_names'],
                    ref_ch_names
                )
                aligned_list.append(aligned_w)
                coverages.append(cov)

            X_rec_aligned = np.stack(aligned_list, axis=0).astype(np.float32)
            avg_cov = float(np.mean(coverages)) if len(coverages) else 0.0
            print(f"       coverage for {edf_path.stem}: {avg_cov*100:.1f}%")

            # skip low-coverage recordings (prevents mostly-zero tensors)
            if avg_cov < min_coverage:
                print(f"       SKIP LOW COVERAGE {edf_path.stem} "
                      f"({avg_cov*100:.1f}% < {min_coverage*100:.1f}%)")
                continue

            # --- KEEP THIS RECORDING ---

            # 1) append features + labels
            X_all_list.append(X_rec_aligned)
            y_all_list.append(y_rec)

            # 2) build meta with coverage + subject_id
            meta_with_cov = dict(meta_rec)
            meta_with_cov['coverage_ratio'] = avg_cov
            if subj_id is not None:
                meta_with_cov['subject_id'] = subj_id
            metas.append(meta_with_cov)

            # 3) record index for each window in this recording
            meta_idx = len(metas) - 1
            rec_idx_rec = np.full(len(y_rec), meta_idx, dtype=np.int32)
            rec_idx_all_list.append(rec_idx_rec)

            count_used += 1

            # candidate for QC plotting
            if dbg is not None:
                debug_candidates.append({
                    'recording_id': dbg['recording_id'],
                    'all_windows': X_rec_aligned,
                    'all_labels': y_rec,
                    'fs': dbg['fs'],
                    'channel_names': ref_ch_names,
                })

    if group_mode == "recording":
        handle_edf_list(edf_files)
    else:
        subj_to_edfs = {}
        for edf_path in edf_files:
            subj_id = _extract_subject_id(split_root, edf_path)
            subj_to_edfs.setdefault(subj_id, []).append(edf_path)

        subjects_sorted = sorted(subj_to_edfs.keys())[:max_subjects]
        print(f"[INFO] Subjects used (cap={max_subjects}): {subjects_sorted}")

        for subj_id in subjects_sorted:
            recs_this_subj = subj_to_edfs[subj_id]
            handle_edf_list(recs_this_subj, subj_id=subj_id)

    # concatenate aligned windows from all accepted recordings
    if len(X_all_list) == 0:
        X_all = np.empty((0,0,0), dtype=np.float32)
        y_all = np.empty((0,), dtype=np.int64)
        rec_idx_all = np.empty((0,), dtype=np.int32)
    else:
        X_all = np.concatenate(X_all_list, axis=0)
        y_all = np.concatenate(y_all_list, axis=0)
        rec_idx_all = np.concatenate(rec_idx_all_list, axis=0)

    # pick best debug candidate for QC plotting (prefer more seizure windows)
    best_dbg = None
    best_sz = -1
    best_total = -1
    for dbg in debug_candidates:
        labels = dbg.get('all_labels', None)
        wins   = dbg.get('all_windows', None)
        if labels is None or wins is None:
            continue

        sz_count = int(np.sum(labels == 1))
        total_w  = int(len(labels))
        if sz_count > best_sz:
            best_dbg = dbg
            best_sz = sz_count
            best_total = total_w
        elif sz_count == best_sz and total_w > best_total:
            best_dbg = dbg
            best_total = total_w

    return X_all, y_all, metas, best_dbg, rec_idx_all


# ## Preprocess all splits (train, dev, eval) with subject cap
#
# For each of the TUSZ splits:
#   - We call build_split(...) on ROOT_DIR / split_name
#   - We cap the number of subjects to MAX_SUBJECTS (per split)
#   - We save outputs into ./tusz_windows/<split_name>/
#   - We generate a QC plot and class-balance plot per split


BASE_OUT_DIR = Path("./tusz_windows").expanduser().resolve()
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_TO_RUN = ["train", "dev", "eval"]  # edit this if you want to skip dev/eval for now

for split_name in SPLITS_TO_RUN:
    split_dir = ROOT_DIR / split_name

    if not split_dir.exists():
        print(f"\n[WARN] Split folder {split_dir.as_posix()} not found. Skipping.")
        continue

    print(f"\n==============================")
    print(f" Building split: {split_name}")
    print(f"==============================")

    # ---- SUBJECT BUDGETS ----
    if split_name == "train":
        max_subj = MAX_SUBJECTS
    elif split_name == "dev":
        max_subj = 0
    elif split_name == "eval":
        max_subj = 5
    else:
        max_subj = MAX_SUBJECTS 

    # Build this split using the SAME MAX_SUBJECTS cap per split
    X_split, y_split, meta_split, dbg_split, rec_idx_split = build_split(
       split_dir,
       group_mode=GROUP_MODE,
       max_subjects=max_subj,
       max_recordings=MAX_RECORDINGS,
       win_len_s=WIN_LEN_S,
       hop_s=HOP_S,
       num_taps=NUM_TAPS,
       min_coverage=0.3,
    )

    print("\n=== SPLIT SUMMARY ===")
    print(f"Split name   : {split_name}")
    print("X_split shape:", getattr(X_split, 'shape', None))
    print("y_split shape:", getattr(y_split, 'shape', None))

    total_windows = len(y_split)
    num_bg = int(np.sum(y_split == 0)) if total_windows > 0 else 0
    num_sz = int(np.sum(y_split == 1)) if total_windows > 0 else 0
    pct_bg = (100.0 * num_bg / total_windows) if total_windows > 0 else 0.0
    pct_sz = (100.0 * num_sz / total_windows) if total_windows > 0 else 0.0

    print(f"Total windows: {total_windows}")
    print(f"Negative windows (0): {num_bg} ({pct_bg:.2f}%)")
    print(f"Pre-Ictal windows (1): {num_sz} ({pct_sz:.2f}%)")
    print(f"Class ratio (bg : sz) ~ {num_bg}:{num_sz}")

    if dbg_split is not None:
        lbls_dbg = dbg_split.get('all_labels', None)
        if lbls_dbg is not None:
            print("Best debug recording:", dbg_split.get('recording_id', 'n/a'))
            print("Pre-Ictal windows in debug rec:", int(np.sum(lbls_dbg == 1)))
    else:
        print('WARNING: No debug candidate available to plot.')

    # --- Per-split output directory ---
    OUT_DIR = (BASE_OUT_DIR / split_name).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- QC plot ----------
    def pick_informative_window(dbg_dict, motion_thresh=1e-6, iqr_eps=1e-6):
        if dbg_dict is None:
            return None, None
        X_all = dbg_dict.get('all_windows')
        y_all = dbg_dict.get('all_labels')
        if X_all is None or y_all is None or len(y_all) == 0:
            return None, None

        # prefer positives
        idx_pos = np.where(y_all == 1)[0]
        idx_neg = np.where(y_all == 0)[0]
        for w_i in list(idx_pos) + list(idx_neg):
            W = X_all[w_i]  # [C,T]
            ch_std = np.std(W, axis=1)
            q25 = np.percentile(W, 25, axis=1)
            q75 = np.percentile(W, 75, axis=1)
            ch_iqr = q75 - q25
            ok = np.where((ch_std > motion_thresh) & (ch_iqr > iqr_eps))[0]
            if ok.size:
                ch_i = int(ok[np.argmax(ch_std[ok])])
                return w_i, ch_i
        return None, None

    if dbg_split is not None:
        w_i, ch_i = pick_informative_window(dbg_split, motion_thresh=1e-6, iqr_eps=1e-6)
        if w_i is not None:
            X_all = dbg_split['all_windows']
            y_all = dbg_split['all_labels']
            fs_dbg = float(dbg_split['fs'])
            sig = X_all[w_i, ch_i, :].astype(float)
            label_dbg = int(y_all[w_i])
            t_axis = np.arange(sig.shape[0]) / fs_dbg

            # re-normalize for plotting
            med = np.median(sig)
            iqr = np.percentile(sig, 75) - np.percentile(sig, 25)
            std = np.std(sig)
            denom = max(iqr, 0.1 * std, 1e-6)
            sig_plot = np.clip((sig - med) / denom, -8.0, 8.0)

            print(f"[QC] plotting window {w_i} label={label_dbg} channel {ch_i}")
            print("     std(raw):", float(std), "IQR(raw):", float(iqr), "denom_used:", float(denom))

            plt.figure(figsize=(10, 4))
            plt.plot(t_axis, sig_plot)
            plt.xlabel('Time (s)')
            plt.ylabel('Robust z-scored amplitude')
            plt.title(f'{split_name} QC window w={w_i} ch={ch_i} label={label_dbg}')
            plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout()
            qc_path = (OUT_DIR / 'qc_window.png').as_posix()
            plt.savefig(qc_path, dpi=200)
            plt.close()
            print('[INFO] Saved QC plot to', qc_path)
        else:
            print('INFO ONLY: No non-flat window/channel found to plot.')
    else:
        print('INFO ONLY: Skipping QC plot; no debug recording found.')

    # ---------- SAVE split_windows.npz + class balance per split ----------
    if X_split is not None and y_split is not None and len(y_split) > 0:
        # 1) Save NPZ (this is what the modeling / SHAP code will load)
        out_npz_path = OUT_DIR / "split_windows.npz"

        if dbg_split is not None and "channel_names" in dbg_split:
            ch_names_ref = np.array(dbg_split["channel_names"], dtype=object)
        else:
            ch_names_ref = None

        np.savez_compressed(
            out_npz_path.as_posix(),
            X=X_split,
            y=y_split,
            meta=meta_split,
            ch_names=ch_names_ref,
            rec_idx=rec_idx_split,
        )
        print("INFO ONLY: Saved NPZ to", out_npz_path.as_posix())

        # 2) Class balance bar plot
        num_bg = int(np.sum(y_split == 0))
        num_sz = int(np.sum(y_split == 1))

        plt.figure(figsize=(4, 4))
        plt.bar(["Pre-ictal (1)", "Negative (0)"], [num_sz, num_bg])
        plt.ylabel("Window count")
        plt.title(f"Class distribution in {split_name}")
        plt.grid(True, axis="y")
        plt.tight_layout()

        dist_path = (OUT_DIR / "class_balance.png").as_posix()
        plt.savefig(dist_path, dpi=200)
        plt.close()
        print("INFO ONLY: Saved class balance bar chart to", dist_path)
    else:
        print(
            f"WARNING: Nothing to save / plot class balance for split {split_name} (no data)."
        )


# ## QC plot
# 
# We pick an informative EEG window from `dbg_split`:
# 1. Prefer seizure windows first (label==1), otherwise background.
# 2. Within that window, pick the channel with the highest std (so we avoid flat zero-fill rows).
# 3. Plot that channel's robust z-scored signal.
# We save this as `qc_window.png` in `OUT_DIR`.
# 
# **Note:** Pre-ictal positives (y=1) are quieter than ictal by design; class balance may be skewed if `SOP_MIN` is small or `SPH_MIN` is large. QC plots use safe robust z-score (denominator clamped by `max(IQR, 0.1*STD, 1e-6)`) to avoid extreme amplitudes on very low-variance channels.
# 
# 


def pick_informative_window(dbg_dict, motion_thresh=1e-6, iqr_eps=1e-6):
    if dbg_dict is None: return None, None
    X_all = dbg_dict.get('all_windows'); y_all = dbg_dict.get('all_labels')
    if X_all is None or y_all is None or len(y_all) == 0: return None, None

    # prefer positives
    idx_pos = np.where(y_all == 1)[0]; idx_neg = np.where(y_all == 0)[0]
    for w_i in list(idx_pos) + list(idx_neg):
        W = X_all[w_i]  # [C,T]
        ch_std = np.std(W, axis=1)
        q25 = np.percentile(W, 25, axis=1); q75 = np.percentile(W, 75, axis=1)
        ch_iqr = q75 - q25
        ok = np.where((ch_std > motion_thresh) & (ch_iqr > iqr_eps))[0]
        if ok.size:
            ch_i = int(ok[np.argmax(ch_std[ok])])
            return w_i, ch_i
    return None, None

if dbg_split is not None:
    w_i, ch_i = pick_informative_window(dbg_split, motion_thresh=1e-6, iqr_eps=1e-6)
    if w_i is not None:
        X_all = dbg_split['all_windows']; y_all = dbg_split['all_labels']; fs_dbg = float(dbg_split['fs'])
        sig = X_all[w_i, ch_i, :].astype(float); label_dbg = int(y_all[w_i])
        t_axis = np.arange(sig.shape[0]) / fs_dbg

        # re-normalize just for the figure so axes are sane
        med = np.median(sig)
        iqr = np.percentile(sig, 75) - np.percentile(sig, 25)
        std = np.std(sig)
        denom = max(iqr, 0.1*std, 1e-6)
        sig_plot = np.clip((sig - med) / denom, -8.0, 8.0)

        print(f"[QC] plotting window {w_i} label={label_dbg} channel {ch_i}")
        print("     std(raw):", float(std), "IQR(raw):", float(iqr), "denom_used:", float(denom))

        plt.figure(figsize=(10,4))
        plt.plot(t_axis, sig_plot)
        plt.xlabel('Time (s)'); plt.ylabel('Robust z-scored amplitude')
        plt.title(f'QC window w={w_i} ch={ch_i} label={label_dbg}')
        plt.grid(True, ls='--', alpha=0.5); plt.tight_layout()
        qc_path = (OUT_DIR / 'qc_window.png').as_posix()
        plt.savefig(qc_path, dpi=200); print('[INFO] Saved QC plot to', qc_path)
    else:
        print('INFO ONLY: No non-flat window/channel found to plot.')
else:
    print('INFO ONLY: Skipping QC plot; no debug recording found.')



# ## Save dataset + class balance
# 
# We save:
# - `split_windows.npz` with:
#   - `X`: all aligned windows for this split `[N_total, C_ref, T]`
#   - `y`: labels `[N_total]`
#   - `meta`: list of per-recording metadata dicts (coverage ratio, seizure sec, etc.)
# - `class_balance.png`: simple seizure vs background bar chart.
# 


if X_split is not None and y_split is not None:
    out_npz_path = OUT_DIR / 'split_windows.npz'

    # Canonical channel names used after alignment
    # dbg_split['channel_names'] is ref_ch_names used in build_split()

    if dbg_split is not None:
        ch_names_ref = np.array(dbg_split['channel_names'], dtype=object)
    else:
        ch_names_ref = None

    total_windows = len(y_split)
    num_bg = int(np.sum(y_split == 0)) if total_windows > 0 else 0
    num_sz = int(np.sum(y_split == 1)) if total_windows > 0 else 0

    plt.figure(figsize=(4,4))
    plt.bar(['Pre-ictal (1)', 'Negative (0)'], [num_sz, num_bg])
    plt.ylabel('Window count')
    plt.title('Class distribution in this split')
    plt.grid(True, axis='y')
    plt.tight_layout()

    dist_path = (OUT_DIR / 'class_balance.png').as_posix()
    plt.savefig(dist_path, dpi=200)
    print('INFO ONLY: Saved class balance bar chart to', dist_path)
else:
    print('WARNING: Nothing to save / plot class balance (no data).')


# Choose which split to load for modeling / SHAP / AUROC sweeps
SPLIT_TO_LOAD = "train"  # or "dev" or "eval"

LOAD_DIR = (BASE_OUT_DIR / SPLIT_TO_LOAD).expanduser().resolve()
npz_path = LOAD_DIR / "split_windows.npz"

print("Loading:", npz_path.as_posix())
data = np.load(npz_path, allow_pickle=True)

X = data["X"]   # shape [N, C_ref, T]
y = data["y"]   # shape [N]
meta_list = data["meta"]  # array of dicts

print("\n=== DATASET SHAPES ===")
print("X.shape:", X.shape) # (N_win, C_ref, T)
print("y.shape:", y.shape) # (N_win,)
N, C_ref, T = X.shape if X.size > 0 else (0,0,0)
print(f"N windows = {N}")
print(f"C_ref (channels after smart canonical) = {C_ref}")
print(f"T samples per window = {T}")

# -------------------------------------------------
# Class balance
# -------------------------------------------------
num_bg = int(np.sum(y == 0))
num_sz = int(np.sum(y == 1))
total = len(y)

pct_bg = 100.0 * num_bg / total if total > 0 else 0.0
pct_sz = 100.0 * num_sz / total if total > 0 else 0.0

print("\n=== CLASS BALANCE ===")
print(f"Negative (0): {num_bg} ({pct_bg:.2f} %)")
print(f"Pre-Ictal (1): {num_sz} ({pct_sz:.2f} %)")
print(f"Negative : Pre-Ictal ratio ~ {num_bg}:{num_sz}")

# -------------------------------------------------
# Metadata summary
# Each element in meta_list corresponds to one *recording* that survived:
# - 'recording_id'
# - 'fs'
# - 'num_windows'
# - 'num_seizure_windows'
# - 'num_background_windows'
# - 'total_seizure_seconds_in_file'
# - 'coverage_ratio'
# - etc.
# -------------------------------------------------

print("\n=== PER-RECORDING META (first 10 rows) ===")
for i, m in enumerate(meta_list[:10]):
    print(f"[{i}] rec_id={m.get('recording_id','?')}")
    print(f" fs_proc={m.get('fs','?')} Hz")
    print(f" windows={m.get('num_windows','?')} "
          f"seiz_win={m.get('num_seizure_windows','?')} "
          f"bg_win={m.get('num_background_windows','?')}")
    print(f" seizure_time_in_file={m.get('total_seizure_seconds_in_file','?')} sec")
    cov = m.get('coverage_ratio', None)
    if cov is not None:
        print(f" coverage={cov*100:.1f}%")
    else:
        print(" coverage=N/A (this was the reference file)")
    print("")

print("=== Seizure onset/offset summary (from CSV) ===")
for m in meta_list:
    print(
        m.get('recording_id', '?'),
        "| onset:", m.get('first_seizure_onset_sec', None),
        "| offset:", m.get('last_seizure_offset_sec', None),
        "| total_sz_sec:", m.get('total_seizure_seconds_in_file', None)
    )
print()

# -------------------------------------------------
# Helper to plot one random window from a given class.
# We'll choose the channel with the highest std in that window
# to avoid plotting a mostly-zero channel.
# -------------------------------------------------

def plot_random_window(X, y, fs, target_label=1, motion_thresh=1e-6):

    idxs = np.where(y == target_label)[0]
    if len(idxs) == 0:
        print(f"No windows with label={target_label}")
        return

    w_idx = int(random.choice(idxs))
    win = X[w_idx] # shape [C_ref, T]

    ch_std = np.std(win, axis=1)
    ch_i = int(np.argmax(ch_std))
    if ch_std[ch_i] <= motion_thresh:
        print(f"Chosen window {w_idx} is too flat (std={ch_std[ch_i]:.2e}). Trying background instead?")
        # we could try fallback logic here, but let's just show anyway
    sig = win[ch_i, :]
    t = np.arange(sig.shape[0]) / fs

    print(f"window {w_idx}, label={target_label}, chosen channel idx={ch_i}, std={ch_std[ch_i]:.3f}")
    plt.figure(figsize=(10,4))
    plt.plot(t, sig)
    plt.xlabel("Time (s)")
    plt.ylabel("Robust z-scored amplitude")
    plt.title(f"Random window {w_idx} (label={target_label}) ch={ch_i}")
    plt.grid(True)
    plt.autoscale()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# We don't store fs directly in X/y, but all aligned data in this split
# share the same fs because we either downsampled to TARGET_FS (250)
# or left lower fs alone. The metadata for any surviving recording
# will tell us. We'll just grab the first one.
# -------------------------------------------------
if len(meta_list) > 0:
    fs_est = meta_list[0].get("fs", TARGET_FS)
else:
    fs_est = TARGET_FS
print(f"\nAssumed sampling rate for plotting: {fs_est} Hz")

# -------------------------------------------------
# Plot example seizure and background windows (if present)
# -------------------------------------------------
plot_random_window(X, y, fs_est, target_label=1) # seizure
plot_random_window(X, y, fs_est, target_label=0) # background