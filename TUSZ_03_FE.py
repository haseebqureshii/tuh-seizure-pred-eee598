
"""
Drop this after your preprocessing. Expects either:
 - `windows`: numpy array (n_windows, n_channels, n_samples), cleaned and optionally already TCP
 - `channel_names`: list of channel names corresponding to columns (e.g., ['FP1-LE', ...] or TCP names)
 - `sfreq`: sampling frequency (Hz)
 - flag `is_le` True if input is linked-ear referential (will convert to TCP)

Outputs:
 - features: numpy array (n_windows, n_features)
 - feature_names: list[str]
 - optional temporal features (n_windows, n_temporal_features)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend, safe with Parallel
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, hilbert
import itertools
import math
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings
from scipy.integrate import trapezoid
from joblib import Parallel, delayed
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split

# --- DISABLE TKINTER DESTRUCTORS IF TK EXISTS (VS Code sometimes loads Tk) ---
try:
    import tkinter as _tk
    if hasattr(_tk, "Image"):
        _tk.Image.__del__ = lambda self: None
    if hasattr(_tk, "Variable"):
        _tk.Variable.__del__ = lambda self: None
except Exception:
    pass

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

# -------------- TCP mapping (ACNS TCP 22 channels) --------------
# mapping: tcp_name -> (electrode_A, electrode_B) where those are LE names without '-LE'
TCP_22_MAPPING = {
    'FP1-F7': ('FP1', 'F7'),
    'F7-T3':  ('F7', 'T3'),
    'T3-T5':  ('T3', 'T5'),
    'T5-O1':  ('T5', 'O1'),
    'FP2-F8': ('FP2', 'F8'),
    'F8-T4':  ('F8', 'T4'),
    'T4-T6':  ('T4', 'T6'),
    'T6-O2':  ('T6', 'O2'),
    'A1-T3':  ('A1', 'T3'),
    'T3-C3':  ('T3', 'C3'),
    'C3-CZ':  ('C3', 'CZ'),
    'CZ-C4':  ('CZ', 'C4'),
    'C4-T4':  ('C4', 'T4'),
    'T4-A2':  ('T4', 'A2'),
    'FP1-F3': ('FP1', 'F3'),
    'F3-C3':  ('F3', 'C3'),
    'C3-P3':  ('C3', 'P3'),
    'P3-O1':  ('P3', 'O1'),
    'FP2-F4': ('FP2', 'F4'),
    'F4-C4':  ('F4', 'C4'),
    'C4-P4':  ('C4', 'P4'),
    'P4-O2':  ('P4', 'O2'),
}
TCP_ORDER = list(TCP_22_MAPPING.keys())

# -------------- Utilities --------------

def le_to_tcp_dataset(windows_le, ch_names_le, mapping=TCP_22_MAPPING):
    """
    Convert LE-referenced windows to TCP bipolar montage using ACNS 22 mapping.

    windows_le: (N_windows, C_le, T)
    ch_names_le: list/array of LE channel names (e.g. 'FP1-LE', 'EEG F7-LE', etc.)

    Returns:
      windows_tcp: (N_windows, n_tcp, T)
      tcp_names:   list[str] of TCP channel names (e.g. 'FP1-F7', ...)
    """
    windows_le = np.asarray(windows_le)
    N, C_le, T = windows_le.shape

    # normalize LE names similarly to your le_to_tcp_array
    norm = [nm.replace('EEG ', '').replace('-LE', '').strip().upper()
            for nm in ch_names_le]
    name_to_idx = {nm: i for i, nm in enumerate(norm)}

    tcp_signals = []
    tcp_names = []

    for tcp_name, (a, b) in mapping.items():
        a_u, b_u = a.upper(), b.upper()
        if a_u in name_to_idx and b_u in name_to_idx:
            ia, ib = name_to_idx[a_u], name_to_idx[b_u]
            # (A-LE) - (B-LE) => bipolar A-B
            sig = windows_le[:, ia, :] - windows_le[:, ib, :]
        else:
            # Missing one or both electrodes → fill NaNs
            sig = np.full((N, T), np.nan, dtype=windows_le.dtype)

        tcp_signals.append(sig)
        tcp_names.append(tcp_name)

    windows_tcp = np.stack(tcp_signals, axis=1)  # (N, n_tcp, T)
    return windows_tcp, tcp_names


def le_to_tcp_array(le_windows, channel_names, sfreq, mapping=TCP_22_MAPPING):
    """
    Convert windows from LE referential to TCP bipolar channels.
    le_windows: (n_windows, n_ch, n_samples)
    channel_names: list of electrode names present in LE (e.g., ['FP1','F7',...,'A1','A2'])
                   can include '-LE' suffix; function strips it.
    Returns: tcp_windows: (n_windows, n_tcp, n_samples), tcp_channel_names
    """
    # Normalize channel names: remove common suffixes/prefixes and uppercase
    norm = [c.replace('-LE', '').replace('EEG ', '').strip().upper() for c in channel_names]
    name_to_idx = {n: i for i, n in enumerate(norm)}
    n_windows = le_windows.shape[0]
    n_samples = le_windows.shape[2]
    tcp_names = []
    tcp_arr = []
    for tcp_ch, (a, b) in mapping.items():
        a_u, b_u = a.upper(), b.upper()
        if a_u in name_to_idx and b_u in name_to_idx:
            ia, ib = name_to_idx[a_u], name_to_idx[b_u]
            # compute bipolar: (A-LE) - (B-LE) => difference of cleaned LE signals
            tcp_signal = le_windows[:, ia, :] - le_windows[:, ib, :]
            tcp_arr.append(tcp_signal)  # list of arrays (n_windows, n_samples)
            tcp_names.append(tcp_ch)
        else:
            # missing -> fill nan
            tcp_arr.append(np.full((n_windows, n_samples), np.nan))
            tcp_names.append(tcp_ch)
    tcp_arr = np.stack(tcp_arr, axis=1)  # (n_windows, n_tcp, n_samples)
    return tcp_arr, tcp_names

def safe_scalar(x, default=0.0):
    """Return a finite scalar; replace NaNs or infinities with a default value."""
    try:
        x = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x

def bandpass(data, sfreq, low, high, order=4):
    ny = 0.5 * sfreq
    b, a = signal.butter(order, [low/ny, high/ny], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)

# %% [markdown]
# ## Features

# %%
# -------------- Morphological features per-channel per-window --------------
def peak_to_peak(x):
    return np.max(x) - np.min(x)

def rms(x):
    return np.sqrt(np.mean(x**2))

def zero_crossing_rate(x):
    # number of sign changes divided by length-1
    zc = np.sum((x[:-1] * x[1:]) < 0)
    return zc / max(1, (len(x)-1))

def line_length(x):
    return np.sum(np.abs(np.diff(x)))

def sharpness_index(x, sfreq):
    # estimate main peak and half-width
    idx = np.argmax(np.abs(x))
    peak = np.abs(x[idx])
    half = peak / 2.0
    # find left index where signal crosses half (absolute)
    left = idx
    while left > 0 and np.abs(x[left]) > half:
        left -= 1
    right = idx
    while right < len(x)-1 and np.abs(x[right]) > half:
        right += 1
    width_samples = max(1, right - left)
    width_sec = width_samples / sfreq
    return (peak / width_sec) if width_sec>0 else np.nan

def asymmetry_ratio(x, sfreq):
    # rise = time from previous trough to peak; fall = time peak to next trough
    idx = np.argmax(x)
    # previous trough
    left = idx
    while left > 0 and x[left] >= x[left-1]:
        left -= 1
    right = idx
    while right < len(x)-1 and x[right] >= x[right+1]:
        right += 1
    trise = (idx - left) / sfreq if idx>left else 1e-9
    tfall = (right - idx) / sfreq if right>idx else 1e-9
    return trise / tfall if tfall>0 else np.nan

def wavelet_energy(x, wavelet='db4', level=4):
    # simple multilevel DWT energy by level using pywt if available
    try:
        import pywt
    except ImportError:
        # fallback: use bandpassed energy (placeholder)
        return np.sum(x**2)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    energies = [np.sum(c**2) for c in coeffs]
    return np.array(energies)  # can be aggregated or returned as vector


# -------------- Nonlinear / local dynamical features --------------
def band_power(x, sfreq, band):
    f, Pxx = welch(x, fs=sfreq, nperseg=min(len(x), int(sfreq * 2)))
    lo, hi = band
    mask = (f >= lo) & (f <= hi)
    if not np.any(mask):
        return 0.0
    return trapezoid(Pxx[mask], f[mask])

def spectral_entropy(x, sfreq, nfft=256):
    f, Pxx = welch(x, fs=sfreq, nperseg=min(len(x), nfft))
    ps = Pxx / np.sum(Pxx) if np.sum(Pxx)>0 else np.ones_like(Pxx)/len(Pxx)
    return -np.sum(ps * np.log(ps + 1e-12))

def approximate_entropy(x, m=2, r=None, max_len=400):
    """
    ApEn with internal downsampling for speed.
    Still conceptually the same feature, but much cheaper.
    """
    x = _downsample_for_nonlinear(x, target_len=max_len)
    x = np.asarray(x, dtype=float)
    N = x.size
    if N <= m + 2:
        return np.nan

    if r is None:
        std = np.std(x)
        r = 0.2 * std if std > 0 else 1e-9

    # build embedding vectors using stride tricks
    def _embed(m):
        shape = (N - m + 1, m)
        strides = (x.strides[0], x.strides[0])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    Xm = _embed(m)
    Xm1 = _embed(m + 1)

    def _phi(X):
        # pairwise Chebyshev distance
        diff = np.abs(X[:, None, :] - X[None, :, :])  # (N-m+1, N-m+1, m)
        d = np.max(diff, axis=-1)
        C = np.mean(d <= r, axis=0)
        return np.mean(np.log(C + 1e-12))

    try:
        return _phi(Xm) - _phi(Xm1)
    except Exception:
        return np.nan

def higuchi_fd(x, kmax=10, max_len=400):
    """
    Higuchi fractal dimension with internal downsampling.
    """
    x = _downsample_for_nonlinear(x, target_len=max_len)
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < 3:
        return np.nan

    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(1, int(np.floor((N - m) / k)) + 1, dtype=int)
            if idx.size == 0:
                continue
            diff = x[m + idx * k - 1] - x[m + (idx - 1) * k - 1]
            Lm = np.sum(np.abs(diff))
            norm = (N - 1) / (idx.size * k)
            Lk.append((Lm * norm) / k)
        if Lk:
            L.append(np.mean(Lk))

    if len(L) < 2:
        return np.nan

    L = np.array(L)
    k = np.arange(1, len(L) + 1, dtype=float)
    # Fit log(L) vs log(1/k)
    coeffs = np.polyfit(np.log(1.0 / k), np.log(L + 1e-12), 1)
    return coeffs[0]

def nonlinear_energy_operator(x):
    # returns average over window
    x = np.asarray(x)
    psi = x[1:-1]**2 - x[2:]*x[:-2]
    return np.mean(psi) if len(psi)>0 else np.nan

# -------------- Pairwise features --------------
def pairwise_pearson(X_window):
    # X_window: (n_channels, n_samples)
    return np.corrcoef(X_window)

def pairwise_plv(band_signals):
    """
    band_signals: (n_channels, n_samples), already band-passed.
    Returns PLV matrix (n_channels, n_channels).
    """
    band_signals = np.asarray(band_signals, dtype=float)
    # analytic signal for all channels at once
    analytic = hilbert(band_signals, axis=-1)
    phases = np.angle(analytic)  # (n_ch, n_samples)

    # phase differences via broadcasting
    phase_diff = phases[:, None, :] - phases[None, :, :]  # (n_ch, n_ch, n_samples)
    plv = np.abs(np.exp(1j * phase_diff).mean(axis=-1))   # (n_ch, n_ch)
    return plv

def pairwise_coherence(band_signals, sfreq, nperseg=None):
    """
    band_signals: (n_channels, n_samples), already band-passed to desired band.
    Computes average coherence across the (already-limited) spectrum.
    """
    band_signals = np.asarray(band_signals, dtype=float)
    n_ch, n_samp = band_signals.shape

    if nperseg is None:
        nperseg = min(n_samp, int(sfreq))  # ~1 second segments

    # Precompute PSD for all channels along axis=-1
    f, Pxx = signal.welch(
        band_signals,
        fs=sfreq,
        axis=-1,
        nperseg=nperseg
    )  # Pxx: (n_ch, n_freq)

    coh_mat = np.zeros((n_ch, n_ch), dtype=float)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            # cross-spectral density
            _, Pxy = signal.csd(
                band_signals[i],
                band_signals[j],
                fs=sfreq,
                nperseg=nperseg
            )
            # coherence = |Pxy|^2 / (Pxx_i * Pxx_j), averaged over freqs
            num = np.abs(Pxy) ** 2
            den = Pxx[i] * Pxx[j] + 1e-12
            coh = np.mean(num / den)
            coh_mat[i, j] = coh
            coh_mat[j, i] = coh

    return coh_mat

def pairwise_mutual_information(X_window, n_bins=16):
    """
    Discrete-binned symmetric MI matrix.
    X_window: (n_channels, n_samples)
    Uses histogram-based MI, avoids sklearn overhead in inner loops.
    """
    X_window = np.asarray(X_window, dtype=float)
    n_ch, n_samp = X_window.shape

    # digitize per channel with its own bins
    bins = [np.histogram_bin_edges(X_window[i], bins=n_bins) for i in range(n_ch)]
    binned = np.empty((n_ch, n_samp), dtype=int)
    for i in range(n_ch):
        binned[i] = np.digitize(X_window[i], bins[i])

    mi = np.zeros((n_ch, n_ch), dtype=float)

    # precompute marginal histograms + entropies
    marg_counts = []
    marg_entropies = []
    for i in range(n_ch):
        counts, _ = np.histogram(binned[i], bins=n_bins)
        p = counts / (counts.sum() + 1e-12)
        H = -np.sum(p * np.log(p + 1e-12))
        marg_counts.append(counts)
        marg_entropies.append(H)

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            # joint histogram
            joint_counts, _, _ = np.histogram2d(
                binned[i], binned[j],
                bins=(n_bins, n_bins)
            )
            pxy = joint_counts / (joint_counts.sum() + 1e-12)
            px = pxy.sum(axis=1, keepdims=True)
            py = pxy.sum(axis=0, keepdims=True)

            # MI = sum pxy * log(pxy / (px py))
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = pxy / (px * py + 1e-12)
                log_frac = np.log(frac + 1e-12)
                mi_ij = np.nansum(pxy * log_frac)

            mi[i, j] = mi_ij
            mi[j, i] = mi_ij

    return mi

def granger_pairwise(X_window, maxlag=5):
    # returns matrix of F-statistic from i->j (higher means stronger causal influence)
    # Using statsmodels grangercausalitytests; requires shape (n_samples,)
    n = X_window.shape[0]
    F = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                F[i,j] = 0
                continue
            try:
                data = np.vstack([X_window[j], X_window[i]]).T  # predict j using past of i & j
                # grangercausalitytests expects array with second column being caused variable
                res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                # pick lag with smallest pvalue or use F-stat for lag=1
                lag = 1
                Fstat = res[lag][0]['ssr_ftest'][0]  # F-stat
                F[i,j] = Fstat
            except Exception:
                F[i,j] = np.nan
    return F

def transfer_entropy_pairwise(X_window, nbins=8):
    """
    Simple discrete TE estimator with NaN-robust guards.
    If data are too degenerate (all-NaN / constant), returns an all-zero TE matrix.
    """
    X_window = np.asarray(X_window, dtype=float)
    n = X_window.shape[0]
    T = X_window.shape[1]
    TE = np.zeros((n, n), dtype=float)

    # If everything is NaN, just return zeros
    if not np.isfinite(X_window).any():
        return TE

    # Build per-channel bin edges using finite values only
    edges = []
    for i in range(n):
        xi = X_window[i]
        finite_xi = xi[np.isfinite(xi)]
        if finite_xi.size == 0:
            # no finite data for this channel -> dummy edges
            edges.append(np.linspace(0.0, 1.0, nbins + 1))
        else:
            # If constant, slightly widen the range so histogram is valid
            if np.allclose(finite_xi, finite_xi[0]):
                lo = finite_xi[0] - 0.5
                hi = finite_xi[0] + 0.5
                edges.append(np.linspace(lo, hi, nbins + 1))
            else:
                try:
                    edges.append(np.histogram_bin_edges(finite_xi, bins=nbins))
                except Exception:
                    lo = finite_xi.min()
                    hi = finite_xi.max() + 1e-6
                    edges.append(np.linspace(lo, hi, nbins + 1))

    # Fill NaNs in the actual series so digitize works
    X_filled = X_window.copy()
    for i in range(n):
        xi = X_filled[i]
        if np.isnan(xi).any():
            m = np.nanmean(xi)
            if not np.isfinite(m):
                m = 0.0
            xi[np.isnan(xi)] = m
            X_filled[i] = xi

    # Discretize
    binned = np.empty((n, T), dtype=int)
    for i in range(n):
        binned[i] = np.digitize(X_filled[i], edges[i])

    for i in range(n):
        for j in range(n):
            if i == j:
                TE[i, j] = 0.0
                continue

            xi = binned[i, :-1]
            xj = binned[j, :-1]
            xj_next = binned[j, 1:]

            # Build joint counts for (xj_next, xj, xi)
            unique = {}
            total = 0
            for a, b_, c in zip(xj_next, xj, xi):
                key = (a, b_, c)
                unique[key] = unique.get(key, 0) + 1
                total += 1

            if total == 0:
                TE[i, j] = 0.0
                continue

            denom_jx = {}
            denom_j = {}
            for a, b_, c in zip(xj_next, xj, xi):
                denom_jx[(b_, c)] = denom_jx.get((b_, c), 0) + 1
                denom_j[b_] = denom_j.get(b_, 0) + 1

            te_val = 0.0
            for (a, b_, c), count in unique.items():
                p_xyz = count / total
                p_xj_given_jx = count / denom_jx[(b_, c)]
                count_ab = sum(v for (aa, bb, cc), v in unique.items()
                               if aa == a and bb == b_)
                p_xj_given_j = count_ab / denom_j[b_]

                if p_xj_given_jx > 0 and p_xj_given_j > 0:
                    te_val += p_xyz * math.log(
                        (p_xj_given_jx / p_xj_given_j) + 1e-12
                    )

            TE[i, j] = te_val

    TE = np.nan_to_num(TE, nan=0.0, posinf=0.0, neginf=0.0)
    return TE


# -------------- Graph / Network features --------------
def graph_features_from_adjacency(A):
    """
    A: adjacency matrix (n,n), undirected (symmetric)
    returns dict of graph-level statistics and node-level stats aggregated (mean, max)
    """
    # Replace NaNs with 0 before creating the graph
    A = np.nan_to_num(A, nan=0.0)
    G = nx.from_numpy_array(A)
    # ensure weights non-neg
    for u,v,data in G.edges(data=True):
        if data.get('weight', None) is None:
            data['weight'] = A[u][v]
    deg = np.array([d for n,d in G.degree(weight='weight')])
    clustering = np.array(list(nx.clustering(G, weight='weight').values()))
    try:
        # Ensure input to eigenvector_centrality_numpy is float
        eig_c = np.array(list(nx.eigenvector_centrality_numpy(G, weight='weight').values()))
    except Exception:
        warnings.warn("Eigenvector centrality calculation failed. Using mean of absolute eigenvalues as fallback.")
        # Fallback: Use absolute eigenvalues if eigenvector centrality fails
        try:
            evals = np.linalg.eigvals(A)
            eig_c = np.abs(evals).real
            # Ensure eig_c has the correct length
            if len(eig_c) != len(A):
                 eig_c = np.ones(len(A)) * np.mean(eig_c) if len(eig_c) > 0 else np.zeros(len(A))
        except Exception:
            warnings.warn("Absolute eigenvalues calculation failed. Using zeros for eigenvector centrality fallback.")
            eig_c = np.zeros(len(A))


    betweenness = np.array(list(nx.betweenness_centrality(G, weight='weight').values()))
    # graph energy: sum absolute eigenvalues of adjacency
    try:
      evals = np.linalg.eigvals(A)
      ge = np.sum(np.abs(evals))
      pk = np.abs(evals) / (np.sum(np.abs(evals))+1e-12)
      graph_entropy = -np.sum(pk * np.log(pk + 1e-12))
    except Exception:
      warnings.warn("Graph energy or entropy calculation failed. Setting to NaN.")
      ge = np.nan
      graph_entropy = np.nan


    feats = {
        'deg_mean': np.nanmean(deg), 'deg_max': np.nanmax(deg),
        'clust_mean': np.nanmean(clustering), 'clust_max': np.nanmax(clustering),
        'eigc_mean': np.nanmean(eig_c), 'eigc_max': np.nanmax(eig_c),
        'btw_mean': np.nanmean(betweenness), 'btw_max': np.nanmax(betweenness),
        'graph_energy': ge, 'graph_entropy': graph_entropy
    }
    return feats

# ------------------------------------------------------------------
# Temporal-lobe channel selection + game-theory & directional metrics
# ------------------------------------------------------------------

# Canonical temporal-dominant TCP channels (10-20 style)
TEMPORAL_TCP_CHANNELS = [
    'F7-T3', 'T3-T5', 'T5-O1',   # left temporal chain
    'F8-T4', 'T4-T6', 'T6-O2',   # right temporal chain
    'A1-T3', 'T4-A2'             # anterior temporal to ears
]

def get_temporal_indices(channel_names, use_core=True):
    """
    Robust temporal-channel selector for TCP montages.

    Parameters
    ----------
    channel_names : list of str
        TCP bipolar channel names, e.g. ['FP1-F7', 'F7-T3', ...].
    use_core : bool
        For now core==full set, but you can widen later.

    Returns
    -------
    idx : np.ndarray of int
        Indices of temporal channels present in channel_names.
        Empty array if none.
    """
    if channel_names is None:
        return np.array([], dtype=int)

    # normalize incoming names
    norm_in = [ch.upper().strip() for ch in channel_names]

    # for now core == full set; you can define TEMPORAL_TCP_CORE later if you like
    temporal_set = TEMPORAL_TCP_CHANNELS
    temporal_norm = {ch.upper().strip() for ch in temporal_set}

    idxs = [i for i, nm in enumerate(norm_in) if nm in temporal_norm]
    return np.array(idxs, dtype=int)


# Canonical left/right temporal TCP chains for region pooling
LEFT_TEMPORAL_TCP = {
    'F7-T3', 'T3-T5', 'T5-O1', 'A1-T3'
}
RIGHT_TEMPORAL_TCP = {
    'F8-T4', 'T4-T6', 'T6-O2', 'T4-A2'
}

def temporal_lr_summary(adj_temp, temporal_idx, channel_names, prefix):
    """
    Region-level temporal GT features:
      - payoff share in LT vs RT
      - LT vs RT asymmetry
      - leader counts per side

    adj_temp:  (n_temp, n_temp) coherence (or other) adjacency for temporal subset
    temporal_idx: indices into channel_names that correspond to temporal channels
    channel_names: full TCP channel list
    prefix: e.g. "delta_temp_"
    """
    feats = {}

    # Build mapping from row index in adj_temp -> (LT / RT / other)
    n_temp = adj_temp.shape[0]
    side = np.zeros(n_temp, dtype=int)  # 0=other, 1=LT, 2=RT

    for local_i, global_i in enumerate(temporal_idx):
        nm = channel_names[global_i].upper().strip()
        if nm in LEFT_TEMPORAL_TCP:
            side[local_i] = 1
        elif nm in RIGHT_TEMPORAL_TCP:
            side[local_i] = 2

    A = np.nan_to_num(adj_temp, nan=0.0)
    np.fill_diagonal(A, 0.0)
    payoff = A.sum(axis=1)  # node strength

    total = payoff.sum() + 1e-12
    if total <= 0:
        feats[prefix + "LT_payoff_share"] = 0.0
        feats[prefix + "RT_payoff_share"] = 0.0
        feats[prefix + "RT_minus_LT_payoff"] = 0.0
        feats[prefix + "LT_leader_count"] = 0.0
        feats[prefix + "RT_leader_count"] = 0.0
        feats[prefix + "temp_leader_total"] = 0.0
        return feats

    lt_mask = (side == 1)
    rt_mask = (side == 2)

    lt_payoff = payoff[lt_mask].sum()
    rt_payoff = payoff[rt_mask].sum()

    feats[prefix + "LT_payoff_share"] = float(lt_payoff / total)
    feats[prefix + "RT_payoff_share"] = float(rt_payoff / total)
    feats[prefix + "RT_minus_LT_payoff"] = float(rt_payoff - lt_payoff)

    # Leaders = payoff > mean(payoff)
    mean_p = payoff.mean()
    leaders = payoff > mean_p
    feats[prefix + "temp_leader_total"] = float(leaders.sum())
    feats[prefix + "LT_leader_count"] = float((leaders & lt_mask).sum())
    feats[prefix + "RT_leader_count"] = float((leaders & rt_mask).sum())

    return feats


def compute_payoff_vector(adj):
    """
    Simple payoff vector from an adjacency matrix: node strength (sum of weights).
    adj: (n, n) symmetric, non-negative.
    """
    A = np.nan_to_num(adj, nan=0.0)
    # ignore self-loops
    np.fill_diagonal(A, 0.0)
    return A.sum(axis=1)


def gini_coefficient(x):
    """
    Gini coefficient of a non-negative vector x.
    0 => perfectly equal; 1 => maximally unequal.
    """
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0.0):
        return 0.0

    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    # classic discrete Gini formula
    gini = (2.0 * np.sum((np.arange(1, n + 1) * x_sorted)) /
            (n * cum[-1])) - (n + 1) / n
    return float(gini)


def game_theory_summary(adj, prefix, top_k=3):
    """
    Compute game-theoretic summary metrics from an adjacency matrix:
      - payoff vector (node strength)
      - payoff entropy
      - Gini of payoff
      - top-K coalition share
      - leader count (payoff > mean)
      - max share

    Returns a dict {feature_name: value}, with names prefixed by `prefix`.
    """
    feats = {}
    payoff = compute_payoff_vector(adj)  # (n_nodes,)
    total = payoff.sum() + 1e-12
    if total <= 0:
        # All-zero graph: return zeros
        feats[prefix + "payoff_entropy"] = 0.0
        feats[prefix + "gini"] = 0.0
        feats[prefix + f"top{top_k}_coalition_share"] = 0.0
        feats[prefix + "leader_count"] = 0.0
        feats[prefix + "max_share"] = 0.0
        feats[prefix + "payoff_mean"] = 0.0
        feats[prefix + "payoff_max"] = 0.0
        return feats

    p = payoff / total
    # payoff entropy: how spread the influence is
    feats[prefix + "payoff_entropy"] = float(
        -np.sum(p * np.log(p + 1e-12))
    )
    # inequality of influence
    feats[prefix + "gini"] = gini_coefficient(payoff)

    payoff_sorted = np.sort(payoff)[::-1]
    k = min(top_k, payoff_sorted.size)
    feats[prefix + f"top{top_k}_coalition_share"] = float(
        payoff_sorted[:k].sum() / total
    )
    # leaders: nodes with payoff above mean
    feats[prefix + "leader_count"] = float(
        np.sum(payoff > payoff.mean())
    )
    feats[prefix + "max_share"] = float(
        payoff_sorted[0] / total
    )
    # convenience stats
    feats[prefix + "payoff_mean"] = float(payoff.mean())
    feats[prefix + "payoff_max"] = float(payoff_sorted[0])
    return feats


def directional_connectivity_features(signals, sfreq, band_name, prefix,
                                      maxlag=3, nbins=8, target_len=400):
    """
    Lightweight directional features on a small set of channels.
    """
    # Only compute for selected bands
    if band_name not in DIRECTIONAL_BANDS:
        return {}

    sig = np.asarray(signals, dtype=float)
    n_ch, n_samp = sig.shape

    # If everything is NaN or only one channel -> nothing directional
    if n_ch < 2 or not np.isfinite(sig).any():
        return {
            f"{prefix}{band_name}_gc_mean": 0.0,
            f"{prefix}{band_name}_gc_max": 0.0,
            f"{prefix}{band_name}_te_mean": 0.0,
            f"{prefix}{band_name}_te_max": 0.0,
            f"{prefix}{band_name}_pdc_like_entropy": 0.0,
            f"{prefix}{band_name}_pdc_like_gini": 0.0,
        }

    # downsample ...
    if n_samp > target_len:
        factor = int(np.floor(n_samp / target_len))
        sig_ds = sig[:, ::factor]
    else:
        sig_ds = sig

    # Granger & TE wrapped in try/except as extra guard
    feats = {}

    try:
        gc_F = granger_pairwise(sig_ds, maxlag=maxlag)
        gc_F = np.nan_to_num(gc_F, nan=0.0)
    except Exception:
        gc_F = np.zeros((n_ch, n_ch))

    iu = np.triu_indices(n_ch, k=1)
    if iu[0].size > 0:
        feats[f"{prefix}{band_name}_gc_mean"] = safe_scalar(np.mean(gc_F[iu]), 0.0)
        feats[f"{prefix}{band_name}_gc_max"]  = safe_scalar(np.max(gc_F[iu]), 0.0)
    else:
        feats[f"{prefix}{band_name}_gc_mean"] = 0.0
        feats[f"{prefix}{band_name}_gc_max"]  = 0.0

    try:
        te = transfer_entropy_pairwise(sig_ds, nbins=nbins)
        te = np.nan_to_num(te, nan=0.0)
    except Exception:
        te = np.zeros((n_ch, n_ch))

    if iu[0].size > 0:
        feats[f"{prefix}{band_name}_te_mean"] = safe_scalar(np.mean(te[iu]), 0.0)
        feats[f"{prefix}{band_name}_te_max"]  = safe_scalar(np.max(te[iu]), 0.0)
    else:
        feats[f"{prefix}{band_name}_te_mean"] = 0.0
        feats[f"{prefix}{band_name}_te_max"]  = 0.0

    out_strength = np.maximum(gc_F.sum(axis=1), 0.0)
    total_out = out_strength.sum() + 1e-12
    if total_out > 0:
        p_out = out_strength / total_out
        feats[f"{prefix}{band_name}_pdc_like_entropy"] = safe_scalar(
            -np.sum(p_out * np.log(p_out + 1e-12)), 0.0
        )
        feats[f"{prefix}{band_name}_pdc_like_gini"] = gini_coefficient(out_strength)
    else:
        feats[f"{prefix}{band_name}_pdc_like_entropy"] = 0.0
        feats[f"{prefix}{band_name}_pdc_like_gini"] = 0.0

    return feats

# %%
# -------------- Temporal evolution features across windows --------------
def temporal_features_over_windows(graph_energy_series, per_channel_power_series, window_seconds):
    """
    Inputs:
      - graph_energy_series: array (n_windows,)
      - per_channel_power_series: array (n_windows, n_channels) or (n_windows,) for aggregate
    Returns: dict temporal features per final window (e.g., delta energy, entropy rate etc.)
    """
    feats = {}
    if len(graph_energy_series) < 2:
        feats['delta_graph_energy'] = 0.0
    else:
        feats['delta_graph_energy'] = (graph_energy_series[-1] - graph_energy_series[-2]) / window_seconds
    # entropy rate approx: H(X_t | X_{t-1}) using histogram of differences
    if per_channel_power_series.ndim == 2:
        agg = np.mean(per_channel_power_series, axis=1)
    else:
        agg = per_channel_power_series
    diffs = np.diff(agg)
    # estimate conditional entropy as entropy of diffs
    if len(diffs) > 1:
        feats['entropy_rate'] = entropy(np.histogram(diffs, bins=16)[0] + 1e-12)
    else:
        feats['entropy_rate'] = 0.0
    # dominance index
    last = per_channel_power_series[-1]
    if last.ndim==0:
        feats['dominance_index'] = 1.0
    else:
        feats['dominance_index'] = np.max(last) / (np.sum(last)+1e-12)
    # pre-ictal stability index: 1 - std_lastN / mean_lastN
    N = min(5, len(agg))
    if N>=1:
        lastN = agg[-N:]
        feats['preictal_stability_index'] = 1 - (np.std(lastN) / (np.mean(lastN) + 1e-12))
    else:
        feats['preictal_stability_index'] = 0.0
    return feats

def load_meta_list(meta_array):
    """
    Robustly convert whatever is in NPZ['meta'] into a list of dicts.

    Handles:
      - numpy object array of dicts
      - numpy object array of 0-d arrays containing dicts (old .item() style)
      - plain Python list of dicts
      - single dict
    """
    # Case 1: plain list (already list of dicts)
    if isinstance(meta_array, list):
        return meta_array

    # Case 2: single dict
    if isinstance(meta_array, dict):
        return [meta_array]

    # Case 3: numpy array (most common when loaded from npz)
    if isinstance(meta_array, np.ndarray):
        # empty array
        if meta_array.size == 0:
            return []

        first = meta_array.flatten()[0]

        # 3a: array of dicts
        if isinstance(first, dict):
            return [m for m in meta_array]

        # 3b: array of 0-d object arrays containing dicts
        if hasattr(first, "item"):
            return [m.item() for m in meta_array]

    # If we get here, it's some unexpected format
    raise TypeError(f"Unsupported meta format: type={type(meta_array)}")

# -------------- Main feature extraction pipeline for a single window --------------
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Only these bands will get GC/TE/PDC-like directional features
DIRECTIONAL_BANDS = {"delta", "theta"}

def extract_window_features(X_window, sfreq, channel_names=None,
                            compute_pairwise=True, compute_graph=True,
                            compute_directional=False):
    """
    X_window: (n_channels, n_samples) cleaned, ideally TCP montage
    Returns: feature_vector (1D), feature_names (list)
    """
    X_window = np.asarray(X_window, dtype=float)
    n_ch = X_window.shape[0]
    feats = {}

    # Determine temporal-lobe-dominant channel indices (for GT & directional features)
    temporal_idx = None
    if channel_names is not None:
        temporal_idx = get_temporal_indices(channel_names)

    # ---- Precompute band-passed data for all bands once (vectorized over channels)
    band_data = {}
    for bname, band in BANDS.items():
        band_data[bname] = bandpass(X_window, sfreq, band[0], band[1])  # (n_ch, n_samples)

    # ------------------------------------------------------------------
    # 1) Broadband nonlinear features (do this ONCE per channel)
    # ------------------------------------------------------------------
    for ch in range(n_ch):
        base = f"broad_ch{ch}_"
        x = X_window[ch]
        feats[base + 'approx_entropy'] = approximate_entropy(x)
        feats[base + 'higuchi'] = higuchi_fd(x)

    # ------------------------------------------------------------------
    # 2) Per-channel, per-band features
    # ------------------------------------------------------------------
    for bname, band in BANDS.items():
        signals = band_data[bname]  # (n_ch, n_samples)

        for ch in range(n_ch):
            base = f"{bname}_ch{ch}_"
            x = signals[ch]

            # morphological
            feats[base + 'pp'] = peak_to_peak(x)
            feats[base + 'rms'] = rms(x)
            feats[base + 'zcr'] = zero_crossing_rate(x)
            feats[base + 'linelen'] = line_length(x)
            feats[base + 'skew'] = skew(x)
            feats[base + 'kurt'] = kurtosis(x)
            feats[base + 'sharpness'] = sharpness_index(x, sfreq)
            feats[base + 'asym'] = asymmetry_ratio(x, sfreq)

            # wavelet energy (still per-band)
            try:
                we = wavelet_energy(x)
                if isinstance(we, np.ndarray):
                    feats[base + 'wavelet_e_sum'] = np.sum(we)
                else:
                    feats[base + 'wavelet_e_sum'] = float(we)
            except Exception:
                feats[base + 'wavelet_e_sum'] = np.sum(x ** 2)

            # local dynamical on band-limited signal
            feats[base + 'bandpower'] = band_power(x, sfreq, band)
            feats[base + 'spec_entropy'] = spectral_entropy(x, sfreq)
            feats[base + 'neo'] = nonlinear_energy_operator(x)

    # ------------------------------------------------------------------
    # 3) Pairwise features aggregated by band
    # ------------------------------------------------------------------
    if compute_pairwise:
        for bname, band in BANDS.items():
            signals = band_data[bname]  # already band-passed: (n_ch, n_samp)

            # PLV
            plv = pairwise_plv(signals)
            plv = np.nan_to_num(plv, nan=np.nan)  # leave NaNs as NaN inside, we'll guard on reduction

            # coherence
            coh = pairwise_coherence(signals, sfreq)
            coh = np.nan_to_num(coh, nan=np.nan)

            # Pearson correlation
            corr = np.corrcoef(signals)
            corr = np.nan_to_num(corr, nan=np.nan)

            iu = np.triu_indices(n_ch, k=1)

            # Helper: if slice is all-NaN, treat summary as 0 instead of crashing / warning
            def _safe_mean(arr):
                sub = arr[iu]
                if not np.isfinite(sub).any():
                    return 0.0
                return safe_scalar(np.nanmean(sub), default=0.0)

            def _safe_max(arr):
                sub = arr[iu]
                if not np.isfinite(sub).any():
                    return 0.0
                return safe_scalar(np.nanmax(sub), default=0.0)

            feats[f"{bname}_plv_mean"]  = _safe_mean(plv)
            feats[f"{bname}_plv_max"]   = _safe_max(plv)
            feats[f"{bname}_coh_mean"]  = _safe_mean(coh)
            feats[f"{bname}_coh_max"]   = _safe_max(coh)
            feats[f"{bname}_corr_mean"] = _safe_mean(corr)
            feats[f"{bname}_corr_max"]  = _safe_max(corr)

            # --- Temporal-lobe restricted game-theory features per band ---
            if temporal_idx is not None and temporal_idx.size >= 3:
                temp_signals = signals[temporal_idx, :]  # (n_temp, n_samp)

                # adjacency from coherence on temporal-only subset
                adj_temp = pairwise_coherence(temp_signals, sfreq)
                if adj_temp is None:
                    adj_temp = np.zeros((temp_signals.shape[0], temp_signals.shape[0]))
                adj_temp = np.nan_to_num(adj_temp, nan=0.0)

                # Game-theoretic summaries (always per-band)
                gt_feats = game_theory_summary(
                    adj_temp,
                    prefix=f"{bname}_temp_"
                )
                # scrub any NaNs coming from GT
                for k, v in gt_feats.items():
                    gt_feats[k] = safe_scalar(v, default=0.0)
                feats.update(gt_feats)

                # Directional GC/TE/PDC-like: only for low-frequency bands
                if compute_directional and bname in ("delta", "theta"):
                    try:
                        dir_feats = directional_connectivity_features(
                            temp_signals,
                            sfreq,
                            band_name=bname,
                            prefix=f"{bname}_temp_"
                        )
                        for k, v in dir_feats.items():
                            feats[k] = safe_scalar(v, 0.0)

                    except Exception:
                        # Clean fallback: zeros for this band
                        feats[f"{bname}_temp_gc_mean"] = 0.0
                        feats[f"{bname}_temp_gc_max"] = 0.0
                        feats[f"{bname}_temp_te_mean"] = 0.0
                        feats[f"{bname}_temp_te_max"] = 0.0
                        feats[f"{bname}_temp_pdc_like_entropy"] = 0.0
                        feats[f"{bname}_temp_pdc_like_gini"] = 0.0

    # ------------------------------------------------------------------
    # 4) Graph features: adjacency from alpha-band coherence
    # ------------------------------------------------------------------
    if compute_graph:
        alpha_signals = band_data['alpha']  # (n_ch, n_samples)
        adj = pairwise_coherence(alpha_signals, sfreq)
        gf = graph_features_from_adjacency(adj)
        feats.update({f"graph_{k}": v for k, v in gf.items()})

    # ---- Flatten to vector
    feature_names = list(feats.keys())
    feature_vector = np.array([feats[k] for k in feature_names], dtype=float)
    return feature_vector, feature_names


# -------------- Batch extraction for windows array --------------
def extract_features_from_windows(
    windows,
    sfreq,
    channel_names,
    is_le=False,
    compute_pairwise=True,
    compute_graph=True,
    compute_directional=False,
    n_jobs=-1,
):
    """
    windows: (n_windows, n_ch, n_samples)
    Returns:
      X: (n_windows, n_features)
      feature_names: list[str]
    """
    # Convert LE to TCP if needed
    if is_le:
        tcp_windows, tcp_names = le_to_tcp_array(windows, channel_names, sfreq)
    else:
        tcp_windows = windows
        tcp_names = channel_names

    n_windows = tcp_windows.shape[0]

    def _feat_one_window(win):
        v, fnames = extract_window_features(
            win,
            sfreq,
            channel_names=tcp_names,
            compute_pairwise=compute_pairwise,
            compute_graph=compute_graph,
            compute_directional=compute_directional,
        )
        return v, fnames

    # Parallel loop over windows
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_feat_one_window)(tcp_windows[i]) for i in range(n_windows)
    )

    feat_list = [r[0] for r in results]
    fnames = results[0][1]  # same for all windows
    X = np.vstack(feat_list)
    
    # -------------------------------------------------------------------------
    # NaN / Inf guard AFTER feature extraction
    # -------------------------------------------------------------------------
    if np.isnan(X).any() or np.isinf(X).any():
        print("[WARN] NaNs or infs found in extracted features — replacing with zero.")
        col_nan = np.isnan(X).any(axis=0) | np.isinf(X).any(axis=0)
        print("NaN/inf columns:", np.where(col_nan)[0])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # -------------------------------------------------------------------------

    return X, fnames


def subjectwise_normalize_gt_features(X, feature_names, meta_list, rec_idx):
    """
    X: [N_windows, N_features]
    feature_names: list of length N_features
    meta_list: list of per-recording dicts (each has 'subject_id')
    rec_idx: [N_windows] -> index into meta_list

    We:
      1) identify which columns correspond to GT/connectivity summaries
      2) for each subject, z-score those columns using that subject's own windows
    """
    X = np.asarray(X, dtype=float)
    N, F = X.shape
    if N == 0 or F == 0:
        return X

    # Which feature names are "GT-like"? You can adjust this pattern later.
    gt_idx = []
    for j, name in enumerate(feature_names):
        if (
            "_temp_" in name                # temporal-chain GT features
            or name.startswith("graph_")    # graph_*
            or "payoff" in name             # payoff_entropy, payoff_mean, etc.
            or "gini" in name               # gini
            or "coalition" in name          # topK_coalition_share
            or "pdc_like" in name           # pdc-like entropy/gini
            or name.endswith("_gc_mean")
            or name.endswith("_gc_max")
            or name.endswith("_te_mean")
            or name.endswith("_te_max")
        ):
            gt_idx.append(j)

    if not gt_idx:
        # nothing GT-like detected
        return X

    gt_idx = np.array(gt_idx, dtype=int)

    # subject_id per recording
    subj_ids_per_recording = [
        m.get("subject_id", f"subj_{i}") for i, m in enumerate(meta_list)
    ]

    # subject_id per window
    subj_ids = np.array([subj_ids_per_recording[i] for i in rec_idx])

    X_norm = X.copy()

    for sid in np.unique(subj_ids):
        mask = (subj_ids == sid)
        if not np.any(mask):
            continue

        block = X_norm[mask][:, gt_idx]  # [N_subj_windows, N_gt_features]
        mu = block.mean(axis=0)
        sigma = block.std(axis=0)
        sigma[sigma < 1e-6] = 1e-6  # avoid divide-by-zero

        X_norm[mask][:, gt_idx] = (block - mu) / sigma

    return X_norm


# %%
# -------------- Feature selection (Shapley) --------------
def shap_feature_selection(X, y, top_k=100, model=None,
                           random_state=0,
                           sample_for_shap=1000):
    """
    Compute feature importance via SHAP, return:
        top_idx: indices of top_k most important features
        sv:      1D array [n_features] with mean |SHAP| per feature

    - If model is None, trains a RandomForest on (X, y).
    - If model is tree-based, uses TreeExplainer (fast).
    - Otherwise, falls back to KernelExplainer (slow).
    """
    np.random.seed(random_state)

    # 1) Prepare / fit model
    if model is None:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

    # Always fit on the X, y passed in
    model.fit(X, y)

    n_features = X.shape[1]

    def _collapse_to_feature_importance(shap_vals_array):
        """
        shap_vals_array: np.ndarray with shape containing n_features somewhere.
        Returns: 1D array of length n_features with mean |SHAP| per feature.
        """
        sv_arr = np.array(shap_vals_array)
        axes_sizes = sv_arr.shape

        # find which axis is the feature axis (size == n_features)
        candidate_axes = [ax for ax, s in enumerate(axes_sizes) if s == n_features]
        if not candidate_axes:
            raise RuntimeError(
                f"Could not find feature axis of size {n_features} in SHAP array shape {axes_sizes}"
            )
        feat_axis = candidate_axes[0]

        abs_sv = np.abs(sv_arr)
        axes_to_mean = tuple(ax for ax in range(abs_sv.ndim) if ax != feat_axis)
        if axes_to_mean:
            sv = abs_sv.mean(axis=axes_to_mean)
        else:
            sv = abs_sv  # already 1D

        sv = np.asarray(sv).reshape(n_features)
        return sv

    # 2) Try TreeExplainer when appropriate, otherwise KernelExplainer
    sv = None

    # Heuristic: tree-based models usually have 'estimators_' or 'get_booster'
    is_tree_like = (
        hasattr(model, "estimators_") or
        hasattr(model, "get_booster")
    )

    if is_tree_like:
        try:
            explainer = shap.TreeExplainer(model)
            idx = np.random.choice(
                np.arange(X.shape[0]),
                size=min(sample_for_shap, X.shape[0]),
                replace=False,
            )
            shap_vals = explainer.shap_values(X[idx])

            if isinstance(shap_vals, list):
                shap_arr = shap_vals[1]  # class 1 for binary
            else:
                shap_arr = shap_vals

            sv = _collapse_to_feature_importance(shap_arr)
        except Exception:
            warnings.warn("TreeExplainer failed. Falling back to KernelExplainer (slow).")
            sv = None

    if sv is None:
        # Generic path (SVM, pipelines, or when TreeExplainer failed)
        bg = shap.sample(X, min(100, X.shape[0]))
        explainer = shap.KernelExplainer(model.predict_proba, bg)

        idx = np.random.choice(
            np.arange(X.shape[0]),
            size=min(sample_for_shap, X.shape[0]),
            replace=False,
        )
        shap_vals = explainer.shap_values(X[idx], nsamples=200)

        if isinstance(shap_vals, list):
            shap_arr = shap_vals[1]  # class 1
        else:
            shap_arr = shap_vals

        sv = _collapse_to_feature_importance(shap_arr)

    # 3) Extract top-k features
    top_idx = np.argsort(sv)[::-1][:top_k]
    return top_idx, sv

def summarize_shap_mass(sv, ks=(20, 50, 100, 200, 400)):
    """
    sv: 1D array of SHAP importance magnitudes per feature (length F).
    ks: tuple of K values at which to report cumulative mass.

    Prints, e.g.:
      K= 20 -> 23.4%
      K=100 -> 52.1%
      ...
    """
    sv = np.asarray(sv, dtype=float)
    sv = np.abs(sv)
    total = sv.sum()
    if total <= 0:
        print("SHAP mass is zero or negative – nothing to summarize.")
        return

    sv_sorted = np.sort(sv)[::-1]  # descending
    cum = np.cumsum(sv_sorted) / total

    max_k = len(sv_sorted)
    print("\n=== SHAP mass summary ===")
    for k in ks:
        if k > max_k:
            continue
        frac = cum[k-1] * 100.0
        print(f"Top-{k:4d} features explain {frac:5.2f}% of total SHAP mass.")

def plot_shap_cumulative(sv, max_k=None, title="Cumulative SHAP mass"):
    sv = np.asarray(sv, dtype=float)
    sv = np.abs(sv)
    total = sv.sum()
    if total <= 0:
        print("SHAP mass is zero or negative – nothing to plot.")
        return

    sv_sorted = np.sort(sv)[::-1]
    cum = np.cumsum(sv_sorted) / total
    F = len(sv_sorted)
    if max_k is None or max_k > F:
        max_k = F

    xs = np.arange(1, max_k+1)
    ys = cum[:max_k]

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='.')
    plt.axhline(0.5, linestyle='--')   # 50% reference
    plt.axhline(0.6, linestyle='--')   # 60% reference
    plt.xlabel("Number of top features (K)")
    plt.ylabel("Cumulative SHAP mass")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_shap_cumulative_multi(shap_dict, max_k=300, title="Cumulative SHAP mass comparison"):
    """
    shap_dict: dict like {"RF": shap_vals_rf, "SVM": shap_vals_svm, "XGB": shap_vals_xgb}
    max_k: maximum K to show on the x-axis

    Plots cumulative SHAP mass vs K for each model on the same figure.
    """
    plt.figure(figsize=(7, 5))

    for label, sv in shap_dict.items():
        sv = np.asarray(sv, dtype=float)
        sv = np.abs(sv)
        total = sv.sum()
        if total <= 0:
            continue

        sv_sorted = np.sort(sv)[::-1]
        cum = np.cumsum(sv_sorted) / total

        K = min(max_k, len(cum))
        xs = np.arange(1, K + 1)
        ys = cum[:K]

        plt.plot(xs, ys, marker='.', linewidth=1.5, label=label)

    # reference lines
    plt.axhline(0.5, linestyle='--', alpha=0.5)
    plt.axhline(0.7, linestyle='--', alpha=0.5)
    plt.axhline(0.8, linestyle='--', alpha=0.5)

    plt.xlabel("Top-K features")
    plt.ylabel("Cumulative SHAP mass")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_shap_mass_summary(name, sv, top_ks=(20, 50, 100, 150, 200, 300)):
    abs_sv = np.abs(sv)
    total = abs_sv.sum()

    print(f"\n[{name}] SHAP feature selection ...")
    if total <= 0:
        print("SHAP mass is zero or negative – nothing to summarize.")
        return

    sorted_abs = np.sort(abs_sv)[::-1]

    print("\n=== SHAP mass summary ===")
    for K in top_ks:
        K_eff = min(K, len(sorted_abs))
        frac = 100.0 * sorted_abs[:K_eff].sum() / total
        print(f"Top-{K:3d} features explain {frac:.2f}% of total SHAP mass.")

def load_features_with_meta(base_out_dir, split_name):
    """
    Loads normalized features (X), labels (y), and feature names (fnames)
    from features.npz for the given split.

    If only raw features are present, re-applies subjectwise GT normalization.
    Also enforces NaN/Inf cleanup before returning.
    """
    base_out_dir = Path(base_out_dir)
    feat_npz = base_out_dir / split_name / "features.npz"
    print(f"[FE] Loading features from: {feat_npz.as_posix()}")

    data = np.load(feat_npz, allow_pickle=True)

    # --- 1) Get feature matrix X (normalized if present, else recompute) ---
    if "X" in data.files:
        # already normalized
        X = data["X"]
    else:
        # fallback: raw features -> re-normalize GT subjectwise
        X_raw = data["X_raw"]
        fnames = list(data["fnames"])
        meta_array = data["meta"]
        rec_idx = data["rec_idx"]

        # make sure meta_list is a proper list of dicts
        meta_list = load_meta_list(meta_array)

        X = subjectwise_normalize_gt_features(
            X_raw,
            fnames,
            meta_list,
            rec_idx,
        )

    y = data["y"]
    fnames = list(data["fnames"])

    # --- 2) Global NaN / Inf guard on loaded features ---
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any() or np.isinf(X).any():
        print(f"[FE] WARN: NaNs or Infs found in loaded features for split='{split_name}'. "
              "Replacing with zero.")
        col_bad = np.isnan(X).any(axis=0) | np.isinf(X).any(axis=0)
        bad_idx = np.where(col_bad)[0]
        print(f"[FE] Columns with NaN/Inf in split='{split_name}': {bad_idx}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, fnames


def evaluate_model_on_split(
    model,
    top_idx,
    base_out_dir,
    train_split_name="train",
    eval_split_name="dev",
    save_model_path=None,   # optional: path to save the fitted model + feature info
):
    """
    Train `model` on precomputed features from train_split_name and
    evaluate on eval_split_name.

    Assumes:
      - compute_and_save_features_for_split(...) has already been run
        for both splits and produced <split>/features.npz
      - top_idx is the array of selected feature indices (e.g., SHAP top-200
        computed on the TRAIN split features).

    Returns: dict with AUROC, AUPRC, best_F1, best_threshold.
    """
    base_out_dir = Path(base_out_dir).expanduser().resolve()

    # ---- Load precomputed features ----
    print(f"\n[Eval] Loading TRAIN features for split '{train_split_name}' ...")
    X_train, y_train, fnames_train = load_features_with_meta(base_out_dir, train_split_name)
    print(f"[Eval] TRAIN: X shape={X_train.shape}, y shape={y_train.shape}")

    print(f"[Eval] Loading EVAL features for split '{eval_split_name}' ...")
    X_eval, y_eval, fnames_eval = load_features_with_meta(base_out_dir, eval_split_name)
    print(f"[Eval] EVAL : X shape={X_eval.shape}, y shape={y_eval.shape}")

    # ---- Sanity: feature name ordering must match ----
    if fnames_train != fnames_eval:
        raise RuntimeError("Feature name mismatch between train & eval splits.")

    # ---- Select SHAP top-k features ----
    top_idx = np.asarray(top_idx, dtype=int)
    X_train_sel = X_train[:, top_idx]
    X_eval_sel  = X_eval[:,  top_idx]

    print(f"[Eval] Using {len(top_idx)} selected features.")

    # ---- NaN / inf guard on selected features ----
    for name, arr in (("TRAIN", X_train_sel), ("EVAL", X_eval_sel)):
        bad_mask = ~np.isfinite(arr)
        if bad_mask.any():
            bad_cols = np.where(~np.isfinite(arr).all(axis=0))[0]
            print(f"[Eval] WARN: {name} features contain NaNs/inf; fixing.")
            print(f"[Eval] {name} bad columns:", bad_cols)
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Fit and predict ----
    print("[Eval] Fitting model on TRAIN selected features ...")
    model.fit(X_train_sel, y_train)

    print("[Eval] Predicting on EVAL selected features ...")
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_eval_sel)[:, 1]
    else:
        scores = model.decision_function(X_eval_sel)

    # ---- Metrics ----
    auroc = roc_auc_score(y_eval, scores)
    auprc = average_precision_score(y_eval, scores)

    prec, rec, thr = precision_recall_curve(y_eval, scores)
    f1_vals = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.argmax(f1_vals))
    best_f1 = float(f1_vals[best_idx])
    best_thresh = float(thr[best_idx - 1]) if best_idx > 0 and best_idx - 1 < len(thr) else 0.5

    print("\n[Eval] === Metrics on EVAL split ===")
    print(f"AUROC : {auroc:.4f}")
    print(f"AUPRC : {auprc:.4f}")
    print(f"Best F1: {best_f1:.4f} at threshold ~ {best_thresh:.3f}")

    # ---- Optional: save trained model ----
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "top_idx": top_idx,
            "feature_names": np.array(fnames_train)[top_idx],
            "train_split": train_split_name,
            "eval_split": eval_split_name,
        }
        joblib.dump(payload, save_model_path.as_posix())
        print(f"[Eval] Saved model + feature subset to: {save_model_path.as_posix()}")

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "best_F1": best_f1,
        "best_threshold": best_thresh,
    }

def load_meta_list(meta_array):
    """
    Safely convert meta_array from npz into a list of dicts.
    Handles cases:
      - already a list of dicts
      - numpy object array of dicts (needs .item())
      - numpy array of dicts without .item()
    """
    # Case 1: already a python list
    if isinstance(meta_array, list):
        return meta_array

    # Case 2: numpy array of Python dicts
    if isinstance(meta_array, np.ndarray):
        meta_list = []
        for m in meta_array:
            # If element is already a dict -> keep as is
            if isinstance(m, dict):
                meta_list.append(m)
            else:
                # If element is a numpy object wrapper, try .item()
                try:
                    meta_list.append(m.item())
                except (AttributeError, ValueError):
                    # Last fallback: assume m itself is the dict
                    meta_list.append(m)
        return meta_list

    # Case 3: unexpected type → wrap in list
    return [meta_array]


def compute_and_save_features_for_split(base_out_dir, split_name):
    """
    Loads split_windows.npz for given split, extracts features,
    applies subject-wise GT normalization, and saves to features.npz.
    """
    base_out_dir = Path(base_out_dir)
    split_npz = base_out_dir / split_name / "split_windows.npz"
    print(f"\n[FE] Loading split windows from: {split_npz.as_posix()}")

    data = np.load(split_npz, allow_pickle=True)
    windows_le = data["X"]             # (N_win, n_ch, n_samp) in LE montage
    y = data["y"]                      # labels
    ch_names_le = list(data["ch_names"])
    meta_array = data["meta"]
    rec_idx = data["rec_idx"]

    meta_list = load_meta_list(meta_array)

    try:
        sfreq = float(meta_list[0].get("fs", 250.0))
    except Exception:
        sfreq = 250.0
    print(f"[FE] Split '{split_name}': windows {windows_le.shape}, sfreq={sfreq}")

    # ---- Feature extraction (LE → TCP handled inside extract_features_from_windows via is_le=True) ----
    X_raw, fnames = extract_features_from_windows(
        windows_le,
        sfreq=sfreq,
        channel_names=ch_names_le,
        is_le=True,
        compute_pairwise=True,
        compute_graph=True,
        compute_directional=True,
        n_jobs=-1,
    )
    print(f"[FE] Raw feature matrix for '{split_name}': {X_raw.shape}")

    # ---- Subject-wise normalization for GT-like features only ----
    X_norm = subjectwise_normalize_gt_features(
        X_raw,
        fnames,
        meta_list,
        rec_idx,
    )
    print(f"[FE] After GT normalization '{split_name}': {X_norm.shape}")

    # ---- Save to features.npz ----
    out_npz = base_out_dir / split_name / "features.npz"
    np.savez_compressed(
        out_npz,
        X=X_norm,
        X_raw=X_raw,
        y=y,
        fnames=np.array(fnames, dtype=object),
        meta=meta_array,
        rec_idx=rec_idx,
    )
    print(f"[FE] Saved features to: {out_npz.as_posix()}")


def save_model_with_metadata(
    model,
    model_name,
    top_idx,
    feature_names,
    metrics,
    base_out_dir,
):
    """
    Save trained model + feature selection + metrics to:
      <base_out_dir>/models/<model_name>_model.pkl
    """
    base_out_dir = Path(base_out_dir).expanduser().resolve()
    model_dir = base_out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "model_name": model_name,
        "top_idx": np.array(top_idx, dtype=int),
        "feature_names": list(feature_names),
        "metrics": metrics,
    }

    out_path = model_dir / f"{model_name}_model.pkl"
    joblib.dump(payload, out_path)
    print(f"\n[Save] Saved {model_name} model + metadata to: {out_path.as_posix()}")
    return out_path


def plot_learning_curves_for_model(
    base_out_dir,
    model,
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    top_idx,
    n_points=5,
    random_state=0,
):
    """
    Plot simple learning curves (AUROC & AUPRC vs train fraction) for a given model.
    Uses *already-extracted* feature matrices (no FE rerun).

    - X_* are full feature matrices (all features)
    - top_idx: indices of selected features (e.g., SHAP top-200)
    """

    base_out_dir = Path(base_out_dir).expanduser().resolve()
    plot_dir = base_out_dir / "learning_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ensure numpy arrays
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_val   = np.asarray(X_val, dtype=float)
    y_val   = np.asarray(y_val)

    # restrict to selected features once
    X_val_sel = X_val[:, top_idx]

    # NaN/inf guard on VAL
    if np.isnan(X_val_sel).any() or np.isinf(X_val_sel).any():
        print(f"[LC][{model_name}] WARN: NaNs/inf in VAL features; replacing with zero")
        col_bad = np.isnan(X_val_sel).any(axis=0) | np.isinf(X_val_sel).any(axis=0)
        print(f"[LC][{model_name}] Bad VAL cols:", np.where(col_bad)[0])
        X_val_sel = np.nan_to_num(X_val_sel, nan=0.0, posinf=0.0, neginf=0.0)

    # train fractions to try
    train_fracs = np.linspace(0.1, 0.95, n_points)

    train_auroc, val_auroc = [], []
    train_auprc, val_auprc = [], []

    rng = np.random.RandomState(random_state)

    for frac in train_fracs:
        frac = float(frac)
        n_target = max(1, int(frac * len(y_train)))
        print(f"[LC][{model_name}] Training with ~{n_target} windows ({frac:.2f} of train)")

        # stratified subset of train
        X_tr_sub, _, y_tr_sub, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_target,
            stratify=y_train,
            random_state=int(rng.randint(0, 10_000)),
        )

        X_tr_sel = X_tr_sub[:, top_idx]

        # NaN/inf guard on TRAIN subset
        if np.isnan(X_tr_sel).any() or np.isinf(X_tr_sel).any():
            print(f"[LC][{model_name}] WARN: NaNs/inf in TRAIN features; replacing with zero")
            col_bad = np.isnan(X_tr_sel).any(axis=0) | np.isinf(X_tr_sel).any(axis=0)
            print(f"[LC][{model_name}] Bad TRAIN cols:", np.where(col_bad)[0])
            X_tr_sel = np.nan_to_num(X_tr_sel, nan=0.0, posinf=0.0, neginf=0.0)

        # fresh clone of the model each time
        m = clone(model)
        m.fit(X_tr_sel, y_tr_sub)

        # scores on train subset
        if hasattr(m, "predict_proba"):
            scores_tr = m.predict_proba(X_tr_sel)[:, 1]
            scores_val = m.predict_proba(X_val_sel)[:, 1]
        else:
            scores_tr = m.decision_function(X_tr_sel)
            scores_val = m.decision_function(X_val_sel)

        # train metrics
        try:
            auroc_tr = roc_auc_score(y_tr_sub, scores_tr)
        except ValueError:
            auroc_tr = np.nan
        try:
            auprc_tr = average_precision_score(y_tr_sub, scores_tr)
        except ValueError:
            auprc_tr = np.nan

        # val metrics
        try:
            auroc_v = roc_auc_score(y_val, scores_val)
        except ValueError:
            auroc_v = np.nan
        try:
            auprc_v = average_precision_score(y_val, scores_val)
        except ValueError:
            auprc_v = np.nan

        train_auroc.append(auroc_tr)
        val_auroc.append(auroc_v)
        train_auprc.append(auprc_tr)
        val_auprc.append(auprc_v)

    train_fracs = np.array(train_fracs)

    # --- Plot AUROC ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_fracs, train_auroc, marker="o", label="Train AUROC")
    plt.plot(train_fracs, val_auroc,   marker="o", label="Dev AUROC")
    plt.xlabel("Fraction of train windows used")
    plt.ylabel("AUROC")
    plt.title(f"Learning Curve (AUROC) - {model_name}")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    auroc_path = (plot_dir / f"learning_curve_auroc_{model_name}_top{len(top_idx)}.png").as_posix()
    plt.tight_layout()
    plt.savefig(auroc_path, dpi=200)
    plt.close()
    print(f"[LC][{model_name}] Saved AUROC curve to {auroc_path}")

    # --- Plot AUPRC ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_fracs, train_auprc, marker="o", label="Train AUPRC")
    plt.plot(train_fracs, val_auprc,   marker="o", label="Dev AUPRC")
    plt.xlabel("Fraction of train windows used")
    plt.ylabel("AUPRC")
    plt.title(f"Learning Curve (AUPRC) - {model_name}")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    auprc_path = (plot_dir / f"learning_curve_auprc_{model_name}_top{len(top_idx)}.png").as_posix()
    plt.tight_layout()
    plt.savefig(auprc_path, dpi=200)
    plt.close()
    print(f"[LC][{model_name}] Saved AUPRC curve to {auprc_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    BASE_OUT_DIR = "./tusz_windows"

    # ============================================
    # Compute and save features for all splits
    # ============================================
    for split in ["train_internal", "dev_internal", "test_internal"]:
        compute_and_save_features_for_split(BASE_OUT_DIR, split_name=split)