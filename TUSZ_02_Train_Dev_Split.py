from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from pathlib import Path
import random

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BASE_OUT_DIR = Path("./tusz_windows").expanduser().resolve()

TRAIN_NPZ = BASE_OUT_DIR / "train" / "split_windows.npz"
EVAL_NPZ  = BASE_OUT_DIR / "eval"  / "split_windows.npz"

# Subject-level filtering thresholds
MIN_SEIZURE_EVENTS = 1  # subject must have at least 1 seizure in metadata
MIN_TOTAL_WINDOWS = 20  # allow low-window subjects

# Desired internal split sizes (will scale if insufficient subjects)
TRAIN_DEV_RATIO = 0.80   # 80% train, 20% dev
N_TEST_SUBJ     = 10     # fixed number from EVAL

RNG_SEED = 42


# ---------------------------------------------------------
# helper for meta_list (handles list/array)
# ---------------------------------------------------------
def load_meta_list(meta_array):
    if isinstance(meta_array, list):
        return meta_array
    out = []
    for m in meta_array:
        if isinstance(m, dict):
            out.append(m)
        else:
            out.append(m.item())
    return out


# ---------------------------------------------------------
# helper: subset arrays/meta/rec_idx by subject set
# ---------------------------------------------------------
def subset_by_subjects(X_all, y_all, meta_list, rec_idx_all, subj_id_per_rec, chosen_subjects):

    chosen_subjects = set(chosen_subjects)

    chosen_rec_indices = [
        r_idx for r_idx, sid in enumerate(subj_id_per_rec)
        if sid in chosen_subjects
    ]
    chosen_rec_set = set(chosen_rec_indices)

    win_mask = np.array([r_idx in chosen_rec_set for r_idx in rec_idx_all], dtype=bool)

    X_sub = X_all[win_mask]
    y_sub = y_all[win_mask]
    old_rec_idx_sub = rec_idx_all[win_mask]

    old_to_new = {old: new for new, old in enumerate(chosen_rec_indices)}
    rec_idx_sub = np.array([old_to_new[ri] for ri in old_rec_idx_sub], dtype=np.int32)

    meta_sub = [meta_list[old] for old in chosen_rec_indices]

    print(f"[Split] subset_by_subjects → {X_sub.shape[0]} windows, {len(meta_sub)} recordings.")
    return X_sub, y_sub, meta_sub, rec_idx_sub


# =========================================================
# === PART A: TRAIN → internal TRAIN / DEV (80/20) =========
# =========================================================
print("\n===============================")
print(" Loading TRAIN for 80/20 split ")
print("===============================\n")

data_train = np.load(TRAIN_NPZ, allow_pickle=True)

X_all       = data_train["X"]
y_all       = data_train["y"]
meta_list   = load_meta_list(data_train["meta"])
rec_idx_all = data_train["rec_idx"]
ch_names    = data_train["ch_names"] if "ch_names" in data_train.files else None

# subject IDs
subj_id_per_rec = [m.get("subject_id", f"subj_{i}") for i, m in enumerate(meta_list)]
subj_id_per_win = np.array([subj_id_per_rec[i] for i in rec_idx_all])

unique_subjects = sorted(set(subj_id_per_win))
print(f"[Split] Found {len(unique_subjects)} subjects in TRAIN.")


# ---- compute per-subject stats
subj_stats = {}
for sid in unique_subjects:
    mask = (subj_id_per_win == sid)
    y_s  = y_all[mask]
    subj_stats[sid] = {
        "n_win": int(mask.sum()),
        "n_pos": int((y_s == 1).sum()),
        "n_neg": int((y_s == 0).sum()),
    }

# ---- SUBJECT FILTERING LOGIC ----
good_subjects = []
for sid in unique_subjects:
    mask = (subj_id_per_win == sid)
    y_s  = y_all[mask]

    # Extract meta-data for that subject from recordings
    recs_for_s = [m for m in meta_list if m.get("subject_id") == sid]

    # Count true seizure events (based on CSV metadata)
    seizure_events = sum(1 for m in recs_for_s 
                         if m.get("first_seizure_onset_sec") is not None)

    if seizure_events >= MIN_SEIZURE_EVENTS and int(mask.sum()) >= MIN_TOTAL_WINDOWS:
        good_subjects.append(sid)

print(f"\n[Split] Subjects passing NEW filters "
      f"(>= {MIN_SEIZURE_EVENTS} seizure event & >= {MIN_TOTAL_WINDOWS} windows): "
      f"{len(good_subjects)}")



# ---- make 80/20 split
rng = random.Random(RNG_SEED)
good_shuffled = good_subjects[:]
rng.shuffle(good_shuffled)

n_total = len(good_shuffled)
n_train_subj = int(round(n_total * TRAIN_DEV_RATIO))
n_dev_subj   = n_total - n_train_subj

train_subjects = good_shuffled[:n_train_subj]
dev_subjects   = good_shuffled[n_train_subj:]

print(f"[Split] Internal TRAIN subjects: {len(train_subjects)}")
print(f"[Split] Internal DEV subjects  : {len(dev_subjects)}")


# ---- subset TRAIN and DEV
X_train_int, y_train_int, meta_train_int, rec_idx_train_int = subset_by_subjects(
    X_all, y_all, meta_list, rec_idx_all, subj_id_per_rec, train_subjects
)

X_dev_int, y_dev_int, meta_dev_int, rec_idx_dev_int = subset_by_subjects(
    X_all, y_all, meta_list, rec_idx_all, subj_id_per_rec, dev_subjects
)


# =========================================================
# === PART B: EVAL → internal TEST (10 subjects) ==========
# =========================================================
print("\n===============================")
print(" Loading EVAL for TEST split   ")
print("===============================\n")

data_eval = np.load(EVAL_NPZ, allow_pickle=True)

X_eval       = data_eval["X"]
y_eval       = data_eval["y"]
meta_eval    = load_meta_list(data_eval["meta"])
rec_idx_eval = data_eval["rec_idx"]
ch_names_eval = data_eval["ch_names"] if "ch_names" in data_eval.files else ch_names

subj_id_per_rec_eval = [m.get("subject_id", f"evalsubj_{i}") for i, m in enumerate(meta_eval)]
subj_id_per_win_eval = np.array([subj_id_per_rec_eval[i] for i in rec_idx_eval])

unique_eval_subj = sorted(set(subj_id_per_win_eval))
print(f"[Split] Found {len(unique_eval_subj)} subjects in EVAL.")


# ---- sort by #windows to pick best test subjects
eval_stats = { sid: int((subj_id_per_win_eval == sid).sum()) for sid in unique_eval_subj }
eval_sorted = sorted(eval_stats.keys(), key=lambda s: -eval_stats[s])

test_subjects = eval_sorted[:N_TEST_SUBJ]
print(f"[Split] TEST subjects chosen (from EVAL): {len(test_subjects)}")


# ---- subset EVAL → TEST_INTERNAL
X_test_int, y_test_int, meta_test_int, rec_idx_test_int = subset_by_subjects(
    X_eval, y_eval, meta_eval, rec_idx_eval, subj_id_per_rec_eval, test_subjects
)


# =========================================================
# === PART C: SAVE ALL RESULTS =============================
# =========================================================
(train_int_dir := BASE_OUT_DIR / "train_internal").mkdir(parents=True, exist_ok=True)
(dev_int_dir   := BASE_OUT_DIR / "dev_internal").mkdir(parents=True, exist_ok=True)
(test_int_dir  := BASE_OUT_DIR / "test_internal").mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    train_int_dir / "split_windows.npz",
    X=X_train_int, y=y_train_int, meta=meta_train_int,
    ch_names=ch_names, rec_idx=rec_idx_train_int
)
np.savez_compressed(
    dev_int_dir / "split_windows.npz",
    X=X_dev_int, y=y_dev_int, meta=meta_dev_int,
    ch_names=ch_names, rec_idx=rec_idx_dev_int
)
np.savez_compressed(
    test_int_dir / "split_windows.npz",
    X=X_test_int, y=y_test_int, meta=meta_test_int,
    ch_names=ch_names_eval, rec_idx=rec_idx_test_int
)

print("\n===============================")
print(" Saved: train_internal / dev_internal / test_internal")
print("===============================\n")
