"""
eval_final_on_eval.py

Use this AFTER:
  1) Preprocessing + feature extraction have been run for the 'eval' split
     (so that ./tusz_windows/eval/features.npz exists).
  2) FE.py has trained and saved models under ./tusz_windows/saved_models/
     (rf_model_200feat.joblib, xgb_model_200feat.joblib, svm_model_200feat.joblib).

This script:
  - Loads eval features via FE.load_features_with_meta()
  - Loads each saved model bundle (model + top_idx + feature_names)
  - Applies the same feature subset
  - Guards against NaNs / infs in the selected features
  - Computes AUROC, AUPRC, best F1 & corresponding threshold
  - (Optionally) saves ROC & PR curves for each model
"""

import TUSZ_03_FE
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

# ----------------------------------------------
# Config
# ----------------------------------------------
BASE_OUT_DIR = Path("./tusz_windows").expanduser().resolve()
MODEL_DIR = BASE_OUT_DIR / "saved_models"
EVAL_SPLIT_NAME = "eval"          # official TUSZ eval split
SAVE_PLOTS = True                 # set False if you don't want ROC/PR PNGs


def evaluate_saved_model_on_split(
    model_bundle_path: Path,
    base_out_dir: Path,
    split_name: str = "eval",
):
    """
    Load a saved model bundle (model, top_idx, feature_names) and evaluate it
    on features from `split_name` (e.g. 'eval').

    Returns: dict with metrics + scores.
    """
    print("\n===================================================")
    print(f"[Eval] Evaluating model bundle: {model_bundle_path.name}")
    print("===================================================")

    # ---------- Load model bundle ----------
    bundle = joblib.load(model_bundle_path.as_posix())
    model = bundle["model"]
    top_idx = np.array(bundle["top_idx"], dtype=int)
    feat_names_model = list(bundle["feature_names"])

    # ---------- Load eval features ----------
    X_eval, y_eval, feat_names_eval = TUSZ_03_FE.load_features_with_meta(base_out_dir, split_name)
    print(f"[Eval] Loaded EVAL features: X_eval.shape={X_eval.shape}, y_eval.shape={y_eval.shape}")

    # Sanity check: feature names align
    if len(feat_names_model) != len(feat_names_eval):
        print("[Eval][WARN] Feature name length mismatch: "
              f"model={len(feat_names_model)}, eval={len(feat_names_eval)}")
    else:
        mismatch = [i for i, (a, b) in enumerate(zip(feat_names_model, feat_names_eval)) if a != b]
        if mismatch:
            print(f"[Eval][WARN] Feature name mismatch at indices (first 10): {mismatch[:10]}")
        else:
            print("[Eval] Feature names match between model and eval split.")

    # ---------- Select top features ----------
    X_eval_sel = X_eval[:, top_idx]
    print(f"[Eval] Using {X_eval_sel.shape[1]} features (top_idx) from {X_eval.shape[1]} total.")

    # ---------- NaN / inf guard ----------
    if np.isnan(X_eval_sel).any() or np.isinf(X_eval_sel).any():
        print("[Eval] WARN: NaNs or infs found in EVAL selected features; replacing with zero.")
        col_bad = np.isnan(X_eval_sel).any(axis=0) | np.isinf(X_eval_sel).any(axis=0)
        bad_idx = np.where(col_bad)[0]
        print(f"[Eval] NaN/inf columns in EVAL (indices): {bad_idx}")
        X_eval_sel = np.nan_to_num(X_eval_sel, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- Predict scores ----------
    print("[Eval] Predicting scores on EVAL ...")
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_eval_sel)[:, 1]
    else:
        scores = model.decision_function(X_eval_sel)

    # ---------- Metrics ----------
    y_eval = np.asarray(y_eval).astype(int)
    pos = int((y_eval == 1).sum())
    neg = int((y_eval == 0).sum())
    print(f"[Eval] Class counts in EVAL: neg={neg}, pos={pos}")

    if pos == 0 or neg == 0:
        print("[Eval][WARN] EVAL has only one class; AUROC/AUPRC may be degenerate.")

    auroc = roc_auc_score(y_eval, scores)
    auprc = average_precision_score(y_eval, scores)

    prec, rec, thr = precision_recall_curve(y_eval, scores)
    f1_vals = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.argmax(f1_vals))
    best_f1 = float(f1_vals[best_idx])

    # precision_recall_curve returns `thr` with length = len(prec) - 1
    if best_idx == 0:
        best_thresh = 0.5  # fallback
    else:
        best_thresh = float(thr[best_idx - 1])

    print("\n[Eval] === FINAL METRICS on EVAL ===")
    print(f"AUROC : {auroc:.4f}")
    print(f"AUPRC : {auprc:.4f}")
    print(f"Best F1: {best_f1:.4f} at threshold ≈ {best_thresh:.3f}")

    # Derived counts at best threshold
    y_hat = (scores >= best_thresh).astype(int)
    tp = int(((y_hat == 1) & (y_eval == 1)).sum())
    fp = int(((y_hat == 1) & (y_eval == 0)).sum())
    tn = int(((y_hat == 0) & (y_eval == 0)).sum())
    fn = int(((y_hat == 0) & (y_eval == 1)).sum())

    precision_best = tp / max(tp + fp, 1)
    recall_best = tp / max(tp + fn, 1)
    spec_best = tn / max(tn + fp, 1)

    print("\n[Eval] Confusion at best F1 threshold:")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Precision={precision_best:.4f}, Recall (Sens)={recall_best:.4f}, Specificity={spec_best:.4f}")

    metrics = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "best_F1": best_f1,
        "best_threshold": best_thresh,
        "precision_at_best_F1": precision_best,
        "recall_at_best_F1": recall_best,
        "specificity_at_best_F1": spec_best,
        "scores": scores,
        "y_true": y_eval,
    }
    return metrics


def plot_roc_pr_curves(scores, y_true, out_dir: Path, model_name: str):
    """
    Save ROC and PR curves as PNGs for the given model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_roc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = out_dir / f"roc_{model_name}.png"
    plt.tight_layout()
    plt.savefig(roc_path.as_posix(), dpi=200)
    plt.close()
    print("[Eval] Saved ROC curve to", roc_path.as_posix())

    # PR
    prec, rec, _ = precision_recall_curve(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)

    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, label=f"AP = {auc_pr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve – {model_name}")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    pr_path = out_dir / f"pr_{model_name}.png"
    plt.tight_layout()
    plt.savefig(pr_path.as_posix(), dpi=200)
    plt.close()
    print("[Eval] Saved PR curve to", pr_path.as_posix())


if __name__ == "__main__":
    print("\n===============================")
    print(" Final EVAL Split Evaluation")
    print("===============================\n")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory {MODEL_DIR.as_posix()} not found. "
            "Run FE.py to train and save models first."
        )

    # Candidate model bundles we expect from FE.py
    candidate_models = [
        ("rf_model_200feat.joblib",  "RF"),
        ("xgb_model_200feat.joblib", "XGB"),
        ("svm_model_200feat.joblib", "SVM"),
    ]

    results = {}

    for fname, tag in candidate_models:
        bundle_path = MODEL_DIR / fname
        if not bundle_path.exists():
            print(f"[Eval] Skipping {tag}: bundle not found at {bundle_path.as_posix()}")
            continue

        metrics = evaluate_saved_model_on_split(
            model_bundle_path=bundle_path,
            base_out_dir=BASE_OUT_DIR,
            split_name=EVAL_SPLIT_NAME,
        )
        results[tag] = metrics

        if SAVE_PLOTS:
            plot_dir = MODEL_DIR / "eval_curves"
            plot_roc_pr_curves(
                scores=metrics["scores"],
                y_true=metrics["y_true"],
                out_dir=plot_dir,
                model_name=tag,
            )

    print("\n===============================")
    print(" Summary of EVAL metrics")
    print("===============================\n")
    for tag, m in results.items():
        print(f"{tag}: "
              f"AUROC={m['AUROC']:.4f}, "
              f"AUPRC={m['AUPRC']:.4f}, "
              f"BestF1={m['best_F1']:.4f}, "
              f"Thr≈{m['best_threshold']:.3f}")