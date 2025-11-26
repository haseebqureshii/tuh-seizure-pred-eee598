"""
Train seizure prediction models from precomputed features.

Assumes:
  - Feature extraction has already been run for:
      ./tusz_windows/train_internal/features.npz
      ./tusz_windows/dev_internal/features.npz
  - Helper functions are available in FE.py (or similar):
      - load_features_with_meta
      - shap_feature_selection
      - print_shap_mass_summary
      - plot_learning_curves_for_model
      - evaluate_model_on_split
"""

import numpy as np
from pathlib import Path
import joblib
import TUSZ_03_FE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

BASE_OUT_DIR = Path("./tusz_windows").expanduser().resolve()


def main():
    # ============================================
    # 1) Load TRAIN & DEV features (internal splits)
    # ============================================
    print("\n[Main] Loading TRAIN_INTERNAL features...")
    X_train, y_train, fnames = TUSZ_03_FE.load_features_with_meta(BASE_OUT_DIR, "train_internal")
    print("[Main] TRAIN feature matrix shape:", X_train.shape)
    print("[Main] TRAIN labels shape:", y_train.shape)

    print("\n[Main] Loading DEV_INTERNAL features...")
    X_dev, y_dev, fnames_dev = TUSZ_03_FE.load_features_with_meta(BASE_OUT_DIR, "dev_internal")
    print("[Main] DEV feature matrix shape:", X_dev.shape)
    print("[Main] DEV labels shape:", y_dev.shape)

    # Compute prevalence on DEV set
    pos = (y_dev == 1).sum()
    total = len(y_dev)
    print(f"[Main] DEV positives: {pos} / {total} "
      f"({pos/total:.4f} prevalence → random AUPRC baseline ≈ {pos/total:.4f})")

    if list(fnames_dev) != list(fnames):
        raise RuntimeError("[Main] Feature name mismatch between TRAIN and DEV.")

    # ============================================
    # 2) SHAP feature selection: RF
    # ============================================
    rf_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )

    print("\n[Main] Running SHAP feature selection for RF...")
    top_idx_rf, sv_rf = TUSZ_03_FE.shap_feature_selection(
        X=X_train,
        y=y_train,
        top_k=200,
        model=rf_model,
        random_state=0,
        sample_for_shap=1000,
    )

    TUSZ_03_FE.print_shap_mass_summary("RF", sv_rf)
    rf_feature_ranking = np.argsort(np.abs(sv_rf))[::-1]
    print("\n[RF] === Top Features ===")
    for rank, feat_idx in enumerate(rf_feature_ranking[:200], start=1):
        print(f"{rank:02d}. {fnames[feat_idx]}")

    # ============================================
    # 3) SHAP feature selection: XGB
    # ============================================
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / max(pos, 1)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        n_jobs=-1,
        scale_pos_weight=scale,
    )

    print("\n[Main] Running SHAP feature selection for XGB...")
    top_idx_xgb, sv_xgb = TUSZ_03_FE.shap_feature_selection(
        X=X_train,
        y=y_train,
        top_k=200,
        model=xgb_model,
        random_state=0,
        sample_for_shap=1000,
    )

    TUSZ_03_FE.print_shap_mass_summary("XGB", sv_xgb)
    xgb_feature_ranking = np.argsort(np.abs(sv_xgb))[::-1]
    print("\n[XGB] === Top Features ===")
    for rank, feat_idx in enumerate(xgb_feature_ranking[:200], start=1):
        print(f"{rank:02d}. {fnames[feat_idx]}")

    # Intersection for inspection
    K = 200
    rf_top_k = np.argsort(np.abs(sv_rf))[::-1][:K]
    xgb_top_k = np.argsort(np.abs(sv_xgb))[::-1][:K]

    print("\n=== RF vs XGB – cumulative SHAP mass ===")
    inter_rf_xgb = set(rf_top_k) & set(xgb_top_k)
    print(f"RF–XGB intersection size: {len(inter_rf_xgb)}")
    for idx in sorted(inter_rf_xgb):
        print(fnames[idx])

    # ============================================
    # 4) Optional SVM (reuse XGB feature set, no SHAP)
    # ============================================
    svm_model = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=0,
    )
    top_idx_svm = top_idx_xgb  # reuse XGB's top-200

    # ============================================
    # 5) Learning curves (Train vs DEV)
    # ============================================
    print("\n[Main] Computing learning curves (Train vs DEV)...")

    # RF
    TUSZ_03_FE.plot_learning_curves_for_model(
        base_out_dir=BASE_OUT_DIR,
        model=rf_model,
        model_name="RF",
        X_train=X_train,
        y_train=y_train,
        X_val=X_dev,
        y_val=y_dev,
        top_idx=top_idx_rf,
        n_points=5,
        random_state=0,
    )

    # XGB
    TUSZ_03_FE.plot_learning_curves_for_model(
        base_out_dir=BASE_OUT_DIR,
        model=xgb_model,
        model_name="XGB",
        X_train=X_train,
        y_train=y_train,
        X_val=X_dev,
        y_val=y_dev,
        top_idx=top_idx_xgb,
        n_points=5,
        random_state=0,
    )

    # SVM (comment out if too slow)
    TUSZ_03_FE.plot_learning_curves_for_model(
        base_out_dir=BASE_OUT_DIR,
        model=svm_model,
        model_name="SVM",
        X_train=X_train,
        y_train=y_train,
        X_val=X_dev,
        y_val=y_dev,
        top_idx=top_idx_svm,
        n_points=5,
        random_state=0,
    )

    # ============================================
    # 6) Evaluation on DEV split (full FE path)
    # ============================================
    EVAL_SPLIT_NAME = "dev_internal"

    print("\n[Main] Evaluating models on DEV_INTERNAL (full FE path)...")

    metrics_xgb = TUSZ_03_FE.evaluate_model_on_split(
        model=xgb_model,
        top_idx=top_idx_xgb,
        base_out_dir=BASE_OUT_DIR,
        train_split_name="train_internal",
        eval_split_name=EVAL_SPLIT_NAME,
    )

    metrics_rf = TUSZ_03_FE.evaluate_model_on_split(
        model=rf_model,
        top_idx=top_idx_rf,
        base_out_dir=BASE_OUT_DIR,
        train_split_name="train_internal",
        eval_split_name=EVAL_SPLIT_NAME,
    )

    metrics_svm = TUSZ_03_FE.evaluate_model_on_split(
        model=svm_model,
        top_idx=top_idx_svm,
        base_out_dir=BASE_OUT_DIR,
        train_split_name="train_internal",
        eval_split_name=EVAL_SPLIT_NAME,
    )

    print("\n[Summary] DEV_INTERNAL metrics:")
    print(" RF :", metrics_rf)
    print(" XGB:", metrics_xgb)
    print(" SVM:", metrics_svm)

    # ============================================
    # 7) Save final trained models
    # ============================================
    SAVE_FINAL_MODELS = True
    if SAVE_FINAL_MODELS:
        model_dir = BASE_OUT_DIR / "saved_models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Refit on full TRAIN with selected features before saving
        X_train_rf = X_train[:, top_idx_rf]
        X_train_xgb = X_train[:, top_idx_xgb]
        X_train_svm = X_train[:, top_idx_svm]

        # NaN/inf guards (belt & suspenders)
        for name, arr in [("RF", X_train_rf), ("XGB", X_train_xgb), ("SVM", X_train_svm)]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"[Save][{name}] WARN: NaNs/inf in TRAIN features; replacing with zero")
                arr[...] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        rf_model.fit(X_train_rf, y_train)
        xgb_model.fit(X_train_xgb, y_train)
        svm_model.fit(X_train_svm, y_train)

        joblib.dump(
            {"model": rf_model, "top_idx": top_idx_rf, "feature_names": fnames},
            (model_dir / "rf_model_200feat.joblib").as_posix(),
        )
        joblib.dump(
            {"model": xgb_model, "top_idx": top_idx_xgb, "feature_names": fnames},
            (model_dir / "xgb_model_200feat.joblib").as_posix(),
        )
        joblib.dump(
            {"model": svm_model, "top_idx": top_idx_svm, "feature_names": fnames},
            (model_dir / "svm_model_200feat.joblib").as_posix(),
        )

        print(f"\n[Save] Models saved under: {model_dir.as_posix()}")


if __name__ == "__main__":
    main()