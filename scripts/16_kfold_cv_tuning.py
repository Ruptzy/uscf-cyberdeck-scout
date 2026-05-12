"""
Phase 12: K-Fold Cross-Validation with GridSearchCV
====================================================
Satisfies rubric item #4a: "perform k-fold cross validation to pick your
hyperparameters and compare metrics across all three samples."

Design notes
------------
- Chess game data is time-ordered. A naive KFold would leak future games
  into past folds (cheating). We use TimeSeriesSplit on the training
  portion so every CV fold's validation block sits strictly AFTER its
  training block, mirroring how the model would be deployed.
- We still report a final, fully held-out chronological test set so
  train / validation / test metrics can be compared as the rubric asks.
- Three model families are tuned: Logistic Regression (benchmark),
  Random Forest (tree ensemble), Gradient Boosting (boosted ensemble).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_metrics(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "accuracy":  accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "f1":        f1_score(y, preds, zero_division=0),
        "roc_auc":   roc_auc_score(y, probs),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


def fmt(d):
    return (
        f"  Accuracy:  {d['accuracy']:.4f}\n"
        f"  Precision: {d['precision']:.4f}\n"
        f"  Recall:    {d['recall']:.4f}\n"
        f"  F1 Score:  {d['f1']:.4f}\n"
        f"  ROC-AUC:   {d['roc_auc']:.4f}"
    )


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "data", "processed", "features_v3_recency.csv")
    out_dir = os.path.join(base_dir, "outputs", "models")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(data_file):
        print(f"Error: Could not find {data_file}")
        sys.exit(1)

    print("=" * 60)
    print("PHASE 12: K-FOLD CROSS-VALIDATION (TimeSeriesSplit + GridSearchCV)")
    print("=" * 60)

    df = pd.read_csv(data_file)
    df["event_end_date"] = pd.to_datetime(df["event_end_date"], errors="coerce")
    df = df.sort_values(by=["event_end_date", "game_id"]).reset_index(drop=True)

    # Cold-start flags
    df["missing_recent_avg_opp_90d"] = df["player_recent_avg_opponent_rating_90d"].isna().astype(int)
    df["missing_recent_win_rate_90d"] = df["player_recent_win_rate_90d"].isna().astype(int)

    num_features = [
        "player_pre_rating",
        "opponent_pre_rating",
        "rating_diff",
        "player_games_last_30d",
        "player_games_last_90d",
        "player_games_last_365d",
        "player_recent_avg_opponent_rating_90d",
        "player_recent_win_rate_90d",
        "missing_recent_avg_opp_90d",
        "missing_recent_win_rate_90d",
    ]
    cat_features = ["time_control"]

    X = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=True)
    y = df["target_binary"].astype(int)

    # Chronological 70 / 15 / 15 split.  CV runs INSIDE the 70%.
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end].copy(), y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:].copy(),  y.iloc[val_end:]

    print(f"\n--- Split Sizes ---")
    print(f"Train (CV pool): {len(X_train)} rows")
    print(f"Validation:      {len(X_val)} rows")
    print(f"Test (hold-out): {len(X_test)} rows")
    print(f"Train date range: {df['event_end_date'].iloc[0].date()} -> {df['event_end_date'].iloc[train_end-1].date()}")
    print(f"Val   date range: {df['event_end_date'].iloc[train_end].date()} -> {df['event_end_date'].iloc[val_end-1].date()}")
    print(f"Test  date range: {df['event_end_date'].iloc[val_end].date()} -> {df['event_end_date'].iloc[-1].date()}")

    # TimeSeriesSplit honours temporal ordering of games.
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Numeric pipeline: impute then (optionally) scale.
    impute_cols = [c for c in X_train.columns if c in num_features]
    imputer = SimpleImputer(strategy="median")
    X_train[impute_cols] = imputer.fit_transform(X_train[impute_cols])
    X_val[impute_cols]   = imputer.transform(X_val[impute_cols])
    X_test[impute_cols]  = imputer.transform(X_test[impute_cols])

    # ---------------------------------------------------------------
    # Model 1 - Logistic Regression (benchmark, rubric item #3)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 1 / 3 : Logistic Regression (Benchmark)")
    print("=" * 60)

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }
    lr_search = GridSearchCV(
        lr_pipe, lr_grid, cv=tscv, scoring="roc_auc",
        n_jobs=-1, refit=True, return_train_score=True, verbose=0,
    )
    lr_search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC (mean over {n_splits} folds): {lr_search.best_score_:.4f}")
    print(f"Best params: {lr_search.best_params_}")

    lr_train_m = get_metrics(lr_search.best_estimator_, X_train, y_train)
    lr_val_m   = get_metrics(lr_search.best_estimator_, X_val,   y_val)
    lr_test_m  = get_metrics(lr_search.best_estimator_, X_test,  y_test)
    print("\n[Train]\n"      + fmt(lr_train_m))
    print("\n[Validation]\n" + fmt(lr_val_m))
    print("\n[Test]\n"       + fmt(lr_test_m))

    # ---------------------------------------------------------------
    # Model 2 - Random Forest
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 2 / 3 : Random Forest")
    print("=" * 60)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 4],
    }
    rf_search = GridSearchCV(
        rf, rf_grid, cv=tscv, scoring="roc_auc",
        n_jobs=-1, refit=True, return_train_score=True, verbose=0,
    )
    rf_search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC (mean over {n_splits} folds): {rf_search.best_score_:.4f}")
    print(f"Best params: {rf_search.best_params_}")

    rf_train_m = get_metrics(rf_search.best_estimator_, X_train, y_train)
    rf_val_m   = get_metrics(rf_search.best_estimator_, X_val,   y_val)
    rf_test_m  = get_metrics(rf_search.best_estimator_, X_test,  y_test)
    print("\n[Train]\n"      + fmt(rf_train_m))
    print("\n[Validation]\n" + fmt(rf_val_m))
    print("\n[Test]\n"       + fmt(rf_test_m))

    # ---------------------------------------------------------------
    # Model 3 - Gradient Boosting
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 3 / 3 : Gradient Boosting")
    print("=" * 60)

    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }
    gb_search = GridSearchCV(
        gb, gb_grid, cv=tscv, scoring="roc_auc",
        n_jobs=-1, refit=True, return_train_score=True, verbose=0,
    )
    gb_search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC (mean over {n_splits} folds): {gb_search.best_score_:.4f}")
    print(f"Best params: {gb_search.best_params_}")

    gb_train_m = get_metrics(gb_search.best_estimator_, X_train, y_train)
    gb_val_m   = get_metrics(gb_search.best_estimator_, X_val,   y_val)
    gb_test_m  = get_metrics(gb_search.best_estimator_, X_test,  y_test)
    print("\n[Train]\n"      + fmt(gb_train_m))
    print("\n[Validation]\n" + fmt(gb_val_m))
    print("\n[Test]\n"       + fmt(gb_test_m))

    # ---------------------------------------------------------------
    # Aggregate + save
    # ---------------------------------------------------------------
    rows = []
    for name, train_m, val_m, test_m, best in [
        ("LogisticRegression", lr_train_m, lr_val_m, lr_test_m, lr_search),
        ("RandomForest",       rf_train_m, rf_val_m, rf_test_m, rf_search),
        ("GradientBoosting",   gb_train_m, gb_val_m, gb_test_m, gb_search),
    ]:
        for split_name, m in [("Train", train_m), ("Validation", val_m), ("Test", test_m)]:
            rows.append({
                "model": name,
                "split": split_name,
                "accuracy":  m["accuracy"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "f1":        m["f1"],
                "roc_auc":   m["roc_auc"],
                "cv_mean_roc_auc": best.best_score_,
                "best_params": json.dumps(best.best_params_),
            })

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(out_dir, "kfold_cv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Per-fold CV details
    fold_records = []
    for name, search in [
        ("LogisticRegression", lr_search),
        ("RandomForest", rf_search),
        ("GradientBoosting", gb_search),
    ]:
        best_idx = search.best_index_
        for fold in range(n_splits):
            fold_records.append({
                "model": name,
                "fold": fold + 1,
                "train_roc_auc": search.cv_results_[f"split{fold}_train_score"][best_idx],
                "val_roc_auc":   search.cv_results_[f"split{fold}_test_score"][best_idx],
            })
    folds_df = pd.DataFrame(fold_records)
    folds_path = os.path.join(out_dir, "kfold_cv_per_fold.csv")
    folds_df.to_csv(folds_path, index=False)

    # Feature importances for tree models
    fi_records = []
    for name, est in [("RandomForest", rf_search.best_estimator_), ("GradientBoosting", gb_search.best_estimator_)]:
        for feat, imp in sorted(zip(X_train.columns, est.feature_importances_), key=lambda x: -x[1]):
            fi_records.append({"model": name, "feature": feat, "importance": imp})
    fi_df = pd.DataFrame(fi_records)
    fi_path = os.path.join(out_dir, "kfold_cv_feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)

    # ---------------------------------------------------------------
    # Final comparison summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL COMPARISON (Test set, k-fold-tuned models)")
    print("=" * 60)
    print(f"{'Model':<22} {'CV AUC':>8} {'Val AUC':>10} {'Test AUC':>10} {'Test F1':>10}")
    for name, val_m, test_m, best in [
        ("LogisticRegression", lr_val_m, lr_test_m, lr_search),
        ("RandomForest",       rf_val_m, rf_test_m, rf_search),
        ("GradientBoosting",   gb_val_m, gb_test_m, gb_search),
    ]:
        print(f"{name:<22} {best.best_score_:>8.4f} {val_m['roc_auc']:>10.4f} {test_m['roc_auc']:>10.4f} {test_m['f1']:>10.4f}")

    print(f"\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {folds_path}")
    print(f"  {fi_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
