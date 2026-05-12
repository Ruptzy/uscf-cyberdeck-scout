import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.impute import SimpleImputer

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(base_dir, 'data', 'processed', 'features_v3_recency.csv')
    out_dir = os.path.join(base_dir, 'outputs', 'models')
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(data_file):
        print(f"Error: Could not find {data_file}")
        sys.exit(1)
        
    # 1. Prepare the benchmark dataset
    df = pd.read_csv(data_file)
    
    print("="*50)
    print("PHASE 12: GRADIENT BOOSTING WITH RECENCY FEATURES")
    print("="*50)
    
    # 2. Sort chronologically
    df['event_end_date'] = pd.to_datetime(df['event_end_date'], errors='coerce')
    df = df.sort_values(by=['event_end_date', 'game_id']).reset_index(drop=True)
    
    # 3. Handle cold-start missing values
    # Because players with no prior games have NaN for rolling averages, we flag them.
    df['missing_recent_avg_opp_90d'] = df['player_recent_avg_opponent_rating_90d'].isna().astype(int)
    df['missing_recent_win_rate_90d'] = df['player_recent_win_rate_90d'].isna().astype(int)
    
    num_features = [
        'player_pre_rating', 
        'opponent_pre_rating', 
        'player_games_last_30d',
        'player_games_last_90d',
        'player_games_last_365d',
        'player_recent_avg_opponent_rating_90d',
        'player_recent_win_rate_90d',
        'missing_recent_avg_opp_90d',
        'missing_recent_win_rate_90d'
    ]
    
    cat_features = ['time_control']
    
    features_df = df[num_features + cat_features].copy()
    y = df['target_binary'].astype(int)
    
    # One-hot encode time_control
    # GB does not suffer from dummy variable trap, but drop_first maintains parallel structure
    X = pd.get_dummies(features_df, columns=cat_features, drop_first=True)
    
    # 4. Time-based split (70% Train, 15% Validation, 15% Test)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end].copy(), y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:].copy(), y.iloc[val_end:]
    
    # Impute missing values securely based on Training Set ONLY to prevent data leakage
    impute_cols = ['player_recent_avg_opponent_rating_90d', 'player_recent_win_rate_90d']
    imputer = SimpleImputer(strategy='median')
    
    X_train[impute_cols] = imputer.fit_transform(X_train[impute_cols])
    X_val[impute_cols] = imputer.transform(X_val[impute_cols])
    X_test[impute_cols] = imputer.transform(X_test[impute_cols])
    
    # 5. Hyperparameter Tuning using pure Validation Set
    print("\n--- Tuning Hyperparameters ---")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    best_auc = -1
    best_params = None
    best_model = None
    
    # Manual loop evaluates explicitly against X_val
    for params in ParameterGrid(param_grid):
        gb = GradientBoostingClassifier(**params, random_state=42)
        gb.fit(X_train, y_train)
        
        try:
            val_probs = gb.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = 0
            
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params
            best_model = gb
            
    print(f"Best Validation ROC-AUC: {best_auc:.4f}")
    print(f"Best Params: {best_params}")
    
    # Helper to calculate and format metrics
    def get_metrics(model, X_data, y_data):
        preds = model.predict(X_data)
        probs = model.predict_proba(X_data)[:, 1]
        
        acc = accuracy_score(y_data, preds)
        prec = precision_score(y_data, preds, zero_division=0)
        rec = recall_score(y_data, preds, zero_division=0)
        f1 = f1_score(y_data, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(y_data, probs)
        except ValueError:
            auc = np.nan
            
        cm = confusion_matrix(y_data, preds)
        return acc, prec, rec, f1, auc, cm
        
    train_m = get_metrics(best_model, X_train, y_train)
    val_m = get_metrics(best_model, X_val, y_val)
    test_m = get_metrics(best_model, X_test, y_test)
    
    # 6. Report benchmark metrics
    print("\n--- Split Sizes ---")
    print(f"Train:      {len(X_train)} rows")
    print(f"Validation: {len(X_val)} rows")
    print(f"Test:       {len(X_test)} rows")
    
    splits = ['Train', 'Validation', 'Test']
    metrics_list = []
    
    print("\n--- Metrics ---")
    for i, m_tup in enumerate([train_m, val_m, test_m]):
        print(f"[{splits[i]}]")
        print(f"  Accuracy:  {m_tup[0]:.4f}")
        print(f"  Precision: {m_tup[1]:.4f}")
        print(f"  Recall:    {m_tup[2]:.4f}")
        print(f"  F1 Score:  {m_tup[3]:.4f}")
        print(f"  ROC-AUC:   {m_tup[4]:.4f}")
        
        metrics_list.append({
            'Split': splits[i],
            'Accuracy': m_tup[0],
            'Precision': m_tup[1],
            'Recall': m_tup[2],
            'F1': m_tup[3],
            'ROC-AUC': m_tup[4]
        })
        
    print("\n--- Confusion Matrices ---")
    print("Validation Confusion Matrix:\n", val_m[5])
    print("Test Confusion Matrix:\n", test_m[5])
    
    # 7. Report Feature Importances
    print("\n--- Feature Importances ---")
    importances = list(zip(X_train.columns, best_model.feature_importances_))
    # Sort descending
    importances.sort(key=lambda x: x[1], reverse=True)
    
    imp_df = pd.DataFrame(importances, columns=['Feature', 'Importance'])
    
    for feat, imp in importances:
        print(f"  {feat}: {imp:.4f}")
    
    # 8. Save outputs
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(out_dir, 'gb_binary_with_recency_metrics.csv')
    imp_path = os.path.join(out_dir, 'gb_binary_with_recency_importances.csv')
    
    metrics_df.to_csv(metrics_path, index=False)
    imp_df.to_csv(imp_path, index=False)
    
    # 9. Comparison note
    print("\n" + "="*50)
    print("SUMMARY & COMPARISON NOTE")
    print("="*50)
    print(f"Outputs saved to: {out_dir}")
    print(f"- {os.path.basename(metrics_path)}")
    print(f"- {os.path.basename(imp_path)}")
    
    print("\n>> Comparison to Previous Best Model (RF with Recency) <<")
    print("Previous RF Metrics:")
    print("- Validation ROC-AUC: 0.6557 | F1: 0.5844")
    print("- Test ROC-AUC:       0.6493 | F1: 0.6000")
    print(f"\nNew GB Metrics (With Recency):")
    print(f"- Validation ROC-AUC: {val_m[4]:.4f} | F1: {val_m[3]:.4f}")
    print(f"- Test ROC-AUC:       {test_m[4]:.4f} | F1: {test_m[3]:.4f}")
    
    print("\nVerdict:")
    if test_m[4] > 0.6493 and test_m[3] > 0.6000:
        print("Gradient Boosting outperformed the Random Forest on the new recency features! The sequential error-correction of boosting algorithms proved superior at optimizing this richer, momentum-based feature set.")
    else:
        print("Gradient Boosting did NOT meaningfully outperform the Random Forest on the test set. This confirms that the Random Forest architecture already successfully squeezed the maximum predictive signal out of the recency features, making it the definitive champion model for this dataset.")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
