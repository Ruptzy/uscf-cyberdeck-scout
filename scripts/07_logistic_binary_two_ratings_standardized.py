import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(base_dir, 'data', 'processed', 'features_v1.csv')
    out_dir = os.path.join(base_dir, 'outputs', 'models')
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(data_file):
        print(f"Error: Could not find {data_file}")
        sys.exit(1)
        
    df = pd.read_csv(data_file)
    
    print("="*50)
    print("PHASE 5C: LOGISTIC REGRESSION BINARY (TWO RATINGS STANDARDIZED)")
    print("="*50)
    
    # 1. Prepare & Sort chronologically
    df['event_end_date'] = pd.to_datetime(df['event_end_date'], errors='coerce')
    df = df.sort_values(by=['event_end_date', 'game_id']).reset_index(drop=True)
    
    # Subset features and target
    features_df = df[['player_pre_rating', 'opponent_pre_rating', 'time_control']].copy()
    y = df['target_binary'].astype(int)
    
    # One-hot encode time_control
    X = pd.get_dummies(features_df, columns=['time_control'], drop_first=True)
    
    # 2. Time-based split (70% Train, 15% Validation, 15% Test)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end].copy(), y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:].copy(), y.iloc[val_end:]
    
    # 3. Standardize numeric features ONLY based on training data
    numeric_cols = ['player_pre_rating', 'opponent_pre_rating']
    scaler = StandardScaler()
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 4. Fit the benchmark model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Helper to calculate and format metrics
    def get_metrics(X_data, y_data):
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
        
    train_m = get_metrics(X_train, y_train)
    val_m = get_metrics(X_val, y_val)
    test_m = get_metrics(X_test, y_test)
    
    # 5. Report benchmark metrics
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
    
    # 6. Report interpretability outputs
    print("\n--- Model Coefficients ---")
    coefs = list(zip(X_train.columns, model.coef_[0]))
    intercept = model.intercept_[0]
    
    coef_df = pd.DataFrame(coefs, columns=['Feature', 'Coefficient'])
    coef_df.loc[len(coef_df)] = ['Intercept', intercept]
    
    for feat, coef in coefs:
        print(f"  {feat}: {coef:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    
    print("\n--- Interpretation ---")
    print("- player_pre_rating (standardized): By how much the log-odds of winning change for every 1 standard deviation increase in player rating.")
    print("- opponent_pre_rating (standardized): By how much the log-odds change for every 1 standard deviation increase in opponent rating.")
    print("- Because the ratings are standardized, the algorithm is less likely to suffer from optimization artifacts caused by large raw integers (e.g. 2000+), allowing it to assign the intuitive signs (+ for player, - for opponent) more cleanly.")
    
    # 7. Save outputs
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(out_dir, 'logistic_binary_two_ratings_std_metrics.csv')
    coefs_path = os.path.join(out_dir, 'logistic_binary_two_ratings_std_coefs.csv')
    
    metrics_df.to_csv(metrics_path, index=False)
    coef_df.to_csv(coefs_path, index=False)
    
    # 8. Comparison note
    print("\n" + "="*50)
    print("SUMMARY & COMPARISON NOTE")
    print("="*50)
    print(f"Outputs saved to: {out_dir}")
    print(f"- {os.path.basename(metrics_path)}")
    print(f"- {os.path.basename(coefs_path)}")
    
    print("\n>> Comparison to Unstandardized Baseline 2 <<")
    print("1. Did metrics change materially? Usually, logistic regression yields very similar metrics before and after standardizing, but sometimes the solver finds a slightly tighter optimization boundary.")
    print("2. Did coefficients become clearer? Standardizing brings large raw values (like Elo ratings ~1500-2800) down to a mean of 0 and std of 1. This prevents the default L2 regularization penalty from disproportionately squashing the rating coefficients, often returning the signs to their intuitive (+ for player, - for opponent) states.")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
