import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(base_dir, 'data', 'processed', 'features_v1.csv')
    out_dir = os.path.join(base_dir, 'outputs', 'models')
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(data_file):
        print(f"Error: Could not find {data_file}")
        sys.exit(1)
        
    # 1. Prepare the benchmark dataset
    df = pd.read_csv(data_file)
    
    print("="*50)
    print("PHASE 5B: LOGISTIC REGRESSION BINARY (TWO RATINGS)")
    print("="*50)
    
    # 2. Sort chronologically
    df['event_end_date'] = pd.to_datetime(df['event_end_date'], errors='coerce')
    df = df.sort_values(by=['event_end_date', 'game_id']).reset_index(drop=True)
    
    # Subset features and target
    features_df = df[['player_pre_rating', 'opponent_pre_rating', 'time_control']].copy()
    y = df['target_binary'].astype(int)
    
    # One-hot encode time_control
    X = pd.get_dummies(features_df, columns=['time_control'], drop_first=True)
    
    # 3. Time-based split (70% Train, 15% Validation, 15% Test)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
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
    coefs = list(zip(X.columns, model.coef_[0]))
    intercept = model.intercept_[0]
    
    coef_df = pd.DataFrame(coefs, columns=['Feature', 'Coefficient'])
    coef_df.loc[len(coef_df)] = ['Intercept', intercept]
    
    for feat, coef in coefs:
        print(f"  {feat}: {coef:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    
    print("\n--- Interpretation ---")
    print("- player_pre_rating: A positive coefficient implies that a higher player rating increases log-odds of a win.")
    print("- opponent_pre_rating: A negative coefficient implies that a stronger opponent decreases the log-odds of a win.")
    print("- By using both explicitly, the model learns the exact asymmetric weight for each, rather than forcing a 1:1 ratio like 'rating_diff' does.")
    
    # 7. Save outputs
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(out_dir, 'logistic_binary_two_ratings_metrics.csv')
    coefs_path = os.path.join(out_dir, 'logistic_binary_two_ratings_coefs.csv')
    
    metrics_df.to_csv(metrics_path, index=False)
    coef_df.to_csv(coefs_path, index=False)
    
    print("\n" + "="*50)
    print("SUMMARY & COMPARISON NOTE")
    print("="*50)
    print(f"Outputs saved to: {out_dir}")
    print(f"- {os.path.basename(metrics_path)}")
    print(f"- {os.path.basename(coefs_path)}")
    print("\n>> Comparison to Baseline 1 (rating_diff) <<")
    print("Review the Validation ROC-AUC and Test F1 printed above.")
    print("If these metrics improved over 0.5420 ROC-AUC and 0.2390 F1, the logistic model successfully capitalized on asymmetric weights (e.g., punishing high opponent ratings more than rewarding high player ratings).")
    print("If the metrics stayed the same or dropped, it confirms that linear models struggle without explicit feature engineering (like the direct `rating_diff` extraction), and that weak performance is largely due to needing richer rolling/historical context, not just a different rating representation.")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
