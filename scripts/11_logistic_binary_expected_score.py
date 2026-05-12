import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(base_dir, 'data', 'processed', 'features_v2_expected_score.csv')
    out_dir = os.path.join(base_dir, 'outputs', 'models')
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(data_file):
        print(f"Error: Could not find {data_file}")
        sys.exit(1)
        
    df = pd.read_csv(data_file)
    
    print("="*50)
    print("PHASE 9: LOGISTIC REGRESSION (EXPECTED SCORE)")
    print("="*50)
    
    # 1. Prepare the benchmark dataset
    df['event_end_date'] = pd.to_datetime(df['event_end_date'], errors='coerce')
    df = df.sort_values(by=['event_end_date', 'game_id']).reset_index(drop=True)
    
    # Subset features and target
    features_df = df[['expected_score_player', 'time_control']].copy()
    y = df['target_binary'].astype(int)
    
    # One-hot encode time_control
    X = pd.get_dummies(features_df, columns=['time_control'], drop_first=True)
    
    # 2. Time-based split (70% Train, 15% Validation, 15% Test)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    # 3. Fit the benchmark model
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
    
    # 4. Report benchmark metrics
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
    
    # 5. Report interpretability outputs
    print("\n--- Model Coefficients ---")
    coefs = list(zip(X.columns, model.coef_[0]))
    intercept = model.intercept_[0]
    
    coef_df = pd.DataFrame(coefs, columns=['Feature', 'Coefficient'])
    coef_df.loc[len(coef_df)] = ['Intercept', intercept]
    
    for feat, coef in coefs:
        print(f"  {feat}: {coef:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    
    print("\n--- Interpretation ---")
    print("- expected_score_player: Because the raw correlation was slightly negative, pay close attention to this sign. If it remains negative, it indicates the model believes higher expected scores paradoxically predict lower actual win odds in this specific dataset (e.g. high-rated focal players underperforming heavily in these specific tournaments).")
    print("- time_control_*: How specific time controls shift the baseline odds of winning.")
    
    # 6. Save outputs
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(out_dir, 'logistic_binary_expected_score_metrics.csv')
    coefs_path = os.path.join(out_dir, 'logistic_binary_expected_score_coefs.csv')
    
    metrics_df.to_csv(metrics_path, index=False)
    coef_df.to_csv(coefs_path, index=False)
    
    # 7. Comparison note
    print("\n" + "="*50)
    print("SUMMARY & COMPARISON NOTE")
    print("="*50)
    print(f"Outputs saved to: {out_dir}")
    print(f"- {os.path.basename(metrics_path)}")
    print(f"- {os.path.basename(coefs_path)}")
    
    print("\n>> Comparison to Best Logistic Baseline (Raw Ratings) <<")
    print("Previous Baseline 2 (player_pre_rating + opponent_pre_rating):")
    print("   - Validation ROC-AUC: 0.6253 | F1: 0.5088")
    print("   - Test ROC-AUC:       0.5970 | F1: 0.4848")
    print(f"\nNew Expected Score Logistic Model:")
    print(f"   - Validation ROC-AUC: {val_m[4]:.4f} | F1: {val_m[3]:.4f}")
    print(f"   - Test ROC-AUC:       {test_m[4]:.4f} | F1: {test_m[3]:.4f}")
    
    print("\nVerdict:")
    if test_m[4] > 0.5970 and test_m[3] > 0.4848:
        print("The engineered expected_score_player feature successfully outperformed the raw ratings. This proves that embedding domain knowledge (Elo math) directly into the feature space helps linear algorithms find better optimization boundaries.")
    else:
        print("The expected_score_player feature did NOT outperform the raw ratings. Given the negative raw correlation discovered in Phase 8, this suggests the pure Elo formula does not perfectly map onto this specific dataset's actual outcomes (e.g., highly rated players might be heavily underperforming historically in these specific tournaments).")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
