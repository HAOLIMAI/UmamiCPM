# =============================================================================
# Stacking ensemble for umami peptide prediction
# Trained on Pep2149 dataset, validated on independent dataset
# Three meta-models (Logistic Regression, Random Forest, GBDT) with GridSearchCV
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef


# ------------------------------
# 1. Load training and validation data
# ------------------------------
# Please modify the file paths according to your local directory.
# For reproducibility, we use relative paths (data files in the same folder as this script).
train_file = "./Training dataset.xlsx"   # Training dataset
valid_file = "./Independent validation dataset.xlsx"   # Independent validation dataset

df_train = pd.read_excel(train_file, sheet_name=0)
df_valid = pd.read_excel(valid_file, sheet_name=0)

print("Training set columns:", df_train.columns.tolist())
print("Validation set columns:", df_valid.columns.tolist())

# Features and labels for training
X_train = df_train[['FRL', 'YYDS', 'TPDM', 'MRNN']].values
y_train = df_train['Real Label'].values

# Features, labels, and peptide names for validation
X_valid = df_valid[['FRL', 'YYDS', 'TPDM', 'MRNN']].values
y_valid = df_valid['Real Label'].values
peptide_names_valid = df_valid['PepName'].values

print(f"Training set size: {len(y_train)} (umami: {sum(y_train == 1)}, non-umami: {sum(y_train == 0)})")
print(f"Validation set size: {len(y_valid)} (umami: {sum(y_valid == 1)}, non-umami: {sum(y_valid == 0)})")

# ------------------------------
# 2. Define meta‑models with hyperparameter grids
# ------------------------------
lr_params = {'C': [0.01, 0.1, 1, 10, 100]}
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
gbdt_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

models = {
    "LogisticRegression": GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        lr_params,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    ),
    "RandomForest": GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    ),
    "GradientBoosting": GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gbdt_params,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
}

# ------------------------------
# 3. Perform grid search with 5‑fold cross‑validation on training set
# ------------------------------
print("\n" + "=" * 60)
print("GridSearchCV 5‑fold cross‑validation results (training set)")
print("=" * 60)

for name, gs in models.items():
    gs.fit(X_train, y_train)
    print(f"{name:20s} | Best CV AUC = {gs.best_score_:.4f} | Best params: {gs.best_params_}")

# ------------------------------
# 4. Evaluate the best models on the independent validation set
# ------------------------------
print("\n" + "=" * 60)
print("Independent validation set evaluation (with optimal hyperparameters)")
print("=" * 60)

all_probs_valid = {}

for name, gs in models.items():
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_valid)
    y_proba = best_model.predict_proba(X_valid)[:, 1]
    acc = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_proba)
    mcc = matthews_corrcoef(y_valid, y_pred)
    print(f"\n--- {name} (best params: {gs.best_params_}) ---")
    print(f"ACC:     {acc:.4f}")
    print(f"AUC:     {auc:.4f}")
    print(f"MCC:     {mcc:.4f}")
    print("Classification report:")
    print(classification_report(y_valid, y_pred, target_names=['non-umami', 'umami']))
    print("Confusion matrix:")
    print(confusion_matrix(y_valid, y_pred))
    all_probs_valid[name] = y_proba

# ------------------------------
# 5. Extract feature weights/importances from the best models
# ------------------------------
lr_best = models["LogisticRegression"].best_estimator_
rf_best = models["RandomForest"].best_estimator_
gbdt_best = models["GradientBoosting"].best_estimator_

coef = lr_best.coef_[0]
rf_importances = rf_best.feature_importances_
gbdt_importances = gbdt_best.feature_importances_

weight_data = {
    "LogisticRegression_Coefficient": coef,
    "RandomForest_Importance": rf_importances,
    "GradientBoosting_Importance": gbdt_importances
}

feature_names = ['FRL', 'YYDS', 'TPDM', 'MRNN']
weight_df = pd.DataFrame(weight_data, index=feature_names)
print("\nFeature weights / importances (based on training set best models):")
print(weight_df)

# ------------------------------
# 6. Save validation predictions to Excel
# ------------------------------
output_df = pd.DataFrame()
output_df['Peptide'] = peptide_names_valid
output_df['True_Label'] = y_valid
output_df['FRL'] = X_valid[:, 0]
output_df['YYDS'] = X_valid[:, 1]
output_df['TPDM'] = X_valid[:, 2]
output_df['MRNN'] = X_valid[:, 3]

for name in models.keys():
    output_df[f'{name}_Probability_Umami'] = all_probs_valid[name]
    output_df[f'{name}_Predicted_Label'] = (all_probs_valid[name] >= 0.5).astype(int)

output_excel = "umami_independent_validation_results.xlsx"
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    output_df.to_excel(writer, sheet_name='Validation_Predictions', index=False)
    weight_df.to_excel(writer, sheet_name='Feature_Weights_Importance')

print(f"\nValidation predictions saved to: {output_excel}")
print("Excel file contains two sheets: 'Validation_Predictions' and 'Feature_Weights_Importance'.")