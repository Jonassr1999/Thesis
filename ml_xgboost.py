import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def run_xgboost_classification(
    df: pd.DataFrame,
    target_col: str = "Aktiv",
    n_splits: int = 5,
    random_state: int = 42,
    numeric_only: bool = True,
    auto_scale_pos_weight: bool = True,
    xgb_kwargs: dict | None = None,
    n_iter: int = 20,
    plot_avg_loss_curves: bool = True,
):
    if numeric_only:
        df_proc = df.select_dtypes(exclude=["object"]).copy()
    else:
        df_proc = pd.get_dummies(df, drop_first=False)

    if target_col not in df_proc.columns:
        raise ValueError(f"Zielspalte '{target_col}' fehlt nach Preprocessing.")

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]

    base_params = dict(
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )
    if xgb_kwargs:
        base_params.update(xgb_kwargs)

    if auto_scale_pos_weight:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        if pos > 0:
            base_params["scale_pos_weight"] = neg / pos

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.3, 0.5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    model = xgb.XGBClassifier(**base_params)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    random_search.fit(X, y)
    best_params = random_search.best_params_
    #st.subheader("Beste Hyperparameter")
    #st.json(best_params)

    best_params_full = base_params.copy()
    best_params_full.update(best_params)
    num_boost_round = best_params_full.pop("n_estimators")

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    all_train_losses, all_val_losses = [], []

    last_test_idx = None
    last_test_pred = None
    last_test_true = None
    last_X_test = None
    last_X_train = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, "train"), (dval, "eval")]

        evals_result = {}

        booster = xgb.train(
            params=best_params_full,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False,
            evals_result=evals_result,
        )

        # Bestes Boosting-Intervall sicher bestimmen
        best_iter = (
            booster.best_iteration + 1
            if hasattr(booster, "best_iteration") and booster.best_iteration is not None
            else num_boost_round
        )

        # Vorhersage mit iteration_range (Obergrenze exklusiv)
        # Innerhalb der Fold-Schleife, direkt nach y_pred erzeugen:
        y_pred_proba = booster.predict(dval, iteration_range=(0, best_iter))
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # --- Neu: Wahrscheinlichkeiten pro Fold speichern ---
        if "y_prob_folds" not in locals():
            y_prob_folds = []
        y_prob_folds.append(y_pred_proba)

        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))

        all_train_losses.append(evals_result["train"]["logloss"])
        all_val_losses.append(evals_result["eval"]["logloss"])

        last_test_idx = X_val.index
        last_test_pred = y_pred
        last_test_true = y_val
        last_X_test = X_val
        last_X_train = X_train

    cm = confusion_matrix(last_test_true, last_test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Inaktiv (0)", "Aktiv (1)"],
        columns=["Vorhergesagt Inaktiv (0)", "Vorhergesagt Aktiv (1)"],
    )

    if plot_avg_loss_curves and all_train_losses and all_val_losses:
        max_len = max(len(l) for l in all_train_losses)
        mean_train_loss = np.nanmean(
            [np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in all_train_losses], axis=0)
        mean_val_loss = np.nanmean(
            [np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in all_val_losses], axis=0)

    return {
        "model": booster,
        "metrics": {
            "accuracy": float(np.mean(accuracies)),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1_scores)),
        },
        "conf_matrix_df": cm_df,
        "X_test": last_X_test,
        "X_train": last_X_train,
        "y_test": last_test_true,
        "y_pred": last_test_pred,
        "last_test_idx": last_test_idx,
        "last_test_pred": last_test_pred,
        "best_params": best_params,
        "y_prob_folds": y_prob_folds,
        "loss_curves": {
        "train_loss": all_train_losses,
        "val_loss": all_val_losses
    },
    "train_loss": all_train_losses,   
    "val_loss": all_val_losses         
}
