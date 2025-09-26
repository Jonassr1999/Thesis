import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from scipy.stats import randint, uniform

def run_lightgbm_classification(df: pd.DataFrame, n_iter: int = 30, random_state: int = 42):
    """
    Führt LightGBM Klassifikation mit Hyperparameter-Tuning (RandomizedSearchCV) durch
    und sammelt Trainings- und Validierungs-Loss-Werte sowie Wahrscheinlichkeiten pro Fold.

    Parameter:
    - df: Pandas DataFrame mit allen Eingabedaten
    - n_iter: Anzahl der Parameterkombinationen für Random Search
    - random_state: Reproduzierbarkeit

    Gibt zurück:
    - model: Finales bestes Modell
    - metrics: Durchschnittliche Metriken über die Folds
    - conf_matrix_df: Konfusionsmatrix vom letzten Fold
    - X_train, X_test, y_train, y_test, y_pred: Testdaten vom letzten Fold
    - y_prob_folds: Wahrscheinlichkeiten für Klasse 1 pro Fold (für ROC)
    - train_loss, val_loss: Listen mit den Loss-Kurven pro Fold
    """
    df = df.select_dtypes(exclude=["object"])

    X = df.drop(columns=["Aktiv"])
    y = df["Aktiv"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Hyperparameter-Raum
    param_dist = {
        "n_estimators": randint(100, 500),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 15),
        "num_leaves": randint(20, 150),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 1.0),
        "reg_lambda": uniform(0, 1.0)
    }

    base_model = lgb.LGBMClassifier(random_state=random_state)

    # 1. Hyperparameter-Tuning mit RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=skf,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X, y)
    best_model = search.best_estimator_
    best_params = search.best_params_

    # 2. Cross-Validation mit den besten Parametern und Loss-Tracking
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    all_train_losses, all_val_losses = [], []
    
    y_prob_folds = []  # Neu: Wahrscheinlichkeiten pro Fold
    last_test_idx = last_test_pred = last_test_true = None
    last_X_test = last_X_train = None

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        evals_result = {}

        best_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='binary_logloss',
            callbacks=[lgb.record_evaluation(evals_result)]
        )

        all_train_losses.append(evals_result['training']['binary_logloss'])
        all_val_losses.append(evals_result['valid_1']['binary_logloss'])

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeiten für Klasse 1
        y_prob_folds.append(y_prob)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        last_test_idx = test_idx
        last_test_pred = y_pred
        last_test_true = y_test
        last_X_test = X_test
        last_X_train = X_train

    avg_metrics = {
        "accuracy": float(np.mean(accuracies)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1_scores)),
    }

    cm = confusion_matrix(last_test_true, last_test_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=["Tatsächlich Inaktiv (0)", "Tatsächlich Aktiv (1)"], 
        columns=["Vorhergesagt Inaktiv (0)", "Vorhergesagt Aktiv (1)"]
    )

    return {
        "model": best_model,
        "best_params": best_params,
        "metrics": avg_metrics,
        "conf_matrix_df": cm_df,
        "X_test": last_X_test,
        "X_train": last_X_train,
        "y_test": last_test_true,
        "y_pred": last_test_pred,
        "y_prob_folds": y_prob_folds,  
        "last_test_idx": last_test_idx,
        "last_test_pred": last_test_pred,
        "train_loss": all_train_losses,
        "val_loss": all_val_losses
    }
