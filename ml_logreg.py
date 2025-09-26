import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

def run_logistic_regression_classification(
    df: pd.DataFrame,
    target_col: str = "Aktiv",
    n_splits: int = 5,
    random_state: int = 42,
    numeric_only: bool = False,
    scale_features: bool = True,
    bayes_search_iter: int = 20,
    verbose: bool = False,
):
    """
    Klassifikation mit logistischer Regression und BayesSearchCV für Hyperparameter-Tuning.
    Gibt jetzt auch Wahrscheinlichkeiten pro Fold zurück (für ROC-Kurven).
    """
    if numeric_only:
        df_proc = df.select_dtypes(exclude=["object"]).copy()
    else:
        df_proc = pd.get_dummies(df, drop_first=False)

    if target_col not in df_proc.columns:
        raise ValueError(f"Zielspalte '{target_col}' fehlt nach Preprocessing.")

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]

    scaler = StandardScaler() if scale_features else None

    param_space = {
        "penalty": Categorical(['l2', None]),
        "C": Real(1e-3, 1e3, prior='log-uniform'),
        "solver": Categorical(['lbfgs', 'newton-cg', 'saga']),
        "max_iter": Categorical([1000]),
        "class_weight": Categorical(['balanced', None]),
    }

    base_model = LogisticRegression(random_state=random_state)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    bayes_search = BayesSearchCV(
        base_model,
        param_space,
        n_iter=bayes_search_iter,
        scoring=make_scorer(f1_score),
        cv=skf,
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
        verbose=0,
    )

    bayes_search.fit(X, y)
    model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_

    if verbose:
        print("Beste Hyperparameter:", best_params)

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    y_prob_folds = []  # Neu: Wahrscheinlichkeiten pro Fold

    last_test_idx = last_test_pred = last_test_true = None
    last_X_test = last_X_train = None

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if scaler:
            X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeiten für Klasse 1
        y_prob_folds.append(y_prob)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        last_test_idx = X_test.index
        last_test_true = y_test
        last_test_pred = y_pred
        last_X_test = X_test
        last_X_train = X_train

    cm = confusion_matrix(last_test_true, last_test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Inaktiv (0)", "Aktiv (1)"],
        columns=["Vorhergesagt Inaktiv (0)", "Vorhergesagt Aktiv (1)"],
    )

    return {
        "model": model,
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
        "y_prob_folds": y_prob_folds, 
        "last_test_idx": last_test_idx,
        "last_test_pred": last_test_pred,
        "best_params": best_params,
    }
