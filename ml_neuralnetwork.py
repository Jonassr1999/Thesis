import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import uniform
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def run_neural_network_classification(
    df: pd.DataFrame,
    target_col: str = "Aktiv",
    n_splits: int = 5,
    random_state: int = 42,
    hidden_layers=(64, 32),
    max_iter=500,
    numeric_only: bool = True,
    scale_features: bool = True,
    mlp_kwargs: dict | None = None,
    plot_avg_loss_curves: bool = True,
    use_random_search: bool = True,
    n_iter_search: int = 20
):
    if numeric_only:
        df_proc = df.select_dtypes(exclude=["object"]).copy()
    else:
        df_proc = pd.get_dummies(df, drop_first=False)

    if target_col not in df_proc.columns:
        raise ValueError(f"Zielspalte '{target_col}' fehlt nach Preprocessing.")

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]

    # Basisparameter für das MLP-Modell
    base_params = dict(
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
    )
    if mlp_kwargs:
        base_params.update(mlp_kwargs)

    # RandomizedSearchCV
    if use_random_search:
        search_space = {
            "hidden_layer_sizes": [(50,), (100,), (64, 32), (128, 64), (128, 64, 32)],
            "activation": ["relu", "tanh"],
            "alpha": uniform(0.0001, 0.01),
            "learning_rate_init": uniform(0.001, 0.01),
        }
        base_model = MLPClassifier(**base_params)
        rs = RandomizedSearchCV(
            base_model,
            search_space,
            n_iter=n_iter_search,
            cv=3,
            random_state=random_state,
            n_jobs=-1,
            scoring="f1"
        )
        rs.fit(X, y)
        best_params = rs.best_params_
        base_params.update(best_params)

    # Finales MLP-Modell
    model = MLPClassifier(**base_params)

    # Pipeline mit (optionalem) Scaler + SMOTE + Modell
    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("mlp", model))
    pipeline = Pipeline(steps)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    all_train_losses = []

    last_test_idx = last_test_pred = last_test_true = last_X_test = last_X_train = None

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Pipeline fit + prediction
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        # MLPClassifier steckt als letztes Element in der Pipeline – loss_curve_
        clf = pipeline.named_steps["mlp"]
        if hasattr(clf, "loss_curve_"):
            all_train_losses.append(clf.loss_curve_)

        last_test_idx = X_test.index
        last_test_pred = y_pred
        last_test_true = y_test
        last_X_test = X_test
        last_X_train = X_train

    cm = confusion_matrix(last_test_true, last_test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Inaktiv (0)", "Aktiv (1)"],
        columns=["Vorhergesagt Inaktiv (0)", "Vorhergesagt Aktiv (1)"],
    )

    return {
        "model": pipeline,
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
        "last_test_pred": last_test_pred,
        "last_test_idx": last_test_idx,
        "train_loss": all_train_losses,
        "val_loss": []  
    }