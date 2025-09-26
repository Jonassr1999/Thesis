from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
def train_decision_tree(df, n_iter=20, random_state=42):
    X = df.drop("Aktiv", axis=1)
    y = df["Aktiv"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Parameterraum für RandomizedSearch
    param_dist = {
        "max_depth": [3, 4, 5, 6, 7, None],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [None, "sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    }

    base_clf = DecisionTreeClassifier(random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    random_search.fit(X, y)

    best_clf = random_search.best_estimator_
    best_params = random_search.best_params_  #nur die Parameter als dict
        # Optional für Debug:
    #st.subheader("Beste Hyperparameter")
    #st.json(best_params)
    # Cross-Validation mit dem besten Modell für finale Metriken und letzte Fold-Ergebnisse
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    last_test_idx = None
    last_test_true = None
    last_test_pred = None
    last_X_test = None
    last_X_train = None

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        last_test_idx = test_idx
        last_test_true = y_test
        last_test_pred = y_pred
        last_X_test = X_test
        last_X_train = X_train


    # Wahrscheinlichkeiten der positiven Klasse (für Schwellenwert-Vergleich)
    y_proba_last = best_clf.predict_proba(last_X_test)[:, 1]
    cm = confusion_matrix(last_test_true, last_test_pred)

    metrics = {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1_scores),
    }



    return {
    "model": best_clf,
    "metrics": {
        "accuracy": sum(accuracies) / len(accuracies),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1_scores) / len(f1_scores)
    },
    "X_test": last_X_test,
    "X_train": last_X_train,
    "y_test": last_test_true,
    "y_pred": last_test_pred,
    "y_proba": y_proba_last,
    "conf_matrix": cm,
    "last_test_idx": last_test_idx,
    "best_params": best_params
}
