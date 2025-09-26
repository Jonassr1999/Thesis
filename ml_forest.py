from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer
)

def train_random_forest(df):
    X = df.drop("Aktiv", axis=1)
    y = df["Aktiv"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 🧪 RandomizedSearch: Parameter-Raum
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
        "max_features": ['sqrt', 'log2', None]
    }

    base_model = RandomForestClassifier(random_state=42)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring=make_scorer(f1_score),
        cv=skf,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )

    # Suche beste Parameter via CV
    search.fit(X, y)
    clf = search.best_estimator_
    best_params = search.best_params_

    # Optional für Debug:
    #st.subheader("Beste Hyperparameter")
    #st.json(best_params)

    # Evaluation auf Basis letzten Folds
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    last_test_idx = None

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        last_test_idx = test_idx
        last_test_true = y_test
        last_test_pred = y_pred
        last_X_test = X_test
        last_X_train = X_train

    return {
        "accuracy": sum(accuracies) / len(accuracies),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1_scores) / len(f1_scores),
        "clf": clf,
        "X": X,
        "X_test": last_X_test,
        "X_train": last_X_train,
        "last_test_idx": last_test_idx,
        "last_test_true": last_test_true,
        "last_test_pred": last_test_pred,
        "best_params": best_params
    }
