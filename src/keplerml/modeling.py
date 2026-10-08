"""Model definitions, cross-validation, and evaluation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config

SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}
PRIMARY_METRIC = "roc_auc"


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Wrap an estimator in the shared imputation and scaling steps.

    Median imputation is robust to the long tails in the transit measurements.
    Scaling is required by logistic regression, k-nearest neighbours, and the
    RBF support vector machine, and is harmless for the forest.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def model_specifications(random_state: int = config.RANDOM_STATE) -> dict[str, dict]:
    """Return the estimator and hyperparameter grid for each candidate model."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
        "k-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "model__n_neighbors": [5, 11, 21, 35],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            "param_grid": {
                "model__max_depth": [None, 12, 24],
                "model__min_samples_leaf": [1, 3, 8],
                "model__max_features": ["sqrt", 0.3],
            },
        },
        "Support Vector Machine": {
            "estimator": SVC(
                kernel="rbf",
                class_weight="balanced",
                cache_size=1000,
                random_state=random_state,
            ),
            "param_grid": {
                "model__C": [1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01, 0.1],
            },
        },
    }


@dataclass
class SearchResult:
    """Outcome of one hyperparameter search."""

    name: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    cv_scores: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0


def run_model_search(
    name: str,
    specification: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cv: StratifiedGroupKFold,
    n_jobs: int = -1,
) -> SearchResult:
    """Grid-search one model and report grouped cross-validation scores.

    The support vector machine is searched without Platt scaling, because the
    ranking metrics only need the decision function and calibration would
    multiply the cost of every fit. Probabilities are enabled once, on the
    winning configuration.
    """
    estimator = clone(specification["estimator"])
    search = GridSearchCV(
        estimator=build_pipeline(estimator),
        param_grid=specification["param_grid"],
        scoring=SCORING,
        refit=PRIMARY_METRIC,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
    )

    started = time.perf_counter()
    search.fit(X, y, groups=groups)
    elapsed = time.perf_counter() - started

    best_index = search.best_index_
    cv_scores = {
        metric: float(search.cv_results_[f"mean_test_{metric}"][best_index])
        for metric in SCORING
    }
    cv_scores[f"{PRIMARY_METRIC}_std"] = float(
        search.cv_results_[f"std_test_{PRIMARY_METRIC}"][best_index]
    )

    best_estimator = search.best_estimator_
    if isinstance(best_estimator.named_steps["model"], SVC):
        best_estimator = clone(best_estimator)
        best_estimator.set_params(model__probability=True)
        best_estimator.fit(X, y)

    return SearchResult(
        name=name,
        best_estimator=best_estimator,
        best_params={k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        cv_scores=cv_scores,
        fit_seconds=elapsed,
    )


def predicted_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores, preferring probabilities when available."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def evaluate(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Score a fitted estimator on a held-out set at a fixed decision threshold."""
    scores = predicted_scores(estimator, X)
    predictions = (scores >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
    }
    if scores.min() >= 0.0 and scores.max() <= 1.0:
        metrics["brier"] = brier_score_loss(y, scores)
    return {key: float(value) for key, value in metrics.items()}


def out_of_fold_scores(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cv: StratifiedGroupKFold,
    n_jobs: int = -1,
) -> np.ndarray:
    """Cross-validated positive-class scores for every training row."""
    method = "predict_proba" if hasattr(estimator, "predict_proba") else "decision_function"
    predictions = cross_val_predict(
        estimator, X, y, groups=groups, cv=cv, method=method, n_jobs=n_jobs
    )
    return predictions[:, 1] if predictions.ndim == 2 else predictions


def threshold_for_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_precision: float,
) -> tuple[float, float]:
    """Smallest threshold that reaches `target_precision`, with its recall.

    Returns the maximum available precision and its threshold if the target is
    not reachable.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]

    eligible = precision >= target_precision
    if not eligible.any():
        best = int(np.argmax(precision))
        return float(thresholds[best]), float(recall[best])

    best = int(np.argmax(np.where(eligible, recall, -np.inf)))
    return float(thresholds[best]), float(recall[best])


def summarize_searches(results: list[SearchResult]) -> pd.DataFrame:
    """Collect cross-validation scores from several searches into one table."""
    rows = []
    for result in results:
        row = {"model": result.name}
        row.update(result.cv_scores)
        row["fit_seconds"] = round(result.fit_seconds, 1)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values(PRIMARY_METRIC, ascending=False)
    )
