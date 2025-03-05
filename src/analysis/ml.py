import logging
from typing import Dict, List, Optional, Union, Tuple, Callable, Any

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from ..core.config import Config

logger = logging.getLogger(__name__)


class MLAnalyzer:

    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        self.cv_metrics = ["accuracy", "precision", "recall", "f1"]

    def run_ml_analysis(self, feature_matrix: np.ndarray, labels: np.ndarray) -> Dict:
        logger.info("Running machine learning analysis")

        if feature_matrix.shape[0] < 10 or feature_matrix.shape[1] < 1:
            logger.warning(
                f"Not enough samples or features for ML analysis: {feature_matrix.shape}"
            )
            return {"error": "Not enough samples or features for ML analysis"}

        if len(np.unique(labels)) < 2:
            logger.warning(f"Not enough classes for ML analysis: {np.unique(labels)}")
            return {"error": "Not enough classes for ML analysis"}

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)

        self._analyze_with_model(
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.clustering.random_seed,
                n_jobs=-1,
            ),
            scaled_features,
            labels,
        )

        self._analyze_with_model(
            "svm",
            SVC(
                kernel="rbf",
                random_state=self.config.clustering.random_seed,
                probability=True,
            ),
            scaled_features,
            labels,
        )

        self._analyze_feature_importance(scaled_features, labels)

        return self.results

    def _analyze_with_model(
        self, model_name: str, model, features: np.ndarray, labels: np.ndarray
    ) -> None:
        logger.info(f"Running {model_name} analysis")

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.config.clustering.random_seed
        )

        model_results = {}

        for metric in self.cv_metrics:
            scores = cross_val_score(model, features, labels, cv=cv, scoring=metric)
            model_results[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "values": [float(v) for v in scores],
            }

        if len(np.unique(labels)) == 2:
            try:
                roc_auc = cross_val_score(
                    model, features, labels, cv=cv, scoring="roc_auc"
                )
                model_results["roc_auc"] = {
                    "mean": float(np.mean(roc_auc)),
                    "std": float(np.std(roc_auc)),
                    "values": [float(v) for v in roc_auc],
                }
            except Exception as e:
                logger.warning(f"Error calculating ROC AUC: {e}")

        self.results[model_name] = model_results

        if model_name == "random_forest":
            try:
                model.fit(features, labels)

                if hasattr(model, "feature_importances_"):
                    feature_importance = model.feature_importances_
                    self.results[model_name]["feature_importance"] = [
                        float(importance) for importance in feature_importance
                    ]
            except Exception as e:
                logger.warning(f"Error extracting feature importance: {e}")

    def _analyze_feature_importance(
        self, features: np.ndarray, labels: np.ndarray
    ) -> None:
        logger.info("Running feature importance analysis")

        if features.shape[1] <= 1:
            return

        try:
            selector = SelectFromModel(
                RandomForestClassifier(
                    n_estimators=100, random_state=self.config.clustering.random_seed
                ),
                threshold="median",
            )

            selector.fit(features, labels)
            support = selector.get_support()

            self.results["feature_selection"] = {
                "selected_indices": [int(i) for i, s in enumerate(support) if s],
                "n_selected": int(np.sum(support)),
            }

            if (
                hasattr(self.config, "feature_names")
                and len(self.config.feature_names) == features.shape[1]
            ):
                selected_names = [
                    self.config.feature_names[i] for i, s in enumerate(support) if s
                ]
                self.results["feature_selection"]["selected_names"] = selected_names

        except Exception as e:
            logger.warning(f"Error in feature importance analysis: {e}")
