from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


class LOF:
    """
    LOF (Local Outlier Factor) model class, used for anomaly detection.

    LOF is a density based anomaly detection method used to identify data points with significantly different densities compared to their neighbors in a dataset.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params: dict = None,
        contamination: float = 0.1,
        n_jobs: int = 1,
    ):
        """
        Initialize the LOF model.

        :param n_neighbors: Used to calculate the number of neighbors in the LOF.
        :param algorithm: The algorithm used for LOF calculation.
        :param leaf_size: The leaf size used when constructing KD trees or ball trees.
        :param metric: The distance metric used to calculate distance.
        :param p: The parameter p in distance measurement.
        :param metric_params: Other parameters for distance measurement.
        :param contamination: The proportion of expected abnormal samples.
        :param n_jobs: The number of worker threads used for parallel computing.
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.model_name = "LOF"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required for the LOF model.

        :return: An empty dictionary indicating that the LOF model does not require additional hyperparameters.
        """
        return {}

    def detect_fit(self, X, y=None):
        """
        Train LOF models.

        :param X: Training data.
        :param y: Label data (optional).
        """
        pass

    def detect_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Use the LOF model to calculate anomaly scores.

        :param X: The data of the score to be calculated.
        :return: Anomaly score array.
        """
        X = X.values.reshape(-1, 1)

        self.detector_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            contamination=self.contamination,
            n_jobs=self.n_jobs,
        )
        self.detector_.fit(X=X)

        self.decision_scores_ = -self.detector_.negative_outlier_factor_

        score = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(self.decision_scores_.reshape(-1, 1))
            .ravel()
        )
        return score, score

    def detect_label(self, X: pd.DataFrame) -> np.ndarray:
        """
        Use LOF model for anomaly detection and generate labels.

        :param X: The data to be tested.
        :return: Anomaly label array.
        """
        X = X.values.reshape(-1, 1)

        self.detector_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            contamination=self.contamination,
            n_jobs=self.n_jobs,
        )
        self.detector_.fit(X=X)

        self.decision_scores_ = -self.detector_.negative_outlier_factor_

        score = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(self.decision_scores_.reshape(-1, 1))
            .ravel()
        )
        preds = (score > np.percentile(score, 100 * (1 - self.contamination))).astype(int)
        return preds, score

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name
