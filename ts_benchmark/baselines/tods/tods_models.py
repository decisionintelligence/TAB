# -*- coding: utf-8 -*-
"""
TODS-compatible baselines implemented directly on top of pyod.

The original implementation wrapped these detectors through the TODS / d3m
primitive stack, which is abandoned and only compatible with Python 3.8.
Every detector exposed here was already a thin wrapper around a pyod model,
so the wrappers below call pyod directly while keeping the same model names,
default hyperparameters and output conventions. Existing configs and scripts
(e.g. ``--model-name "tods.lofski"``) keep working unchanged.

Note: ``lstmodetectorski`` is unavailable in this implementation — the LSTM
outlier detector was a TensorFlow model internal to TODS with no pyod
equivalent. Use the deep-learning baselines of the benchmark instead.
"""
import logging

import numpy as np
import pandas as pd

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

from ts_benchmark.baselines.tods.pyod_core.PCA import PCA as _WindowPCA

logger = logging.getLogger(__name__)


class PCAODetector(_WindowPCA):
    """
    Sliding-window PCA detector.

    Same algorithm and defaults as TODS' ``PCAODetectorSKI`` (window_size=10,
    step_size=1): the series is unfolded into overlapping windows and pyod's
    PCA detector is applied to the resulting matrix.
    """

    def __init__(self, window_size: int = 10, **kwargs):
        super().__init__(window_size=window_size, **kwargs)


# [exported name (kept from the original TODS wrappers), model class, required params]
TODS_MODELS = [
    ["IsolationForestSKI", IForest, {}],
    ["KNNSKI", KNN, {}],
    ["AutoEncoderSKI", AutoEncoder, {}],
    ["LOFSKI", LOF, {}],
    ["OCSVMSKI", OCSVM, {}],
    ["HBOSSKI", HBOS, {}],
    ["LODASKI", LODA, {}],
    ["PCAODetectorSKI", PCAODetector, {}],
    ["COFSKI", COF, {}],
    ["CBLOFSKI", CBLOF, {}],
]


class TodsModelAdapter:
    """
    Adapts pyod detection models to meet the requirements of prediction strategies.
    """

    def __init__(
        self,
        model_name: str,
        model_class: object,
        model_args: dict,
    ):
        """
        Initialize the model adapter object.

        :param model_name: Model name.
        :param model_class: pyod model class.
        :param model_args: Model initialization parameters.
        """
        self.model = None
        self.model_class = model_class
        self.model_args = model_args
        self.model_name = model_name

    def detect_fit(self, series: pd.DataFrame, label: pd.DataFrame) -> object:
        """
        Fit a suitable pyod model on time series data.

        :param series: Time series data.
        :param label: Label data (ignored, unsupervised models).
        :return: The fitted model object.
        """
        self.model = self.model_class(**self.model_args)
        X = series.values
        self.model.fit(X)

        return self.model

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        """
        Calculate anomaly scores using the fitted pyod model.

        :param train: Data used to calculate scores.
        :return: Anomaly score array.
        """
        X = train.values
        prediction_score = self.model.decision_function(X)
        if isinstance(prediction_score, tuple):
            # collective (window-based) detectors return (scores, left_inds, right_inds)
            prediction_score = prediction_score[0]
        prediction_score = np.asarray(prediction_score).reshape(-1)

        return prediction_score, prediction_score

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        """
        Use the fitted pyod model for anomaly detection and generate labels.

        :param train: Data used for anomaly detection.
        :return: Anomaly label array.
        """
        X = train.values
        prediction_labels = self.model.predict(X)
        if isinstance(prediction_labels, tuple):
            # collective (window-based) detectors return (labels, left_inds, right_inds)
            prediction_labels = prediction_labels[0]
        prediction_labels = np.asarray(prediction_labels).reshape(-1)

        return prediction_labels, prediction_labels

    def __repr__(self):
        """
        Returns a string representation of the model name.
        """
        return self.model_name


def generate_model_factory(
    model_name: str,
    model_class: object,
    required_args: dict,
) -> object:
    """
    Generate model factory information for creating model adapters.

    :param model_name: Model name.
    :param model_class: pyod model class.
    :param required_args: Required parameters for model initialization.
    :return: A dictionary containing the model factory and required parameters.
    """

    def model_factory(**kwargs) -> object:
        """
        Model factory, used to create model adapter objects.

        :param kwargs: Model initialization parameters.
        :return: Model adapter object.
        """
        return TodsModelAdapter(
            model_name,
            model_class,
            kwargs,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


# Generate model factories for each model class and required parameters in TODS_MODELS
# and add them to global variables under their historical names (e.g. "lofski")
for model_name, model_class, required_args in TODS_MODELS:
    globals()[model_name.lower()] = generate_model_factory(
        model_name, model_class, required_args
    )


def _lstmodetector_factory(**kwargs):
    raise NotImplementedError(
        "lstmodetectorski is not available: the TODS LSTM outlier detector relied on "
        "the abandoned d3m/TensorFlow stack (Python 3.8 only) and has no pyod "
        "equivalent. Use the deep-learning baselines of the benchmark instead."
    )


lstmodetectorski = {
    "model_factory": _lstmodetector_factory,
    "required_hyper_params": {},
}
