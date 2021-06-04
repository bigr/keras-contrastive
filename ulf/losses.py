"""Implements losses used in contrastive learning."""
from typing import Callable, Tuple, Union

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.utils.types import TensorLike

from keras_contrastive.distance_metrics import named_distance_metric_fn


class ContrastiveLoss:
    """Contrastive loss."""

    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: Union[str, Callable[[TensorLike, TensorLike], TensorLike]] = "L2",
        reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        pair_miner: Callable[[TensorLike], Tuple[TensorLike, TensorLike]] = lambda x: (
            x[:-1],
            x[1:],
        ),
    ):
        self._distance_metric = named_distance_metric_fn(distance_metric)
        self._pair_miner = pair_miner
        self._tfa_class = tfa.losses.ContrastiveLoss(margin, reduction)

    def __call__(self, y_true, y_pred):
        """Loss computation."""
        y_pred_a, y_pred_b = self._pair_miner(y_pred)
        y_true_a, y_true_b = self._pair_miner(y_true)
        distances = self._distance_metric(y_pred_a, y_pred_b)
        y_true_binary = y_true_a == y_true_b
        return self._tfa_class(y_true_binary, distances)
