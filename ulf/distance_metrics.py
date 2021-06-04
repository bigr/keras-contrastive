"""Implements distance metrics, used mainly in contrastive losses."""
from functools import partial
from typing import Callable, Union

import tensorflow as tf
from tensorflow_addons.losses import metric_learning
from tensorflow_addons.utils.types import TensorLike

L2_DIST_NAMES = {"l2", "euc", "euclidean"}
SQUARED_L2_DIST_NAMES = {"squared-l2"}
ANGULAR_DIST_NAMES = {"angular", "cosine"}


def euclidean_dist(e1: TensorLike, e2: TensorLike, squared=False) -> TensorLike:
    """Euclidean distance."""
    dist = tf.maximum(tf.reduce_sum((e1 - e2) ** 2, axis=1, keepdims=True), 0.0)
    if not squared:
        error_mask = tf.math.less_equal(dist, 0.0)
        dist = tf.math.sqrt(dist + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16)
        dist = dist * tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32)
    return dist


def angular_dist(e1: TensorLike, e2: TensorLike) -> TensorLike:
    """Cosine distance."""
    e1n, e2n = tf.math.l2_normalize(e1, axis=1), tf.math.l2_normalize(e2, axis=1)
    dist = 1.0 - tf.reduce_sum(e1n * e2n, axis=1, keepdims=True)
    return tf.maximum(dist, 0.0)


def named_distance_metric_fn(
    distance_metric_fn: Union[str, Callable[[TensorLike, TensorLike], TensorLike]]
) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """Make distance metric function defined by it's name."""

    if isinstance(distance_metric_fn, str) and distance_metric_fn.lower() in L2_DIST_NAMES:
        return partial(euclidean_dist, squared=False)
    elif (
        isinstance(distance_metric_fn, str) and distance_metric_fn.lower() in SQUARED_L2_DIST_NAMES
    ):
        return partial(euclidean_dist, squared=True)
    elif isinstance(distance_metric_fn, str) and distance_metric_fn.lower() in ANGULAR_DIST_NAMES:
        return metric_learning.angular_distance
    elif callable(distance_metric_fn):
        return distance_metric_fn
    else:
        raise ValueError(f"Unknown distance_metric_fn {distance_metric_fn}")


def named_distance_metric_matrix_fn(
    distance_metric_fn: Union[str, Callable[[TensorLike], TensorLike]]
) -> Callable[[TensorLike], TensorLike]:
    """Make distance metric function producing distance matrix defined by it's name."""

    if isinstance(distance_metric_fn, str) and distance_metric_fn.lower() in L2_DIST_NAMES:
        return partial(metric_learning.pairwise_distance, squared=False)
    elif (
        isinstance(distance_metric_fn, str) and distance_metric_fn.lower() in SQUARED_L2_DIST_NAMES
    ):
        return partial(metric_learning.pairwise_distance, squared=True)
    elif isinstance(distance_metric_fn, str) and distance_metric_fn.lower() in ANGULAR_DIST_NAMES:
        return metric_learning.angular_distance
    elif callable(distance_metric_fn):
        return distance_metric_fn
    else:
        raise ValueError(f"Unknown distance_metric_fn {distance_metric_fn}")
