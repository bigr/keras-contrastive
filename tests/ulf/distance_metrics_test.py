import pytest
import tensorflow as tf

from keras_contrastive.distance_metrics import euclidean_dist


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            tf.constant([[1, 2, 3, 4]], dtype=tf.float32),
            tf.constant([[2, 3, 4, 5]], dtype=tf.float32),
            tf.constant([2], dtype=tf.float32),
        )
    ],
)
def test_euclidean_dist(a, b, expected):
    actual = euclidean_dist(a, b)
    assert actual.numpy() == expected.numpy()
