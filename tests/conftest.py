"""Commons for testing."""

import random

import tensorflow as tf
from pytest import fixture
from sklearn.datasets import load_digits


@fixture(autouse=True)
def fix_random_seed():
    """Set random seed to constant."""
    random.seed(0)
    tf.random.set_seed(0)


@fixture
def digits_dataset():
    """Return toy dataset for tests (all sets)."""
    data, targets = load_digits(n_class=5, return_X_y=True)
    return data / 16.0, targets


@fixture
def digits_dataset_train(digits_dataset):
    """Return yoy dataset for tests (train set)."""
    data, targets = digits_dataset
    return data[256:768], targets[256:768]


@fixture
def digits_dataset_val(digits_dataset):
    """Return toy dataset for tests (validation set)."""
    data, targets = digits_dataset
    return data[128:256], targets[128:256]


@fixture
def digits_dataset_test(digits_dataset):
    """Return toy dataset for tests (test set)."""
    data, targets = digits_dataset
    return data[:128], targets[:128]
