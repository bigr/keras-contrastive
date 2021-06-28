"""Commons for testing."""

import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pytest import fixture
from sklearn.datasets import load_digits


@fixture(autouse=True)
def fix_random_seed():
    """Set random seed to constant."""
    random.seed(0)
    tf.random.set_seed(0)


@fixture
def simple_dense_model():
    """Return simple full connected keras model for tests."""
    inp = layers.Input((64,))
    out = layers.Dense(16, name="dense1", activation="relu")(inp)
    out = layers.Dense(8, name="dense2")(out)
    out = layers.Activation("linear", name="final_activation")(out)
    return keras.Model(inp, out)


@fixture
def simple_dense_decoder():
    """Return decoder model."""
    inp = layers.Input((8,))
    out = layers.Dense(64, name="decoder_dense1")(inp)
    out = layers.Activation("relu", name="decoder_final_activation")(out)
    return keras.Model(inp, out)


@fixture
def simple_dense_generator(simple_dense_decoder):
    """Return generator model."""
    inp = layers.Input((8,))
    out = layers.Dense(64, name="generator_dense1")(inp)
    out = layers.Activation("relu", name="generator_final_activation")(out)
    return keras.Model(inp, out)


@fixture
def simple_dense_discriminator():
    """Return discriminator model."""
    inp = layers.Input((64,))
    out = layers.Dense(16, name="discriminator_dense1", activation="relu")(inp)
    out = layers.Dense(1, name="discriminator_dense2")(out)
    out = layers.Activation("sigmoid", name="discriminator_final_activation")(out)
    return keras.Model(inp, out)


@fixture
def simple_dense_noice_discriminator():
    """Return latent vectors discriminator model."""
    inp = layers.Input((8,))
    out = layers.Dense(16, name="discriminator_dense1", activation="relu")(inp)
    out = layers.Dense(1, name="discriminator_dense2")(out)
    out = layers.Activation("sigmoid", name="discriminator_final_activation")(out)
    return keras.Model(inp, out)


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
