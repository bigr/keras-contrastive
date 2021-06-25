"""Tests for vae technique."""

from tensorflow import keras
from tensorflow.keras import layers

from ulf.technique.vae import VAE


def test_vae_fit(
    simple_dense_decoder, simple_dense_model, digits_dataset_train, digits_dataset_val
):
    """Test for VAE::fit method."""
    data, _ = digits_dataset_train
    data_val, target_val = digits_dataset_val
    out = layers.Dense(8)(simple_dense_model.layers[-2].output)
    sigma_encoder = keras.Model(simple_dense_model.inputs, out)

    vae = VAE(simple_dense_model, sigma_encoder, simple_dense_decoder)
    vae.compile("adam", "mse")
    history = vae.fit(x=data, validation_data=(data_val,), epochs=18)

    assert history.history["val_loss"][-1] < 0.12
    assert history.history["val_kl_loss"][-1] < 0.009
