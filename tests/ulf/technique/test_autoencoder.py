"""Tests for autoencoder technique."""

from ulf.technique.autoencoder import Autoencoder


def test_autoencoder_fit(
    simple_dense_decoder, simple_dense_model, digits_dataset_train, digits_dataset_val
):
    """Test for Autoencoder::fit method."""
    data, _ = digits_dataset_train
    data_val, target_val = digits_dataset_val
    autoencoder = Autoencoder(simple_dense_model, simple_dense_decoder)
    autoencoder.compile("adam", "mse")
    history = autoencoder.fit(x=data, validation_data=(data_val,), epochs=12)
    assert history.history["val_reconstruction_loss"][-1] < 0.09
