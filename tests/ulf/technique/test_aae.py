"""Tests for aae technique."""

from ulf.technique.aae import AAE


def test_aae_fit(
    simple_dense_decoder,
    simple_dense_model,
    simple_dense_noice_discriminator,
    digits_dataset_train,
    digits_dataset_val,
):
    """Test for AAE::fit method."""
    data, _ = digits_dataset_train
    data_val, target_val = digits_dataset_val

    aae = AAE(simple_dense_model, simple_dense_decoder, simple_dense_noice_discriminator)
    aae.compile("adam", "adam", "binary_crossentropy", "mse")
    history = aae.fit(x=data, validation_data=(data_val,), epochs=12)

    assert history.history["val_reconstruction_loss"][-1] < 0.1
