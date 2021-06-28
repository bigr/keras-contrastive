"""Tests for gan technique."""

from ulf.technique.gan import GAN


def test_gan_fit(
    simple_dense_generator, simple_dense_discriminator, digits_dataset_train, digits_dataset_val
):
    """Test for GAN::fit method."""
    data, _ = digits_dataset_train
    data_val, target_val = digits_dataset_val

    gan = GAN(simple_dense_generator, simple_dense_discriminator)
    gan.compile("adam", "adam", "binary_crossentropy", "binary_crossentropy")
    history = gan.fit(x=data, validation_data=(data_val,), epochs=16)

    d_loss = history.history["val_discriminator_loss"][-1]
    g_loss = history.history["val_generator_loss"][-1]

    assert d_loss < 1.25
    assert g_loss < 1.25

    assert 0.25 < g_loss / d_loss < 4
