"""Tests for bigan technique."""

from ulf.technique.bigan import BiGAN


def test_bigan_fit(
    simple_dense_decoder,
    simple_dense_model,
    simple_dense_bi_discriminator,
    digits_dataset_train,
    digits_dataset_val,
):
    """Test for BiGAN::fit method."""
    data, _ = digits_dataset_train
    data_val, target_val = digits_dataset_val

    bigan = BiGAN(simple_dense_model, simple_dense_decoder, simple_dense_bi_discriminator)
    bigan.compile(
        "adam",
        "adam",
        "adam",
        "binary_crossentropy",
        "binary_crossentropy",
        "binary_crossentropy",
    )
    history = bigan.fit(x=data, validation_data=(data_val,), epochs=12)

    ds_loss = history.history["val_discriminator_loss"][-1]
    e_loss = history.history["val_encoder_loss"][-1]
    dc_loss = history.history["val_decoder_loss"][-1]

    assert ds_loss < 1.5
    assert e_loss < 1.5
    assert dc_loss < 1.5

    assert 0.25 < dc_loss / ds_loss < 4
    assert 0.25 < e_loss / ds_loss < 4
