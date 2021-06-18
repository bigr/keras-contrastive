"""Test autoencoder trainer."""

from pytest import fixture
from tensorflow import keras
from tensorflow.keras import layers

from ulf.trainers.autoencoder import AutoencoderTrainer, VAETrainer


@fixture
def simple_dense_decoder(simple_dense_model):
    """Return decoder model."""
    inp = layers.Input((8,))
    out = layers.Dense(64, name="decoder_dense1")(inp)
    out = layers.Activation("relu", name="decoder_final_activation")(out)
    return keras.Model(inp, out)


def test_autoencoder_trainer(
    simple_dense_decoder, simple_dense_model, digits_dataset_train, digits_dataset_val
):
    """Test autoencoder trainer."""
    data, _ = digits_dataset_train
    data_val, target_val = digits_dataset_val
    trainer = AutoencoderTrainer(simple_dense_decoder)
    history = trainer.train(simple_dense_model, x=data, x_val=data_val, epochs=12)
    assert history.history["val_loss"][-1] < 0.09


def test_vae_trainer(
    simple_dense_decoder, simple_dense_model, digits_dataset_train, digits_dataset_val
):
    data, _ = digits_dataset_train
    data_val, target_val = digits_dataset_val
    trainer = VAETrainer(simple_dense_decoder)
    history = trainer.train(simple_dense_model, x=data, x_val=data_val, epochs=18)
    assert history.history["val_loss"][-1] < 0.12
    assert history.history["val_kl_loss"][-1] < 0.005
