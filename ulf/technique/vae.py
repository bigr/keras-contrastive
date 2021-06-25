"""Variational autoencoder method."""
from contextlib import nullcontext
from functools import cached_property

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend, layers

from ulf.technique.autoencoder import Autoencoder


def vae_step(
    original,
    reconstruction_loss_fn,
    optimizer,
    encoder,
    sampling_encoder,
    sigma_encoder,
    decoder,
    is_training=True,
):
    """Do generic variation autoencoder training/testing step."""
    with (tf.GradientTape() if is_training else nullcontext()) as tape:
        z = sampling_encoder(original)
        z_log_sigma = sigma_encoder(original)
        z_mean = encoder(original)

        reconstructed = decoder(z)

        kl_loss = -0.5 * backend.sum(
            1 + z_log_sigma - backend.square(z_mean) - backend.exp(z_log_sigma), axis=-1
        )
        r_loss = reconstruction_loss_fn(original, reconstructed) * encoder.inputs[0].shape[-1]

        loss = backend.mean(r_loss + kl_loss)

    if is_training:
        grads = tape.gradient(loss, sampling_encoder.trainable_weights + decoder.trainable_weights)
        trainable_weights = sampling_encoder.trainable_weights + decoder.trainable_weights
        optimizer.apply_gradients(zip(grads, trainable_weights))

    return backend.mean(r_loss), backend.mean(kl_loss), loss


class VAE(Autoencoder):
    """
    Variational autoencoder.

    For more information see:
    * Kingma, Diederik P., and Max Welling.
      "Auto-encoding variational bayes."
      arXiv preprint arXiv:1312.6114 (2013).
    """

    def __init__(
        self,
        encoder: keras.Model,
        sigma_encoder: keras.Model,
        decoder: keras.Model,
        epsilon_std: float = 0.1,
    ):
        super().__init__(encoder, decoder)
        self.sigma_encoder = sigma_encoder
        self.epsilon_std = epsilon_std

    @tf.function
    def train_step(self, data):
        """One step in training phase."""
        real = data[0] if isinstance(data, tuple) else data
        r_loss, kl_loss, loss = self._vae_step(real)
        return {"reconstruction_loss": r_loss, "kl_loss": kl_loss, "loss": loss}

    @tf.function
    def test_step(self, data):
        """One step in validation phase."""
        real = data[0] if isinstance(data, tuple) else data
        r_loss, kl_loss, loss = self._vae_step(real, is_training=False)
        return {"reconstruction_loss": r_loss, "kl_loss": kl_loss, "loss": loss}

    def _vae_step(self, original, is_training=True):
        return vae_step(
            original,
            self.autoencoder_loss,
            self.autoencoder_optimizer,
            self.encoder,
            self.sampling_encoder,
            self.sigma_encoder,
            self.decoder,
            is_training,
        )

    @cached_property
    def sampling_encoder(self) -> keras.Model:
        """Return encoder with normal noise added."""
        inps = [layers.Input(tuple(inp.shape[1:])) for inp in self.encoder.inputs]
        z_mean = self.encoder(inps)
        z_log_sigma = self.sigma_encoder(inps)
        z = layers.Lambda(self._sampling)([z_mean, z_log_sigma])
        return keras.Model(inps, z)

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = backend.random_normal(mean=0.0, stddev=self.epsilon_std, shape=tf.shape(z_mean))
        return z_mean + backend.exp(z_log_sigma) * epsilon
