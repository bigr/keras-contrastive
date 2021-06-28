"""Autoencoder method."""
from contextlib import nullcontext

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend


def autoencoder_step(original, loss_fn, optimizer, encoder, decoder, is_training=True):
    """Do generic autoencoder training/testing step."""
    with (tf.GradientTape() if is_training else nullcontext()) as tape:
        encoded = encoder(original)
        reconstructed = decoder(encoded)
        loss = loss_fn(original, reconstructed)

    if is_training:
        grads = tape.gradient(loss, encoder.trainable_weights + decoder.trainable_weights)
        trainable_weights = encoder.trainable_weights + decoder.trainable_weights
        optimizer.apply_gradients(zip(grads, trainable_weights))

    return backend.mean(loss)


class Autoencoder(keras.Model):
    """
    Basic autoencoder method.

    For more information see:
    * Baldi, Pierre.
      "Autoencoders, unsupervised learning, and deep architectures."
      Proceedings of ICML workshop on unsupervised and transfer learning.
      JMLR Workshop and Conference Proceedings, 2012.
    * Hinton, Geoffrey E.
      "Learning translation invariant recognition in a massively parallel networks."
      International Conference on Parallel Architectures and Languages Europe.
      Springer, Berlin, Heidelberg, 1987.
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model):
        super().__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.autoencoder_loss = None
        self.autoencoder_optimizer = None

    def compile(
        self,
        optimizer,
        loss,
    ):
        """Compile keras model."""
        super().compile()
        self.autoencoder_optimizer = keras.optimizers.get(optimizer)
        self.autoencoder_loss = keras.losses.get(loss)

    @tf.function
    def train_step(self, data):
        """One step in training phase."""
        real = data[0] if isinstance(data, tuple) else data
        r_loss = self._autoencoder_step(real)
        return {"reconstruction_loss": r_loss}

    @tf.function
    def test_step(self, data):
        """One step in validation phase."""
        real = data[0] if isinstance(data, tuple) else data
        r_loss = self._autoencoder_step(real, is_training=False)
        return {"reconstruction_loss": r_loss}

    def _autoencoder_step(self, original, is_training=True):
        return autoencoder_step(
            original,
            self.autoencoder_loss,
            self.autoencoder_optimizer,
            self.encoder,
            self.decoder,
            is_training,
        )

    def call(self, inputs):
        """Autoencoder logic."""
        return self.decoder(self.encoder(inputs))
