"""Adversial autoencoders."""
import tensorflow as tf
from tensorflow import keras

from ulf.technique.autoencoder import Autoencoder
from ulf.technique.gan import discriminator_step


class AAE(Autoencoder):
    """
    Basic adversial autoencoder.

    For more information see:
    * Makhzani, Alireza, et al. "Adversarial autoencoders."
      arXiv preprint arXiv:1511.05644 (2015).
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model, discriminator: keras.Model):
        super().__init__(encoder, decoder)

        self.discriminator = discriminator
        self.discriminator_optimizer = None
        self.discriminator_loss = None

    def compile(
        self,
        discriminator_optimizer,
        autoencoder_optimizer,
        discriminator_loss,
        autoencoder_loss,
        **kwargs,
    ):
        """Compile keras adversial autoencoder model."""
        super().compile(autoencoder_optimizer, autoencoder_loss, **kwargs)
        self.discriminator_optimizer = keras.optimizers.get(discriminator_optimizer)
        self.discriminator_loss = keras.losses.get(discriminator_loss)

    @tf.function
    def train_step(self, data):
        """One step in training phase."""
        losses = super().train_step(data)
        real = data[0] if isinstance(data, tuple) else data
        losses["discriminator_loss"] = self._discriminator_step(real)
        return losses

    @tf.function
    def test_step(self, data):
        """One step in validation phase."""
        losses = super().test_step(data)
        real = data[0] if isinstance(data, tuple) else data
        losses["discriminator_loss"] = self._discriminator_step(real, is_training=False)
        return losses

    def _discriminator_step(self, original, is_training=True):
        generated = self.encoder(original)
        real = self.random_latent_vectors(tf.shape(original)[0])

        return discriminator_step(
            self.discriminator,
            self.discriminator_loss,
            self.discriminator_optimizer,
            generated,
            real,
            is_training,
        )

    def random_latent_vectors(self, count):
        """Generate samples from z distribution."""
        shape = (count,) + self.latent_shape
        return tf.random.normal(shape=shape)

    @property
    def latent_shape(self):
        """Shape of latent vector space (z)."""
        return tuple(self.decoder.inputs[0].shape[1:])
