"""Bi-directional gans."""
import tensorflow as tf
from tensorflow import keras

from ulf.technique.gan import discriminator_step, generator_step


class BiGAN(keras.Model):
    """
    Basic bi-directional gans implementation.

    For more information see:
    * Donahue, Jeff, Philipp Krähenbühl, and Trevor Darrell.
      "Adversarial feature learning."
      arXiv preprint arXiv:1605.09782
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model, discriminator: keras.Model):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.discriminator_optimizer = None
        self.encoder_loss = None
        self.decoder_loss = None
        self.discriminator_loss = None

    def compile(
        self,
        discriminator_optimizer,
        encoder_optimizer,
        decoder_optimizer,
        discriminator_loss,
        encoder_loss,
        decoder_loss,
        **kwargs,
    ):
        """Compile keras bigan model."""
        super().compile(**kwargs)

        self.encoder_optimizer = keras.optimizers.get(encoder_optimizer)
        self.decoder_optimizer = keras.optimizers.get(decoder_optimizer)
        self.discriminator_optimizer = keras.optimizers.get(discriminator_optimizer)

        self.encoder_loss = keras.losses.get(encoder_loss)
        self.decoder_loss = keras.losses.get(decoder_loss)
        self.discriminator_loss = keras.losses.get(discriminator_loss)

    @tf.function
    def train_step(self, data):
        """One step in training phase."""
        real = data[0] if isinstance(data, tuple) else data
        e_loss = self._encoder_step(real)
        dc_loss = self._decoder_step(batch_size=tf.shape(real)[0])
        ds_loss = self._discriminator_step(real)
        return {"encoder_loss": e_loss, "decoder_loss": dc_loss, "discriminator_loss": ds_loss}

    @tf.function
    def test_step(self, data):
        """One step in validation phase."""
        real = data[0] if isinstance(data, tuple) else data
        e_loss = self._encoder_step(real, is_training=False)
        dc_loss = self._decoder_step(batch_size=tf.shape(real)[0], is_training=False)
        ds_loss = self._discriminator_step(real, is_training=False)
        return {"encoder_loss": e_loss, "decoder_loss": dc_loss, "discriminator_loss": ds_loss}

    def _discriminator_step(self, original, is_training=True):
        z = self.random_latent_vectors(tf.shape(original)[0])
        generated = [self.decoder(z), z]
        real = [original, self.encoder(original)]

        return discriminator_step(
            self.discriminator,
            self.discriminator_loss,
            self.discriminator_optimizer,
            generated,
            real,
            is_training,
        )

    def _decoder_step(self, batch_size, is_training=True):
        return generator_step(
            self.decoder,
            self.decoder_loss,
            self.decoder_optimizer,
            self.random_latent_vectors(batch_size),
            lambda decoder, z: self.discriminator([decoder(z), z]),
            is_training,
        )

    def _encoder_step(self, original, is_training=True):
        return generator_step(
            self.encoder,
            self.encoder_loss,
            self.encoder_optimizer,
            original,
            lambda encoder, x: self.discriminator([x, encoder(x)]),
            is_training,
            positive=True,
        )

    @property
    def latent_shape(self):
        """Shape of latent vector space (z)."""
        return tuple(self.decoder.inputs[0].shape[1:])

    def random_latent_vectors(self, count):
        """Generate samples from z distribution."""
        shape = (count,) + self.latent_shape
        return tf.random.normal(shape=shape)

    def call(self, inputs):
        """Autoencoder logic."""
        return self.encoder(self.decoder(inputs))
