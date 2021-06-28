"""Generative adversarial networks."""
from contextlib import nullcontext

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend


def discriminator_step(model, loss, optimizer, generated, real, is_training=True):
    """Do generic discriminator training/testing step."""
    if isinstance(generated, list):
        combined = [tf.concat([g, r], axis=0) for g, r in zip(generated, real)]
        generated_count = tf.shape(generated[0])[0]
        real_count = tf.shape(real[0])[0]
    else:
        combined = tf.concat([generated, real], axis=0)
        generated_count = tf.shape(generated)[0]
        real_count = tf.shape(real)[0]
    labels = tf.concat([tf.ones((generated_count, 1)), tf.zeros((real_count, 1))], axis=0)

    if is_training:
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

    with (tf.GradientTape() if is_training else nullcontext()) as tape:
        preds = model(combined)
        loss = loss(labels, preds)

    if is_training:
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return backend.mean(loss)


def generator_step(
    model, loss, optimizer, latent_vectors, discriminate, is_training=True, positive=False
):
    """Do generic generator training/testing step."""
    with (tf.GradientTape() if is_training else nullcontext()) as tape:
        preds = discriminate(model, latent_vectors)
        shape = (tf.shape(preds)[0], 1)
        labels = tf.ones(shape) if positive else tf.zeros(shape)
        loss = loss(labels, preds)

    if is_training:
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return tf.reduce_mean(loss)


class GAN(keras.Model):
    """
    Basic generative adversarial network method.

    For more information see:
    * Goodfellow, Ian J., et al. "Generative adversarial networks."
      arXiv preprint arXiv:1406.2661 (2014).
    """

    def __init__(self, generator: keras.Model, discriminator: keras.Model):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.discriminator_loss = None
        self.generator_loss = None

    def compile(
        self,
        discriminator_optimizer,
        generator_optimizer,
        discriminator_loss,
        generator_loss,
        **kwargs,
    ):
        """Compile keras GAN model."""
        super().compile(**kwargs)

        self.discriminator_optimizer = keras.optimizers.get(discriminator_optimizer)
        self.generator_optimizer = keras.optimizers.get(generator_optimizer)
        self.discriminator_loss = keras.losses.get(discriminator_loss)
        self.generator_loss = keras.losses.get(generator_loss)

    @tf.function
    def train_step(self, data):
        """One step in training phase."""
        real = data[0] if isinstance(data, tuple) else data
        d_loss = self._discriminator_step(real)
        g_loss = self._generator_step(batch_size=tf.shape(real)[0])
        return {"discriminator_loss": d_loss, "generator_loss": g_loss}

    @tf.function
    def test_step(self, data):
        """One step in validation phase."""
        real = data[0] if isinstance(data, tuple) else data
        d_loss = self._discriminator_step(real, is_training=False)
        g_loss = self._generator_step(batch_size=tf.shape(real)[0], is_training=False)
        return {"discriminator_loss": d_loss, "generator_loss": g_loss}

    @tf.function
    def _discriminator_step(self, real, is_training=True):
        batch_size = tf.shape(real)[0]
        generated = self.generator(self.random_latent_vectors(batch_size))
        return discriminator_step(
            self.discriminator,
            self.discriminator_loss,
            self.discriminator_optimizer,
            generated,
            real,
            is_training,
        )

    @tf.function
    def _generator_step(self, batch_size, is_training=True):
        return generator_step(
            self.generator,
            self.generator_loss,
            self.generator_optimizer,
            self.random_latent_vectors(batch_size),
            lambda model, z: self.discriminator(model(z)),
            is_training,
        )

    def call(self, inputs):
        """Use model for generation."""
        return self.generator(inputs)

    def random_latent_vectors(self, count):
        """Generate samples from z distribution."""
        shape = (count,) + self.latent_shape
        return tf.random.normal(shape=shape)

    @property
    def latent_shape(self):
        """Shape of latent vector space (z)."""
        return tuple(self.generator.inputs[0].shape[1:])
