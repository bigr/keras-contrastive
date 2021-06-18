"""Autoencoder trainers."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend, layers
from tensorflow.keras.models import clone_model

from ulf.trainers.base import Trainer


class AutoencoderTrainer(Trainer):
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

    def __init__(self, decoder: keras.Model, optimizer="adam", reconstruction_loss="mse"):
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss
        self.optimizer = optimizer

    def train(
        self,
        model: keras.Model,
        x,
        y=None,
        x_val=None,
        y_val=None,
        sample_weight_val=None,
        **kwargs,
    ):
        """Train given model."""

        if "validation_data" in kwargs:
            raise ValueError("Use *_val instead validation_data.")

        if y is not None or y_val is not None:
            raise ValueError("Semi-supervised learning is not supported.")

        decoder = clone_model(self.decoder)

        autoencoder = self._build_autoencoder(model, decoder)
        autoencoder.compile(self.optimizer, self.reconstruction_loss)
        if sample_weight_val is not None:
            validation_data = (x_val, x_val, sample_weight_val)
        else:
            validation_data = (x_val, x_val)

        return autoencoder.fit(x, x, validation_data=validation_data, **kwargs)

    @staticmethod
    def _build_autoencoder(encoder: keras.Model, decoder: keras.Model) -> keras.Model:
        inps = [layers.Input(tuple(inp.shape[1:])) for inp in encoder.inputs]

        encoded = encoder(inps)
        decoded = decoder(encoded)

        return keras.Model(inps, decoded)


class VAETrainer(Trainer):
    """
    Variational autoencoder trainer.

    For more information see:
    * Kingma, Diederik P., and Max Welling.
      "Auto-encoding variational bayes."
      arXiv preprint arXiv:1312.6114 (2013).
    """

    def __init__(
        self, decoder: keras.Model, epsilon_std=0.1, optimizer="adam", reconstruction_loss="mse"
    ):
        self.decoder = decoder
        self.epsilon_std = epsilon_std
        self.reconstruction_loss = reconstruction_loss
        self.optimizer = optimizer

    def train(
        self,
        model: keras.Model,
        x,
        y=None,
        x_val=None,
        y_val=None,
        sample_weight_val=None,
        **kwargs,
    ):
        """Train given model."""
        if y is not None:
            raise ValueError("Semi-supervised learning is not supported.")

        if "validation_data" in kwargs:
            raise ValueError("Use *_val instead validation_data.")

        if y is not None or y_val is not None:
            raise ValueError("Semi-supervised learning is not supported.")

        decoder = clone_model(self.decoder)

        vae = self._build_vae(model, decoder)
        vae.compile(self.optimizer)

        if sample_weight_val is not None:
            validation_data = (x_val, x_val, sample_weight_val)
        else:
            validation_data = (x_val, x_val)

        return vae.fit(x, x, validation_data=validation_data, **kwargs)

    def _build_vae(self, encoder: keras.Model, decoder: keras.Model) -> keras.Model:
        inps = [layers.Input(tuple(inp.shape[1:])) for inp in encoder.inputs]
        i = self._find_last_trainable_layer_index(encoder)
        encoder_core = keras.Model(encoder.inputs, encoder.layers[i - 1].output)
        encoded_core = encoder_core(inps)
        z_mean = encoder.layers[i](encoded_core)
        z_mean_config = encoder.layers[i].get_config()
        z_mean_config["name"] = f"{z_mean_config['name']}_log_sigma"
        z_log_sigma = type(encoder.layers[i]).from_config(z_mean_config)(encoded_core)
        z = layers.Lambda(self._sampling)([z_mean, z_log_sigma])
        decoded = decoder(z)
        vae = keras.Model(inps, decoded)
        kl_loss = 1 + z_log_sigma - backend.square(z_mean) - backend.exp(z_log_sigma)
        kl_loss = backend.mean(-0.5 * backend.sum(kl_loss, axis=-1))
        reconstruction_loss = backend.mean(
            keras.losses.get(self.reconstruction_loss)(inps, decoded)
        )
        vae.add_loss(kl_loss)
        vae.add_loss(reconstruction_loss)
        vae.add_metric(reconstruction_loss, name="reconstruction_loss")
        vae.add_metric(kl_loss, name="kl_loss")
        return vae

    @staticmethod
    def _find_last_trainable_layer_index(encoder: keras.Model) -> int:
        i = 0
        while True:
            i -= 1
            if encoder.layers[i].weights:
                break
        return i

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = backend.random_normal(mean=0.0, stddev=self.epsilon_std, shape=tf.shape(z_mean))
        return z_mean + backend.exp(z_log_sigma) * epsilon
