"""Autoencoder trainers."""

from tensorflow import keras
from tensorflow.keras import layers
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
