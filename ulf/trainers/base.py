"""Definition of base interface."""

from abc import ABC, abstractmethod

from tensorflow import keras


class Trainer(ABC):
    """Inherid this interface for each specific (un/semi)supervised training method."""

    @abstractmethod
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
        """
        Train given model with some (un/semi)supervised method.

        Parameters
        ----------
        model
            Keras model to be trained.
        x
            Input data passed to the keras fit.
        y
            Target data (for semi-supervised learning) otherwise `None`
        x_val
            Validation input data
        y_val
            Validation target data (for semi-supervised learning) otherwise `None`
        sample_weight_val
            Validation sample weights
        kwargs
            Additional parameters passed to the keras fit model
        Returns
        -------
        Keras History object.
        """
        pass
