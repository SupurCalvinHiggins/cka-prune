import tensorflow as tf
from tensorflow import keras


from .mlp_base import MLPBase


class MLPToken(keras.Model):
    def __init__(
            self, 
            num_tokens: int, 
            num_hidden: int, 
            dropout_rate: float,
        ) -> None:
        """
        Token-mixing MLP as described in the MLP-Mixer paper.

        Parameters
        ----------
            num_tokens: 
                The number of tokens. This is also referred to as the number of patches
                or the sequence length $S$.

            num_hidden:
                The number of neurons in the hidden layer. This is also referred to as
                the MLP dimension $D_S$. 

            dropout_rate:
                The rate at which to apply dropout.
        """

        super().__init__()

        # Save the parameters.
        self._hidden_dim = num_tokens
        self._num_hidden = num_hidden
        self._dropout_rate = dropout_rate

        # Construct the layers.
        self.mlp = MLPBase(
            num_input=num_tokens,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate
        )
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.layer_norm(inputs)
        x = tf.linalg.matrix_transpose(x)
        x = self.mlp(x)
        x = tf.linalg.matrix_transpose(x)
        x = x + inputs
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "hidden_dim": self._hidden_dim,
                "num_hidden": self._num_hidden,
                "dropout_rate": self._dropout_rate,
            }
        )
        return config