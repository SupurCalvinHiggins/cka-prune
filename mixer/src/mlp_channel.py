from tensorflow import keras


from .mlp_base import MLPBase


class MLPChannel(keras.Model):
    def __init__(
            self, 
            hidden_dim: int, 
            num_hidden: int, 
            dropout_rate: float,
        ) -> None:
        """
        Token-mixing MLP as described in the MLP-Mixer paper.

        Parameters
        ----------
            hidden_dim: 
                The hidden dimension. This is also referred to as the hidden size $C$.

            num_hidden:
                The number of neurons in the hidden layer. This is also referred to as
                the MLP dimension $D_C$. 

            dropout_rate:
                The rate at which to apply dropout.
        """

        super().__init__()

        # Save the parameters.
        self._hidden_dim = hidden_dim
        self._num_hidden = num_hidden
        self._dropout_rate = dropout_rate

        # Create the layers.
        self.mlp = MLPBase(
            num_input=hidden_dim, 
            num_hidden=num_hidden, 
            dropout_rate=dropout_rate
        )
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        x = self.layer_norm(inputs)
        x = self.mlp(x)
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

    def summary(self, **kwargs):
        x = keras.layers.Input(shape=(128, 64))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary(**kwargs)