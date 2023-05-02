from tensorflow import keras
import tensorflow_addons as tfa


class MLPBase(keras.Model):
    def __init__(
            self, 
            num_input: int, 
            num_hidden: int, 
            dropout_rate: float
        ) -> None:
        """
        Base MLP used to create the channel-mixing MLP and token-mixing MLP described 
        in the MLP-Mixer paper.

        Parameters
        ----------
            num_input: 
                The number of neurons in the input layer.

            num_hidden:
                The number of neurons in the hidden layer.

            dropout_rate:
                The rate at which to apply dropout.
        """

        super().__init__()

        # Save the parameters.
        self._num_input = num_input
        self._num_hidden = num_hidden
        self._dropout_rate = dropout_rate

        # Create the layers.
        self.dense1 = keras.layers.Dense(num_hidden, input_shape=(num_input,))
        self.gelu = tfa.layers.GELU()
        self.dense2 = keras.layers.Dense(num_input, input_shape=(num_hidden,))
        self.dropout = keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        # print("in mlp: ", inputs.shape)
        x = self.dense1(inputs)
        # print("out 1: ", x.shape)
        x = self.gelu(x)
        x = self.dense2(x)
        # print("out 2: ", x.shape)
        x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_input": self._num_input,
                "num_hidden": self._num_hidden,
                "dropout_rate": self._dropout_rate,
            }
        )
        return config