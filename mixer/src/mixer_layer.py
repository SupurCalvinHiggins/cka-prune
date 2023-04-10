from tensorflow import keras


from .mlp_token import MLPToken
from .mlp_channel import MLPChannel


class MixerLayer(keras.Model):
    def __init__(
            self,
            num_tokens: int, 
            hidden_dim: int, 
            num_token_hidden: int, 
            num_channel_hidden: int, 
            dropout_rate: float,
        ) -> None:
        """
        Single layer of a MLP-Mixer as described in the MLP-Mixer paper.
        
        Parameters
        ----------
            num_tokens:
                The number of tokens. This is also referred to as the number of patches
                or the sequence length $S$.

            hidden_dim:
                The hidden dimension. This is also referred to as the hidden size $C$.

            num_token_hidden:
                The number of neurons in the hidden layer of the token-mixing MLP. This 
                is also referred to as the MLP dimension $D_S$. 

            num_channel_hidden:
                The number of neurons in the hidden layer of the channel-mixing MLP. 
                This is also referred to as the MLP dimension $D_C$. 

            dropout_rate:
                The rate at which to apply dropout to both the channel-mixing MLP and
                the token-mixing MLP.
        """
        
        super().__init__()

        # Save the parameters.
        self._num_tokens = num_tokens
        self._hidden_dim = hidden_dim
        self._num_token_hidden = num_token_hidden
        self._num_channel_hidden = num_channel_hidden
        self._dropout_rate = dropout_rate

        # Create the layers.
        self.token_mixing_mlp = MLPToken(
            num_tokens=num_tokens,
            num_hidden=num_token_hidden,
            dropout_rate=dropout_rate,
        )
        self.channel_mixing_mlp = MLPChannel(
            hidden_dim=hidden_dim,
            num_hidden=num_channel_hidden,
            dropout_rate=dropout_rate,
        )
    
    def call(self, inputs):
        x = self.token_mixing_mlp(inputs)
        x = self.channel_mixing_mlp(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_tokens": self._num_tokens,
                "hidden_dim": self._hidden_dim,
                "num_token_hidden": self._num_token_hidden,
                "num_channel_hidden": self._num_channel_hidden,
                "dropout_rate": self._dropout_rate,
            }
        )
        return config

    @property
    def mlp1(self):
        return self.token_mixing_mlp
    
    @property
    def mlp2(self):
        return self.channel_mixing_mlp