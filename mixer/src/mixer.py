from tensorflow import keras
from typing import Optional, Any


from .mixer_layer import MixerLayer
from .patch_extractor import PatchExtractor


class Mixer(keras.Model):
    def __init__(
        self, 
        num_classes: int, 
        num_blocks: int, 
        patch_size: int, 
        num_patches: int, 
        hidden_dim: int, 
        num_token_hidden: int, 
        num_channel_hidden: int,
        dropout_rate: float,
        postional_encoding: Optional[Any] = None,
    ) -> None:
        """
        A MLP-Mixer as described in the MLP-Mixer paper.
        
        Parameters
        ----------
            num_classes:
                The number of classes in the dataset. Corresponds to the number of 
                output neurons in the classifier head.
            
            num_blocks:
                The number of MLP-Mixer layers in the model.
            
            patch_size:
                The size of the patches to extract. Patches will have shape 
                (patch_size, patch_size).
            
            num_patches:
                The total number of patches that will be extracted from each image. This 
                is also referred to as the number of tokens or the sequence length $S$.

            hidden_dim:
                The hidden dimension. This is also referred to as the hidden size $C$.

            num_token_hidden:
                The number of neurons in the hidden layer of the token-mixing MLPs. This 
                is also referred to as the MLP dimension $D_S$. 

            num_channel_hidden:
                The number of neurons in the hidden layer of the channel-mixing MLPs. 
                This is also referred to as the MLP dimension $D_C$. 

            dropout_rate:
                The rate at which to apply dropout to both the channel-mixing MLPs and
                the token-mixing MLPs.
        """

        super().__init__()

        if postional_encoding is not None:
            raise ValueError("Postional encoding is not currently supported.")

        # TODO: Add postional encoding.
        # self._postional_encoding = postional_encoding        
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size, 
            num_patches=num_patches
        )

        self.embedding = keras.layers.Dense(hidden_dim)

        # TODO: Add positional encoding.
        # if postional_encoding:
        #     positions = tf.range(start=0, limit=num_tokens, delta=1)
        #     self.position_embedding = keras.layer.Embedding(
        #         input_dim=num_tokens, output_dim=hidden_dim,
        #     )(positions)

        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(
                MixerLayer(
                        num_tokens=num_patches, 
                        hidden_dim=hidden_dim, 
                        num_token_hidden=num_token_hidden, 
                        num_channel_hidden=num_channel_hidden, 
                        dropout_rate=dropout_rate
                    )
                )

        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(rate=dropout_rate)
        self.logits = keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.patch_extractor(inputs)
        x = self.embedding(x)

        # TODO: Add postional encoding.
        # if self._postional_encoding:
        #     x = x + self.position_embedding
    
        for block in self.blocks:
            x = block(x)
    
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.logits(x)
        return x

    # def summary(self, **kwargs):
    #     x = keras.layers.Input(shape=(32, 32, 3))
    #     model = keras.Model(inputs=[x], outputs=self.call(x))
    #     return model.summary(**kwargs)