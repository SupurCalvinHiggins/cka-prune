import tensorflow as tf
from tensorflow import keras


class PatchExtractor(keras.layers.Layer):
    def __init__(
            self, 
            patch_size: int, 
            num_patches: int,
        ) -> None:
        """
        Implement patch extraction as a layer.

        Parameters
        ----------
            patch_size:
                The size of the patches to extract. Patches will have shape 
                (patch_size, patch_size).

            num_patches:
                The total number of patches that will be extracted from each image. This 
                is also referred to as the number of tokens or the sequence length $S$.
        """

        super().__init__()

        # Save the parameters.
        self._patch_size = patch_size
        self._num_patches = num_patches

    def call(self, images):
        # Ensure that the images are square.
        shape = tf.shape(images)
        batch_size = shape[0]

        # TODO: Fix assertion.
        # image_height = shape[1]
        # image_width = shape[2]
        # assert image_height == image_width

        # Ensure that the number of patches is consistent.
        # TODO: Fix assertion.
        # act_num_patches = (
        #     (image_height * image_width) / (self._patch_size * self._patch_size)
        # )
        # assert act_num_patches == self._num_patches

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self._patch_size, self._patch_size, 1],
            # Without overlapping, stride horizontally and vertically.
            strides=[1, self._patch_size, self._patch_size, 1],
            # Rate: Dilation factor [1 1* 1* 1] controls the spacing between the kernel points.
            rates=[1, 1, 1, 1],
            # Patches contained in the images are considered, no zero padding.
            padding="VALID",
        )

        # The number of elements in each patch.
        patch_dims = tf.shape(patches)[-1]
        # Flatten the inner dimensions.
        patches = tf.reshape(patches, [batch_size, self._num_patches, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'patch_size': self._patch_size,
                'num_patches': self._num_patches,
            }
        )
        return config


if __name__ == '__main__':
    (X, _), (_, _) = keras.datasets.cifar10.load_data()
    patch_extractor = PatchExtractor(4, 64)
    print(tf.shape(patch_extractor(X)))
