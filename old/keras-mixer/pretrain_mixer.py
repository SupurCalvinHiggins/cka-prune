import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow import keras_cv
from mixer import Mixer


# RandAugment (2 layers), m=15 (scale is 0-30)
# mixup (1 layer), 0.5
# stochastic depth, 0.1
# dropout, 0.0
# optim adam, 
# lr 0.001, 
# weight decay 0.1, 
# TODO: lr schedule (linear decay),
# resolution 224,
# TODO: linear lr warmup 10K steps, 
# gradiant clipping (global norm 1)
# batch size 4096


def main():
    tf.random.seed(0)

    model = Mixer(
        num_classes=1000,
        num_blocks=8,
        patch_size=32,
        num_patches=49,
        hidden_dim=512,
        num_token_hidden=256,
        num_channel_hidden=2048,
        dropout_rate=0.0,
        stochastic_depth_rate=0.1,
    )
    model = keras.Sequential(
        [
            keras_cv.layers.MixUp(alpha=0.5),
            keras_cv.layers.RandAugment([0, 1], augmentations_per_image=2, magnitude=0.50),
            keras.layers.Resizing(224, 224),
            model,
        ],
    )
    model.build(input_shape=(None, 224, 224, 3))

    optimizer = tfa.optimizers.AdamW(
        weight_decay=0.1, 
        learning_rate=0.001, 
        global_clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )

    batch_size = 4096
    (X_train, y_train), _ = keras.datasets.cifar10.load_data()
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        horizontal_flip=True,
        rescale=1/255,
        validation_split=0.1,
    )
    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='training')
    val_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='validation')

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=10,
    )
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=1000,
        steps_per_epoch=int((len(X_train) * 0.8) / batch_size),
        validation_steps=int((len(X_train) * 0.2) / batch_size),
        callbacks=[reduce_lr],
    )