import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from source.layers import build_swin_model

num_classes = 100
input_shape = (32, 32, 3)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.dataset.cifar100.load_data()
    x_train, x_test = x_train/255., x_test/255.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imsho(x_train[i])
    plt.show()

    # hyperparameters
    patch_size = (2, 2) # 2-by-2 size patches
    dropout_rate = 0.03 # Dropout rate
    num_heads = 8 # Attention heads
    embed_dim = 64 # Embedding dimesion
    num_mlp = 64 # MLP layer size
    qkv_bias = True # Convert embeded patches to query, key, and values with a learnable
    window_size = 2 # Size of attention window
    shift_size = 1 # Size of shifting window
    image_dimension = 32 # Initial image size

    num_patch_x = input_shape[0] // patch_size[0]
    num_patch_y = input_shape[1] // patch_size[1]

    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 40
    validation_split = 0.1
    weight_decay = 0.0001
    label_smoothing = 0.1

    # build model
    model = build_swin_model(
                input_shape,
                image_dimension,
                patch_size,
                num_patch_x,
                num_patch_y,
                embed_dim,
                num_heads,
                window_size,
                num_mlp,
                qkv_bias,
                dropout_rate,
                shift_size,
                num_classes
            )
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        optimizer=tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay,
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(5, name='top-5-accuracy'),
        ],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=validation_split,
    )

    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig('train_loss.jpg')

    loss, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {round(loss, 2)}")
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")