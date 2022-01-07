import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpus)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

import tensorflow_addons as tfa
from tensorflow import keras
from source.layers import build_swin_T_model
import sys

num_classes = 100
# input_shape = (32, 32, 3)

def draw_sample(x_train, label):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
    plt.savefig(f'{label}.jpg')
    plt.clf()
    
def draw_loss(history, label):
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(f'{label}.jpg')
    plt.clf()
    
    plt.plot(history.history["accuracy"], label="train_loss")
    plt.plot(history.history["val_accuracy"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.title("Train and Validation Acc Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(f'{label}_acc.jpg')
    plt.clf()
    
    loss, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {round(loss, 2)}")
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

if __name__ == '__main__':
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=42, restore_best_weights=True)
    cd = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=40, t_mul=2.0, m_mul=0.9, alpha=1e-2)
    ls = tf.keras.callbacks.LearningRateScheduler(cd)
    
    if sys.argv[2] == 'pretrain':
        # pre-train cifar100
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        
        x_train, x_test = x_train/255., x_test/255.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        input_shape = x_train.shape[1:]
        num_classes = y_train.shape[-1]

        print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

        draw_sample(x_train, 'pretrain_sample')

        # hyperparameters
        patch_size = (4, 4) # 2-by-2 size patches
        dropout_rate = 0.03 # Dropout rate
        num_heads = 8 # Attention heads
        embed_dim = 96 # Embedding dimesion
        num_mlp = 512 # MLP layer size
        qkv_bias = True # Convert embeded patches to query, key, and values with a learnable
        window_size = 2 # Size of attention window
        shift_size = 1 # Size of shifting window
        image_dimension = 128 # Initial image size

        num_patch_x = image_dimension // patch_size[0]#input_shape[0] // patch_size[0]
        num_patch_y = image_dimension // patch_size[1]#input_shape[1] // patch_size[1]

        learning_rate = 1e-3
        batch_size = 128
        num_epochs = 20
        validation_split = 0.2
        weight_decay = 0.0001
        label_smoothing = 0.1

        # build model
        model = build_swin_T_model(
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
            callbacks=[es, ls],
        )
        draw_loss(history, 'pre_train_loss')
        model.save_weights('pre_trained.h5')

    if sys.argv[1] == 'mnist':
        # mnist
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    if len(x_train.shape) !=4 :
        x_train = np.concatenate([np.expand_dims(x_train, axis=-1) for i in range(3)], axis=-1)
        x_test = np.concatenate([np.expand_dims(x_test, axis=-1) for i in range(3)], axis=-1)
    
    x_train, x_test = x_train/255., x_test/255.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    input_shape = x_train.shape[1:]#(32, 32, 3)
    num_classes = y_train.shape[-1]

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    draw_sample(x_train, f'{sys.argv[1]}_train_sample')

    # hyperparameters
    patch_size = (4, 4) # 2-by-2 size patches
    dropout_rate = 0.03 # Dropout rate
    num_heads = 8 # Attention heads
    embed_dim = 96 # Embedding dimesion
    num_mlp = 512 # MLP layer size
    qkv_bias = True # Convert embeded patches to query, key, and values with a learnable
    window_size = 2 # Size of attention window
    shift_size = 1 # Size of shifting window
    image_dimension = 128 # Initial image size

    num_patch_x = image_dimension // patch_size[0]#input_shape[0] // patch_size[0]
    num_patch_y = image_dimension // patch_size[1]#input_shape[1] // patch_size[1]

    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 20
    validation_split = 0.2
    weight_decay = 0.00001
    label_smoothing = 0.1

    # build model
    model = build_swin_T_model(
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

    if sys.argv[2] == 'pretrain':
        model.load_weights('pre_trained.h5')
        # reset last layer's weight
        initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        for i in [-2, -1]:
            layer_new_weights = []
            for layer_weights in model.layers[i].get_weights():
                weights = initializer(np.shape(layer_weights))
                layer_new_weights.append(weights)
            model.layers[i].set_weights(layer_new_weights)
        
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=validation_split,
        callbacks=[es, ls],
    )
    if sys.argv[2] == 'pretrain':
        draw_loss(history, f'{sys.argv[1]}_train_loss')
    else:
        draw_loss(history, f'{sys.argv[1]}_only_loss')
        