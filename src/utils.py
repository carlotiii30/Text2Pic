import os

import keras
import numpy as np
import tensorflow as tf

batch_size = 64
latent_dim = 128


def load_dataset(dataset_name):
    dataset_dict = {
        "mnist": (keras.datasets.mnist, 10, 1, 28),
        "cifar10": (keras.datasets.cifar10, 10, 3, 32),
        "cifar100": (keras.datasets.cifar100, 100, 3, 32),
    }

    if dataset_name not in dataset_dict:
        raise ValueError("Invalid dataset name")

    dataset, num_classes, num_channels, image_size = dataset_dict[dataset_name]
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    all_images = np.concatenate([x_train, x_test]).astype("float32") / 255.0
    all_labels = keras.utils.to_categorical(np.concatenate([y_train, y_test]), num_classes)

    all_images = np.reshape(all_images, (-1, image_size, image_size, num_channels))

    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return dataset


def train_model(dataset, cond_gan):
    cond_gan.fit(dataset, epochs=50)


def save_model_weights(cond_gan, filename):
    if os.path.exists(filename):
        os.remove(filename)
    cond_gan.save_weights(filename)


def load_model_with_weights(filename, cond_gan):
    cond_gan.load_weights(filename)
    return cond_gan
