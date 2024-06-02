import os

import keras
import numpy as np
import tensorflow as tf

from src.nums import builders

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128


def load_dataset():
    """
    Loads the MNIST dataset, preprocesses it, and returns it as a TensorFlow dataset.

    Returns:
        tf.data.Dataset: Dataset containing the preprocessed MNIST images and labels.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    all_labels = keras.utils.to_categorical(all_labels, 10)

    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return dataset


def train_model(dataset, cond_gan):
    """
    Trains the conditional GAN (cGAN) model.

    Args:
        dataset (tf.data.Dataset): Dataset containing the training images and labels.
        cond_gan (conditionalGAN): Compiled cGAN model.
    """
    cond_gan.fit(dataset, epochs=50)


def save_model_weights(cond_gan, filename):
    """
    Saves the weights of the conditional GAN (cGAN) model.

    Args:
        cond_gan (conditionalGAN): Compiled cGAN model.
        filename (str): Filepath to save the model weights.
    """
    if os.path.exists(filename):
        os.remove(filename)
    cond_gan.save_weights(filename)


def load_model_with_weights(filename):
    """
    Loads the conditional GAN (cGAN) model with saved weights.

    Args:
        filename (str): Filepath to the saved model weights.

    Returns:
        conditionalGAN: cGAN model with loaded weights.
    """
    generator, discriminator = builders.build_models()

    new_cond_gan = builders.build_conditional_gan(generator, discriminator)

    new_cond_gan.load_weights(filename)
    return new_cond_gan
