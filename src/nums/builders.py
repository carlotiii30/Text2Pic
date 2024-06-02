import keras

from src import utils
from src.nums import cgan


def build_models():
    """
    Builds the generator and discriminator models for the cGAN.

    Returns:
        keras.Model: Generator model.
        keras.Model: Discriminator model.
    """
    # - - - - - - - Calculate the number of input channels - - - - - - -
    gen_channels = utils.latent_dim + utils.num_classes
    dis_channels = utils.num_channels + utils.num_classes

    # - - - - - - - Generator - - - - - - -
    generator = keras.Sequential(
        [
            keras.layers.InputLayer((gen_channels,)),
            keras.layers.Dense(7 * 7 * gen_channels),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Reshape((7, 7, gen_channels)),
            keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2DTranspose(utils.batch_size, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2DTranspose(1, kernel_size=7, strides=1, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

    # - - - - - - - Discriminator - - - - - - -
    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((28, 28, dis_channels)),
            keras.layers.Conv2D(utils.batch_size, kernel_size=3, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.GlobalMaxPool2D(),
            keras.layers.Dense(1),
        ],
        name="discriminator",
    )

    return generator, discriminator


def build_conditional_gan(generator, discriminator):
    """
    Builds the conditional GAN (cGAN) model.

    Args:
        generator (keras.Model): Generator model.
        discriminator (keras.Model): Discriminator model.

    Returns:
        conditionalGAN: Compiled cGAN model.
    """
    config = cgan.GANConfig(
        discriminator=discriminator,
        generator=generator,
        latent_dim=utils.latent_dim,
        image_size=utils.image_size,
        num_classes=utils.num_classes,
    )

    cond_gan = cgan.ConditionalGAN(
        config=config,
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    cond_gan.compile()

    return cond_gan
