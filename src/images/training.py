from src import utils
from src.nums import builders

dataset = utils.load_dataset("cifar10")
generator, discriminator = builders.build_models()
cond_gan = builders.build_conditional_gan(generator, discriminator)
utils.train_model(dataset, cond_gan)
utils.save_model_weights(cond_gan, "cgan_images.weights.h5")
