from src import utils
from src.nums import builders

dataset = utils.load_dataset("mnist")
generator, discriminator = builders.build_models()
cond_gan = builders.build_conditional_gan(generator, discriminator)
utils.train_model(dataset, cond_gan)
utils.save_model_weights(cond_gan, "cond_weights.weights.h5")
