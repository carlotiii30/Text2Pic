import unittest
from unittest.mock import MagicMock

from src import utils
from src.nums import builders


class TestTraining(unittest.TestCase):
    def test_load_dataset(self):
        utils.load_dataset = MagicMock(return_value="mocked_dataset")

        dataset = utils.load_dataset()
        self.assertEqual(dataset, "mocked_dataset")

    def test_build_models(self):
        builders.build_models = MagicMock(return_value=("mocked_generator", "mocked_discriminator"))
        generator, discriminator = builders.build_models()

        self.assertEqual(generator, "mocked_generator")
        self.assertEqual(discriminator, "mocked_discriminator")

    def test_build_conditional_gan(self):
        builders.build_conditional_gan = MagicMock(return_value="mocked_cond_gan")
        cond_gan = builders.build_conditional_gan("mocked_generator", "mocked_discriminator")

        self.assertEqual(cond_gan, "mocked_cond_gan")

    def test_train_model(self):
        utils.train_model = MagicMock()

        dataset = "mocked_dataset"
        cond_gan = "mocked_cond_gan"

        utils.train_model(dataset, cond_gan)

        utils.train_model.assert_called_once_with(dataset, cond_gan)

    def test_save_model_weights(self):
        utils.save_model_weights = MagicMock()

        cond_gan = "mocked_cond_gan"

        utils.save_model_weights(cond_gan, "cond_weights.weights.h5")

        utils.save_model_weights.assert_called_once_with(cond_gan, "cond_weights.weights.h5")
