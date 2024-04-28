import unittest
import numpy as np
import tensorflow as tf
from models.generative_model import GenerativeModel

class TestGenerativeModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (4,)
        self.hidden_units = 64
        self.output_units = 1
        self.model = GenerativeModel(self.input_shape, self.hidden_units, self.output_units)

    def test_model_architecture(self):
        self.assertEqual(len(self.model.model.layers), 3)
        self.assertEqual(self.model.model.layers[0].units, self.hidden_units)
        self.assertEqual(self.model.model.layers[1].units, self.output_units)

    def test_model_compilation(self):
        self.assertIsInstance(self.model.model.compile, tf.keras.callbacks.History)

    def test_model_training(self):
        X_train = np.random.rand(100, *self.input_shape)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(50, *self.input_shape)
        y_val = np.random.randint(0, 2, 50)
        self.model.train(X_train, y_train, epochs=1, batch_size=32, X_val, y_val)
        self.assertGreater(self.model.model.history.history['loss'][0], 0)

    def test_model_evaluation(self):
        X_train = np.random.rand(100, *self.input_shape)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(50, *self.input_shape)
        y_val = np.random.randint(0, 2, 50)
        self.model.train(X_train, y_train, epochs=1, batch_size=32, X_val, y_val)
        loss, accuracy = self.model.evaluate(X_val, y_val)
        self.assertGreater(accuracy, 0.5)

    def test_model_prediction(self):
        X_new = np.random.rand(1, *self.input_shape)
        prediction = self.model.predict(X_new)
        self.assertEqual(prediction.shape, (1, self.output_units))

if __name__ == '__main__':
    unittest.main()
