import argparse
import numpy as np
import tensorflow as tf
from models.generative_model import GenerativeModel

def train_model(X_train, y_train, X_val, y_val, model_path, epochs=10, batch_size=32):
    """
    Trains a generative model using the provided training and validation data.

    Args:
        X_train (numpy.ndarray): An array containing input features for training data.
        y_train (numpy.ndarray): An array containing target labels for training data.
        X_val (numpy.ndarray): An array containing input features for validation data.
        y_val (numpy.ndarray): An array containing target labels for validation data.
        model_path (str): A string containing the path to save the trained model.
        epochs (int): An integer specifying the number of epochs to train the model.
        batch_size (int): An integer specifying the batch size for training the model.

    """
    # Create a new instance of the generative model
    model = GenerativeModel(X_train.shape[1], 64, 1)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, X_val=X_val, y_val=y_val)

    # Save the trained model
    model.save(model_path)

def evaluate_model(X_test, y_test, model_path):
    """
    Evaluates a generative model using the provided test data.

    Args:
        X_test (numpy.ndarray): An array containing input features for test data.
        y_test (numpy.ndarray): An array containing target labels for test data.
        model_path (str): A string containing the path to load the trained model.

    Returns:
        loss (float): The loss value for the test data.
        accuracy (float): The accuracy value for the test data.

    """
    # Load the trained model
    model = GenerativeModel(X_test.shape[1], 64, 1)
    model.load(model_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    return loss, accuracy

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, nargs=2, metavar=('TRAIN_DATA', 'VAL_DATA'), required=True)
    parser.add_argument('--test', type=str, nargs=2, metavar=('TEST_DATA', 'MODEL_PATH'), required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    # Train the model
    X_train, y_train = np.load(args.train[0]), np.load(args.train[1])
    X_val, y_val = np.load(args.train[0]), np.load(args.train[1])
    train_model(X_train, y_train, X_val, y_val, args.test[1], epochs=args.epochs, batch_size=args.batch_size)

    # Evaluate the model
    X_test, y_test = np.load(args.test[0]), np.load(args.test[1])
    loss, accuracy = evaluate_model(X_test, y_test, args.test[1])

    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
