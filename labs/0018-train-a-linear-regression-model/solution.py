import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.

    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)

    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    n_samples = X.shape[0]
    lr = 0.1
    epochs = 1000

    for _ in range(epochs):
        y_pred = X @ W + b
        error = y_pred - y

        dW = (1 / n_samples) * (X.T @ error)
        db = (1 / n_samples) * np.sum(error)

        W -= lr * dW
        b -= lr * db

    return W, b