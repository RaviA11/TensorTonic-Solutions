import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        for i in range(n_samples):
            z = np.dot(weights, X[i]) + bias
            if z >= 0:
                y_hat = 1
            else:
                y_hat = 0
            
            error = y[i] - y_hat
            weights += (lr * error * X[i])
            bias += (lr * error)

    return (weights.tolist(), float(bias))