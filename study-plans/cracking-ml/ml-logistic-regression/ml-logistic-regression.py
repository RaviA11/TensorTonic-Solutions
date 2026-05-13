import numpy as np

def sigmoid(z):
    z = 1 / (1 + np.exp(-z))
    return z

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n, d = X.shape

    w = np.zeros(d)
    b = 0.0

    for _ in range(n_iters):
        z = X @ w + b
        y_hat = sigmoid(z)

        dw = (1/n) * X.T @ (y_hat - y)
        db = (1/n) * np.sum(y_hat - y)
        
        w -= lr * dw
        b -= lr * db

    return (w,b)
        