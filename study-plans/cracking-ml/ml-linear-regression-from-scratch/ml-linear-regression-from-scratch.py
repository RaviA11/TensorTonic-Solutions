import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for epoch in range(epochs):
        y_hat = X @ w + b
        dw = (2/n) * X.T @ (y_hat - y)
        db = (2/n) * np.sum(y_hat - y)

        w -= lr * dw
        b -= lr * db

    return (w,b)
    
