import numpy as np

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=1)
    probs = np.exp(z) / np.sum(np.exp(z), keepdims=True, axis=1 )
    return probs

def onehot_encoding(y, n_classes):
    n = len(y)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1
    return Y    

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X, dtype=float)
    y = np.array(y)
    n, d = X.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Y = onehot_encoding(y, n_classes)

    for _ in range(n_iters):
        z = X @ W + b.T
        P = softmax(z)

        dW = (1/n) * X.T @ (P - Y)
        db = (1/n) * np.sum((P - Y), axis=0)

        W -= lr * dW
        b -= lr * db

    weights = W.tolist()
    bias    = b.tolist()
    return (weights, bias)
    
