import numpy as np

def activation_functions(x, activation):
    if activation == "relu":
        out = np.maximum(0.0, x)
        dout = 1.0 if out > 0 else 0.0
    elif activation == "sigmoid":
        out = 1 / (1 + np.exp(-x))
        dout = out * (1 - out)
    elif activation == "tanh":
        out = np.tanh(x)
        dout = 1 - out**2
    elif activation == "leaky_relu":
        if x > 0:
            out = x
            dout = 1.0
        else:
            out = 0.01 * x
            dout = 0.01
    elif activation == "gelu":
        c = np.sqrt(2.0 / np.pi)
        u = c * (x + 0.044715 * x**3)
        t = np.tanh(u)
        w = 1.0 + 3 * 0.044715 * x**2
        out = 0.5 * x * (1.0 + t)
        dout = 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * c * w
    elif activation == "swish":
        s = 1.0 / (1.0 + np.exp(-x))
        out = x * s
        dout = s + x * (s * (1 - s))
    else:
        raise ValueError(f"unknown activation: {activation}")

    return [round(float(out), 4), round(float(dout), 4)]