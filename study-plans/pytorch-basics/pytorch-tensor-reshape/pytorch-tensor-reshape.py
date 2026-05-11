import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype=torch.float32)

    if op == "flatten":
        return torch.flatten(x).tolist()
    elif op == "squeeze":
        return torch.squeeze(x).tolist()
    elif op == "transpose":
        return torch.transpose(x,0,1).tolist()
    