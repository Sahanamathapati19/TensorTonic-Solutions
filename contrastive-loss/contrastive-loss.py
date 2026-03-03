import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    # Convert to numpy arrays
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    y = np.array(y, dtype=float)
    
    # Handle single sample (D,) → (1, D)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    
    # Validate shapes
    if a.shape != b.shape:
        raise ValueError("a and b must have same shape")
    
    N = a.shape[0]
    
    # Ensure y shape is (N,)
    y = y.reshape(-1)
    if y.shape[0] != N:
        raise ValueError("y must match number of samples")
    
    # Validate y values
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")
    
    # Compute Euclidean distance
    d = np.linalg.norm(a - b, axis=1)
    
    # Compute loss per sample
    loss = y * (d ** 2) + (1 - y) * np.maximum(0, margin - d) ** 2
    
    # Reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")