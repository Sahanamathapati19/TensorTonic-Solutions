import numpy as np

def train_logistic_regression(X, y, lr, steps):
    # Convert inputs to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0
    
    # Numerically stable sigmoid
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Gradient Descent Loop
    for _ in range(steps):
        # Forward pass
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # Compute gradients
        dw = np.dot(X.T, (p - y)) / N
        db = np.mean(p - y)
        
        # Update parameters
        w = w - lr * dw
        b = b - lr * db
    
    return (w, float(b))