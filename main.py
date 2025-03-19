import numpy as np

def compute_cost(X, y, weights):
    m = y.shape[0]
    predictions = X.dot(weights).reshape(-1, 1)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def gradient(X, y, weights):
    m = y.shape[0]
    predictions = X.dot(weights).reshape(-1, 1)
    grad = (1 / m) * X.T.dot(predictions - y)
    return grad.flatten()

def compute_linear_regression(X, y, alpha=0.01, max_iterations=1000):
    m, n = X.shape
    # Add bias column to the feature matrix
    X_bias = np.concatenate((np.ones((m, 1)), X), axis=1)
    weights = np.zeros(n + 1)  # Initialize weights for bias

    for _ in range(max_iterations):
        grad = gradient(X_bias, y, weights)
        weights -= alpha * grad

    final_cost = compute_cost(X_bias, y, weights)
    return weights, final_cost
