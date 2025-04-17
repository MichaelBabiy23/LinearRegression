import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Cost Function: Mean Squared Error
# -------------------------------
def compute_cost(X, y, weights):
    m = y.shape[0]
    predictions = X.dot(weights).reshape(-1, 1)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# -------------------------------
# Gradient Function
# -------------------------------
def gradient(X, y, weights):
    m = y.shape[0]
    predictions = X.dot(weights).reshape(-1, 1)
    grad = (1 / m) * X.T.dot(predictions - y)
    return grad.flatten()

# -------------------------------
# Linear Regression via Gradient Descent
# -------------------------------
def compute_linear_regression(X, y, alpha=0.01, max_iterations=1000):
    m, n = X.shape
    X_bias = np.concatenate((np.ones((m, 1)), X), axis=1)  # Add bias column
    weights = np.zeros(n + 1)  # Initialize weights including bias
    cost_history = []

    for _ in range(max_iterations):
        grad = gradient(X_bias, y, weights)
        weights -= alpha * grad
        cost = compute_cost(X_bias, y, weights)
        cost_history.append(cost)

    return weights, cost_history[-1], cost_history

# -------------------------------
# Step 1: Sample Apartment Dataset
# -------------------------------
# Features: [size, number of rooms, floor]
# Target: price in thousands of dollars
X = np.array([
    [50, 2, 1],
    [65, 3, 2],
    [80, 3, 5],
    [120, 4, 10],
    [100, 4, 8],
    [45, 2, 0],
    [70, 3, 3]
])

y = np.array([
    [200],
    [250],
    [300],
    [500],
    [450],
    [180],
    [275]
])

# -------------------------------
# Step 2: Normalize Features and Target
# -------------------------------
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_norm = (y - y_mean) / y_std

# -------------------------------
# Step 3: Train Model
# -------------------------------
weights, final_cost, cost_history = compute_linear_regression(
    X_norm, y_norm, alpha=0.01, max_iterations=1000
)

# -------------------------------
# Step 4: Plot Cost over Iterations
# -------------------------------
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.title("Cost Function over Iterations")
plt.grid(True)
plt.show()

# -------------------------------
# Step 5: Predict Price for New Apartment
# -------------------------------
X_new = np.array([[90, 3, 5]])  # 90 sqm, 3 rooms, 5th floor
X_new_norm = (X_new - X_mean) / X_std
X_new_bias = np.concatenate((np.ones((1, 1)), X_new_norm), axis=1)

# Predict normalized and rescale back
y_pred_norm = X_new_bias.dot(weights)
y_pred = y_pred_norm * y_std + y_mean

print(f"Predicted price for 90sqm, 3 rooms, floor 5: ${y_pred[0]}")

