import numpy as np

# -------------------------------
# Step 1: Sample Apartment Dataset
# -------------------------------
# Features: [size (sqm), number of rooms, floor]
X = np.array([
    [50, 2, 1],
    [65, 3, 2],
    [80, 3, 5],
    [120, 4, 10],
    [100, 4, 8],
    [45, 2, 0],
    [70, 3, 3]
])

# Target values: price in thousands of dollars
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
# Step 2: Normalize Features
# -------------------------------
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# -------------------------------
# Step 3: KNN with Gaussian Weights
# -------------------------------
def knn_predict_gaussian(X_train, y_train, x_query, k=3, sigma=1.0):
    """
    KNN regression with Gaussian (exponential) weights:
        w_i = exp( -||x - x_i||^2 / (2 * sigma^2) )

    Parameters:
        X_train (ndarray): normalized training features (m x n)
        y_train (ndarray): target values (m x 1)
        x_query (ndarray): single normalized query point (1 x n)
        k (int): number of neighbors to consider
        sigma (float): bandwidth parameter (controls weight spread)

    Returns:
        float: predicted value (weighted average)
    """
    # Compute Euclidean distances
    distances = np.linalg.norm(X_train - x_query, axis=1)

    # Find k nearest neighbors
    nearest_indices = np.argsort(distances)[:k]
    nearest_X = X_train[nearest_indices]
    nearest_y = y_train[nearest_indices]

    # Compute squared distances
    squared_distances = np.sum((nearest_X - x_query) ** 2, axis=1)

    # Compute Gaussian weights
    weights = np.exp(-squared_distances / (2 * sigma ** 2))

    # Compute weighted prediction
    weighted_prediction = np.sum(weights * nearest_y.flatten()) / np.sum(weights)
    return weighted_prediction

# -------------------------------
# Step 4: Predict New Apartment Price
# -------------------------------
# New input: 90 sqm, 3 rooms, 5th floor
X_new = np.array([[90, 3, 5]])
X_new_norm = (X_new - X_mean) / X_std

# Predict using KNN with Gaussian weights
k = 3
sigma = 1.0
predicted_price = knn_predict_gaussian(X_norm, y, X_new_norm[0], k=k, sigma=sigma)

# -------------------------------
# Step 5: Print Result
# -------------------------------
print(f"Predicted price (KNN with Gaussian weights, k={k}, σ={sigma}): ${predicted_price}")
