import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Sample Apartment Dataset
# -------------------------------
# Features: [size, number of rooms, floor]
X = np.array([
    [50, 2, 1],
    [65, 3, 2],
    [80, 3, 5],
    [120, 4, 10],
    [100, 4, 8],
    [45, 2, 0],
    [70, 3, 3]
])

# Target: price in thousands of dollars
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


# No need to normalize y for KNN

# -------------------------------
# Step 3: KNN Prediction Function
# -------------------------------
def knn_predict(X_train, y_train, x_query, k=3):
    """
    Predict the target for a single input using K-Nearest Neighbors (regression).

    Parameters:
        X_train: (m x n) Normalized training features
        y_train: (m x 1) Training target values
        x_query: (1 x n) Normalized feature of the new sample
        k: Number of neighbors to consider

    Returns:
        float: Predicted value (average of k nearest y values)
    """
    # Compute Euclidean distances
    distances = np.linalg.norm(X_train - x_query, axis=1)

    # Get indices of the k nearest neighbors
    nearest_indices = np.argsort(distances)[:k]

    # Average their y values
    return np.mean(y_train[nearest_indices])


# -------------------------------
# Step 4: Predict Price for New Apartment
# -------------------------------
X_new = np.array([[90, 3, 5]])
X_new_norm = (X_new - X_mean) / X_std

k = 3  # Number of neighbors to consider
predicted_price = knn_predict(X_norm, y, X_new_norm[0], k)

print(f"KNN (k={k}) predicted price for 90sqm, 3 rooms, floor 5: ${predicted_price}")
