# Linear Regression Functions

This repository contains a minimal implementation of linear regression using gradient descent. It includes three primary functions:

- **`compute_cost`**: Calculates the mean squared error (MSE) cost for a given set of predictions.
- **`gradient`**: Computes the gradient of the cost function with respect to the model weights.
- **`compute_linear_regression`**: A wrapper function that performs linear regression by adding a bias term to the input data and iteratively updating the weights using gradient descent.

## Files

- **`linear_regression.py`**: Contains the implementation of the functions.
- **`README.md`**: Provides an overview of the project and usage instructions.

## Usage

1. **Importing the functions:**

   ```python
   from linear_regression import compute_cost, gradient, compute_linear_regression
   ```

2. **Preparing your data:**

   - `X`: A NumPy array of shape `(m, n)` representing the features.
   - `y`: A NumPy array of shape `(m, 1)` representing the target values.

3. **Training the model:**

   ```python
   # Example:
   import numpy as np
   X = np.random.randn(100, 3)  # 100 samples, 3 features
   y = 5 + X.dot(np.array([[2], [-3], [1]])) + np.random.randn(100, 1) * 0.5  # Linear model with noise

   weights, final_cost = compute_linear_regression(X, y, alpha=0.01, max_iterations=1000)
   print("Learned weights (including bias):", weights)
   print("Final cost:", final_cost)
   ```

## Function Details

- **`compute_cost(X, y, weights)`**:
  - **Parameters:**
    - `X`: The design matrix including the bias term (shape `(m, n+1)`).
    - `y`: The target values (shape `(m, 1)`).
    - `weights`: The weight vector (shape `(n+1,)`).
  - **Returns:** The mean squared error cost as a float.

- **`gradient(X, y, weights)`**:
  - **Parameters:**
    - `X`: The design matrix including the bias term (shape `(m, n+1)`).
    - `y`: The target values (shape `(m, 1)`).
    - `weights`: The weight vector (shape `(n+1,)`).
  - **Returns:** The gradient as a 1D NumPy array of shape `(n+1,)`.

- **`compute_linear_regression(X, y, alpha=0.01, max_iterations=1000)`**:
  - **Parameters:**
    - `X`: The feature matrix (shape `(m, n)`).
    - `y`: The target values (shape `(m, 1)`).
    - `alpha`: The learning rate for gradient descent.
    - `max_iterations`: The maximum number of iterations to run gradient descent.
  - **Returns:** A tuple `(weights, final_cost)` where:
    - `weights` is a 1D NumPy array of shape `(n+1,)` including the bias weight.
    - `final_cost` is the cost computed after the final iteration.

## Dependencies

- [NumPy](https://numpy.org/) for numerical computations.
