import numpy as np
"""
Gradiant descent is a fundamental optimization algorithm 
used to minimize the cost funciton by 
iteratively updating the model parameters (Coefficients) in the direction
of the steepest descent of the cost funciton gradient

"""
def gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Perform gradient descent to minimize the cost funciton.
    Parameters:
    X: array-like, feature matrix
    y: array-like, target variable
    theta: array-like, initial coefficients
    alpha: float, learning rate
    num_iterations: int, number of iterations

    Returns:
    theta: array-like, optimized coefficeints
    cost_history: list, cost function history
    """
    m = len(y) # length of y
    cost_history = [] #initializing an empty array

    for i in range(num_iterations):
        # Calculate predictions
        predictions = np.dot(X,theta)

        # Calculate error 
        error = predictions - y

        # Calculate gradient
        gradient = np.dot(X.T, error) / m

        # update coefficients 
        theta -= alpha * gradient

        # Calculate cost
        cost = np.sum(error ** 2) / (2 * m)
        cost_history.append(cost)

    return theta, cost_history

np.random.seed(1)
X = 2 * 