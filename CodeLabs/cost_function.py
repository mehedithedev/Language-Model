""""
Cost functions play a crucial role in traininig machine learning models
by 
quantifying the error betwen predicted values and actual values


"""



import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.
    Parameters: 
    y_true: array-like, true values
    y_pred: array-like, predicted values

    Returns:
    mse: float, mean squared error
    """

    N = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / N
    return mse

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100,1) # Generate 100 random numbers betwen 0 and 2
y = 4 + 3 * X + np.random.randn(100,1) # Linear relationship with noise

# print(X,"\n \n", y, "\n\n\n")

# Split the data between X and y test and train
# Import sckitit learn train_test_split module:
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
# Import scikit learn LinearRegression module from linera_model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# fit X_train and y_train to train the model 
model.fit(X_train, y_train)

# make predictions based on test data X_test
y_pred = model.predict(X_test)

print(y_test, "\n\n" ,y_pred)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred=y_pred)

# Show the mean squared error: 
print(f"Mean Squared Error (MSE): {mse})")