import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Seed value for generating synthetic data 
np.random.seed(0)
synth_array = np.random.rand(100, 1)  # Generate 100 samples with 1 feature
X = 2 * synth_array

m = 5
b = 3
y = m * X + b + np.random.randn(100, 1)  # Added noise

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# print(X_train, X_test, y_train, y_test)
# print(X, y)

# Train the model: 
model = LinearRegression()
model.fit(X_train, y_train) # fitting the training data from both X_train and y_train

# Getting predicted data for y while using testing data from x
# input X_test and output y_predciton 
y_prediction = model.predict(X_test)

# print(y_prediction)

# Evlaute the model by mean_squared_error while giving it the 
# y_test data and checking it with y_prediction data to see how much is it matching here

mean_square_evaluation = mean_squared_error(y_test, y_pred=y_prediction)
print(mean_square_evaluation)

# Now lets plot the data using matplotlib

plt.scatter(X_test, y_test, color = 'violet')
plt.plot(X_test, y_prediction, color='green')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')

plt.show()


# Ge the output 

coefficients = model.coef_
intercept = model.intercept_

print(f"Coefficients: {coefficients}, intercept: {intercept}, mean_square_evaluation:{mean_square_evaluation}")