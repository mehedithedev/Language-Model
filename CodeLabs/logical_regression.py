import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 

# specifiying rando seed
np.random.seed(0)

# Generate 100 samples with 2 features
X = np.random.rand(100, 2)
y = (X[:,0] + X[:, 1]> 1).astype(int) # classify based on the sum of features

print(X)
print(y)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predicitons
y_predict = model.predict(X_test)

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred=y_predict)
conf_matrix = confusion_matrix(y_test, y_pred=y_predict)

print(accuracy, conf_matrix)

# Visualization 
plt.scatter(X_test[:,0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()

