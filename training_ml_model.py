import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
music_data = pd.read_csv('music.csv')

# Define features and target
X = music_data.drop(columns=['genre'])  # Dropping the target values
y = music_data['genre']  # Target or output values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Load the pre-trained model
model = joblib.load('music-recommender.joblib')

# Generate predictions for the entire test set
predictions = model.predict(X_test)

# Calculate the accuracy score
score = accuracy_score(y_test, predictions)

print(score)