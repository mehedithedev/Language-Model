import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

music_data = pd.read_csv('music.csv')

X = music_data.drop(columns=['genre']) # dropping the target values

y = music_data['genre'] # target or output values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# model = DecisionTreeClassifier()  # been using before making our own model

model = joblib.load('music-recommender.joblib')


# We already trained our model 
# model.fit(X_train, y_train)


# Creating our own model
# joblib.dump(model, 'music-recommender.joblib')

predictions = model.predict([[21,1], [22,0], [120, 0], [29,0]])
print(predictions)

# score = accuracy_score(y_test, predictions)
