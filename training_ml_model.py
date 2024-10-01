import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')

X = music_data.drop(columns=['genre']) # dropping the target values

y = music_data['genre'] # target or output values

model = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)

score = accuracy_score(y_test, predictions)

print(score)