import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
emails = [
    'Free money now',
    'Win a free lottery',
    'Hello friend, how are you?',
    'Meeting at noon',
    'Win money now'
]

labels= [1,1,0,0,1] 
# Specifying 1 as a match
# Specifying 0 as a unmatch

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the classifier 
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions 
y_pred = classifier.predict(X_test)

# print(f"""
# Xtrain: {X_train},
# ytrain: {y_train}

# Xtest: {X_test}
# ytest: {y_test}
#       """)

# print(y_pred)

# Check the accuracy of the model: 
accuracy = accuracy_score(y_test, y_pred=y_pred)

accuracy_percentage = f'{accuracy*100:.2f}%'

# print("Accuracy: ", accuracy_percentage)

example = [
    "Hello world",
    "Hello there",
    "Hi world"
]

Y = CountVectorizer().fit_transform(example)

array = Y.toarray()

print(Y)
print(array)