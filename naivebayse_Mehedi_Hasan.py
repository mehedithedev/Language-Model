import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = {
    'document': [
        'I love playing football',
        'The new smartphone is amazing',
        'Football match tonight',
        'Latest technology trends',
        'Smartphone reviews',
        'Football fans cheering'
    ],
    'category':[
        0,1,0,1,1,0
    ] # 0 = Sports category, 
      # 1 = Technology category
}

df = pd.DataFrame(data)

print(df)

X = df['document']
y = df['category']

# CountVectorizer tokenizes the test (splits into words) and creates binary vector
# Representing the present (1) or absence(0) 
vectorizer = CountVectorizer(binary=True)
X_vectorized = vectorizer.fit_transform(X) # Ensures that we get a one-hot encoding style matrix

print(X_vectorized.toarray())

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size = 0.2, random_state = 42)

# Initialize the Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier with the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accurace of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model:{accuracy}")

# Create two new documents and classify them either as Sports or Technology
sample_documents = [
    'Football fever grisps the nation',
    'Exciting new smartphone launch event'
]

# Convert the sample documents into binary feature vectors
smample_X_documents = vectorizer.transform(sample_documents)

print(smample_X_documents.toarray())

# Use the trained model to predict the category for these new documents
predictions = clf.predict(smample_X_documents)
print(f"Predictions for the sample documents: {predictions}")