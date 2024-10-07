import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Generate sample data
np = np.random.seed(0)
X = np.random.rand(100,2) # Generate 100 samples with 2 features
y = (X[:,0] + X[:, 1] > 1).astype(int) # Classify based on the sum of features
