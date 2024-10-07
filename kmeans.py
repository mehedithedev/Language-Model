import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(40)
X = np.random.rand(100, 2)

# Create two clusters
X[:50] += 1
X[50:] += 2

# Apply KMeans
Kmeans = KMeans(n_clusters=2)
Kmeans.fit(X)
print(Kmeans)

# Get cluster labels and cetroids
labels = Kmeans.labels_
print(labels)
centroids = Kmeans.cluster_centers_
print(centroids)

# Plot the results
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], marker='o', c = labels, edgecolors='k', cmap='viridis', s = 50)
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', label= 'Centroids', s = 200)
plt.title('K-Means clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()
