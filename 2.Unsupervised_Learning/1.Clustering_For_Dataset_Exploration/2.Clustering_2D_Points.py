# Create a KMeans model to find 3 clusters, and fit it to the data points from the previous exercise. 
# After the model has been fit, obtain the cluster labels for some new points using the .predict() method.
# Given the array points from the previous exercise, and also an array new_points.
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters = 3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)