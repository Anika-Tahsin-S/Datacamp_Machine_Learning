# Given an array samples giving measurements of fish. Each row represents an individual fish. 
# The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales. 
# In order to cluster this data effectively, what needed is to standardize these features first. 
# In this exercise, a pipeline would be built to standardize and cluster the data.
# These fish measurement data were sourced from the Journal of Statistics Education.

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters = 4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)