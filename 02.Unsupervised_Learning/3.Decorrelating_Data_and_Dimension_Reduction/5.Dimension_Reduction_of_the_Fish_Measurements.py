# Use PCA for dimensionality reduction of the fish measurements, retaining only the 2 most important components.
# The fish measurements have already been scaled for you, and are available as scaled_samples.

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components = 2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)