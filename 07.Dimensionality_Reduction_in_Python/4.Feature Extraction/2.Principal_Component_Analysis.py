import pandas as pd
import numpy as np

#################################################
ansur_df = ansur_df[['stature_m', 'buttockheight', 'waistcircumference', 'shouldercircumference']]
# Transform the test set with the pre-fitted scaler
X_test_std = scaler.transform(X_test)

# Calculate the coefficient of determination (R squared) on X_test_std
r_squared = la.score(X_test_std, y_test)
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))

# Create a list that has True values when coefficients equal 0
zero_coef = la.coef_ == 0

# Calculate how many features have a zero coefficient
n_ignored = sum(zero_coef)
print("The model has ignored {} out of {} features.".format(n_ignored, len(la.coef_)))
#################################################



##                   Calculating Principal Components                  ##
# Part 1
# Create a pairplot to inspect ansur_df
sns.pairplot(ansur_df)

plt.show()

# Part 2, 3, 4
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create the scaler
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

# Create a pairplot of the principal component dataframe
sns.pairplot(pc_df)
plt.show()




##                   PCA on a Larger Dataset                  ##
# PCA on a somewhat larger ANSUR datasample with 13 dimensions, once again pre-loaded as ansur_df
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Apply PCA
pca = PCA()
pca.fit(ansur_std)



##                   PCA Explained Variance                  ##
# Part 1
# Inspect the explained variance ratio per component
print(pca.explained_variance_ratio_)

# Part 2
# How much of the variance is explained by the 4th principal component?
# About 3.77%

# Part 3
# Print the cumulative sum of the explained variance ratio
print(pca.explained_variance_ratio_.cumsum())

# Part 4
# What's the lowest number of principal components you should keep if you don't want to lose more than 10% of explained variance during dimensionality reduction?
# Using just 4 principal components we can explain more than 90% of the variance in the 13 feature dataset.