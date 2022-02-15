##                   Using PCA                  ##
from sklearn.decomposition import PCA

# Set up PCA and the X vector for diminsionality reduction
pca = PCA()
wine_X = wine.drop("Type", axis = 1)

# Apply PCA to the wine dataset X vector
transformed_X = pca.fit_transform(wine_X)
print(transformed_X)
# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)



##                   Training a Model With PCA                  ##
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()

# Split the transformed X and the y labels into training and test sets
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(transformed_X, y)

# Fit knn to the training data
knn.fit(X_wine_train, y_wine_train)

# Score knn on the test data and print it out
print(knn.score(X_wine_test, y_wine_test))