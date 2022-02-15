##                   Scaling Data - Investigating Columns                  ##
wine[['Ash', 'Alcalinity of ash', 'Magnesium']].describe()
# 1 and 2 are true
# Both of these statements are true according to the statistics returned by describe()




##                   Scaling Data - Standardizing Columns                  ##
# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Create the scaler
ss = StandardScaler()

# Take a subset of the DataFrame you want to scale 
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

# Apply the scaler to the DataFrame subset
wine_subset_scaled = ss.fit_transform(wine_subset)






##                   KNN on Non-Scaled Data                  ##

wine = pd.read_csv('./dataset/wine_types.csv')
X = wine.drop('Type', axis=1)
y = wine.Type

X = X.drop('Proline_log', axis=1)

X.shape

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))
# 0.6444444444444445






##                   KNN on Scaled Data                  ##
knn = KNeighborsClassifier()

# Create the scaling method.
ss = StandardScaler()

# Apply the scaling method to the dataset used for modeling.
X_scaled = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Fit the k-nearest neighbors model to the training data.
knn.fit(X_train, y_train)

# Score the model on the test data.
print(knn.score(X_test, y_test))
# 0.9555555555555556
# The increase in accuracy is worth the extra step of scaling the dataset.