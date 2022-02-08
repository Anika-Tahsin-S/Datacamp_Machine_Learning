##                   When To Standardizing                  ##
# When it is appropriate to standardize your data, which of these scenarios would you NOT want to standardize?
# Your dataset is comprised of categorical data.
# Standardization is a preprocessing task performed on numerical, continuous data.





import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
wine = pd.read_csv('wine_types.csv')
wine.head()
wine.shape
y = wine.Type
X = wine[['Proline', 'Total phenols', 'Hue', 'Nonflavanoid phenols']]

knn = KNeighborsClassifier()

##                   Modeling without Normalizing                  ##
# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))
# output: 0.5333333333333333




##                   Checking the Variance                  ##
# Check the variance of the columns in the wine dataset. Out of the four columns listed in the multiple choice section, which column is a candidate for normalization?
print(wine["Proline"].var())
# 99166.71735542436

print(wine["Alcohol"].var())
# 0.6590623278105759

print(wine["Proanthocyanins"].var())
#0.32759466768234624

print(wine["Ash"].var())
#0.07526463530756046

# The Proline column has an extremely high variance.


##                   Log Normalizing in Python                  ##
import numpy as np
# Print out the variance of the Proline column
print(wine["Proline"].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the normalized Proline column
print(wine['Proline_log'].var())
# 0.17231366191842012