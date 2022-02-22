##                   Selecting Relevant Features                  ##
import pandas as pd
import numpy as np

# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify = y)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))


# Create a list of redundant column names to drop
to_drop = ["category_desc", "created_date", "locality", "region", "vol_requests"]

volunteer[to_drop].head()

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis = 1)

# Print out the head of the new dataset
print(volunteer_subset.head())
## It's often easier to collect a list of columns to drop, rather than dropping them individually.




##                   Checking For Correlated Features                  ##
# Print out the column correlations of the wine dataset
print(wine.corr())

# Take a minute to find the column where the correlation value is greater than 0.75 at least twice
to_drop = "Flavanoids"

# Drop that column from the DataFrame
wine = wine.drop(to_drop, axis = 1)