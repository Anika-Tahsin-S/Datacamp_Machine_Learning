# One-hot encoding with pandas
import pandas as pd
df = pd.get_dummies(df, columns = ['sex'])


##                   Data Format for ML Algorithms                  ##
# In this exercise, you have been given four dataframes df1, df2, df3 and df4. The final column of each dataframe is the predictor variable and the rest of the columns are training features.
# Using the console, determine which dataframe is in a suitable format to be trained by a classifier.
# Answer: df3
# This dataframe has numerical training features and the predictor variable is a class. Therefore, it is in a suitable format for applying a classification algorithm.




##                   One-hot Encoding                  ##
# Part 1
# Print the features of df1
print(df1.columns)
# output:
#     Index(['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'label'], dtype='object')


# Part 2
# Perform one-hot encoding
df1 = pd.get_dummies(df1, columns = ['feature 5'])


# Part 3
# Print the new features of df1
print(df1.columns)

# Print first five rows of df1
print(df1.head())
# output:
#     Index(['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'label'], dtype='object')
#     Index(['feature 1', 'feature 2', 'feature 3', 'feature 4', 'label', 'feature 5_female', 'feature 5_male'], dtype='object')
#        feature 1  feature 2  feature 3  feature 4  label  feature 5_female  feature 5_male
#     0    29.0000          0          0   211.3375      1                 1               0
#     1     0.9167          1          2   151.5500      1                 0               1
#     2     2.0000          1          2   151.5500      0                 1               0
#     3    30.0000          1          2   151.5500      0                 0               1
#     4    25.0000          1          2   151.5500      0                 1               0