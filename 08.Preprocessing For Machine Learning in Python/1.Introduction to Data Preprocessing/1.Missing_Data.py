import pandas as pd

volunteer = pd.read_csv('volunteer_opportunities.csv')
volunteer.head()
volunteer.info()

##                   Missing Data - Columns                  ##
# How many features are in the original dataset, and how many features are in the set after columns with at least 3 missing values are removed?
volunteer.dropna(axis = 1, thresh = 3).shape
# (665, 24)
volunteer.shape
# (665, 35)
# 35, 24. A lot of operations are done on a column basis, so it's useful to remember axis=1 when working with Pandas.




##                   Missing Data - Rows                  ##
# Check how many values are missing in the category_desc column
print(volunteer['category_desc'].isnull().sum())

# Subset the volunteer dataset
volunteer_subset = volunteer[volunteer['category_desc'].notnull()]

# Print out the shape of the subset
print(volunteer_subset.shape)
# Remember that you can use boolean indexing to effectively subset DataFrames.