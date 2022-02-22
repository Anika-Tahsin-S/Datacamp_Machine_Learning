import pandas as pd

volunteer = pd.read_csv('volunteer_opportunities.csv')
volunteer.head()
volunteer.info()

##                   Exploring Data Types                  ##
# Which data types are present in the volunteer dataset?
volunteer.dtypes
# Float, int, and object




##                   Converting a Column Types                  ##
# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype(int)

# Look at the dtypes of the dataset
print(volunteer.dtypes)




##                   Class Imbalance                  ##
# Which descriptions occur less than 50 times in the volunteer dataset?
volunteer.category_desc.value_counts()
# Both Emergency Prepardness and Environment occur less than 50 times.



##                   Stratified Sampling                  ##
from sklearn.model_selection import train_test_split

# Create a data with all columns except category_desc
volunteer_X = volunteer.drop('category_desc', axis = 1)

# Create a category_desc labels dataset
volunteer_y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X, volunteer_y, stratify = volunteer_y)

# Print out the category_desc counts on the training y labels
print(y_train['category_desc'].value_counts())