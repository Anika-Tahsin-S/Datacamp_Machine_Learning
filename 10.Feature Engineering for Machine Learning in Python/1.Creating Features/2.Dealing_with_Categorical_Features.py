# One-hot encoding
pd.get_dummies(df, columns = ['Country'], prefix = 'C')

# Dummy encoding
pd.get_dummies(df, columns = ['Country'], drop_first = True, prefix = 'C')



##                   One-hot Encoding and Dummy Variables                  ##
import pandas as pd

# Convert the Country column to a one hot encoded Data Frame
one_hot_encoded = pd.get_dummies(so_survey_df, columns = ['Country'], prefix = 'OH')

# Print the columns names
print(one_hot_encoded.columns)


# Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns = ['Country'], drop_first = True, prefix = 'DM')

# Print the columns names
print(dummy.columns)




##                   Dealing with Uncommon Categories                  ##
# Part 1
# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Print the count values for each category
print(country_counts)
# output:
#   South Africa    166
#     USA             164
#     Spain           134
#     Sweeden         119
#     France          115
#     Russia           97
#     UK               95
#     India            95
#     Ukraine           9
#     Ireland           5
#     Name: Country, dtype: int64




# Part 2
# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Print the top 5 rows in the mask series
print(mask.head())
# output:
#     0      False
#     1      False
#     2      False
#     3      False
#     4      False
#            ...  
#     994    False
#     995    False
#     996    False
#     997    False
#     998    False
#     Name: Country, Length: 999, dtype: bool


# Part 3
# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Label all other categories as Other
countries[mask] = 'Other'

# Print the updated category counts
print(pd.value_counts(countries))
# output:
#     South Africa    166
#     USA             164
#     Spain           134
#     Sweeden         119
#     France          115
#     Russia           97
#     UK               95
#     India            95
#     Other            14
#     Name: Country, dtype: int64