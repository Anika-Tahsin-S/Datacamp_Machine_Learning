##                   How sparse is my data?                  ##
# Part 1
# Subset the DataFrame
sub_df = so_survey_df[['Age', 'Gender']]

# Print the number of non-missing values
print(sub_df.info())

# Part 2
# Based on the results, how many non-missing entries are there in the Gender column?
# output:
#     <class 'pandas.core.frame.DataFrame'>
#     RangeIndex: 999 entries, 0 to 998
#     Data columns (total 2 columns):
#      #   Column  Non-Null Count  Dtype 
#     ---  ------  --------------  ----- 
#      0   Age     999 non-null    int64 
#      1   Gender  693 non-null    object
#     dtypes: int64(1), object(1)
#     memory usage: 15.7+ KB
#     None
# Answer: 693





##                   Finding the Missing Values                  ##
# Part 1
# Print the top 10 entries of the DataFrame
print(sub_df.head(10))

# Part 2
# Print the locations of the missing values
print(sub_df.head(10).isnull())

# Part 3
# Print the locations of the non-missing values
print(sub_df.head(10).notnull())


