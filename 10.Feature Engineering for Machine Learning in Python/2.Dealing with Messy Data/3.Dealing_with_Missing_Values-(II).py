##                   Filling Continuous Missing Values                  ##
# Part 1
# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df.StackOverflowJobsRecommend.head())
# output:
#     0    NaN
#     1    7.0
#     2    8.0
#     3    NaN
#     4    8.0
#     Name: StackOverflowJobsRecommend, dtype: float64


# Part 2
# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace = True)

# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head())
# output:
#     0    7.062
#     1    7.000
#     2    8.000
#     3    7.062
#     4    8.000
#     Name: StackOverflowJobsRecommend, dtype: float64

# Part 3
# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())
# output:
#     0    7.0
#     1    7.0
#     2    8.0
#     3    7.0
#     4    8.0
#     Name: StackOverflowJobsRecommend, dtype: float64



##                   Imputing Values in Predictive Models                  ##
# When working with predictive models you will often have a separate train and test DataFrames. 
# In these cases you want to ensure no information from your test set leaks into your train set. 
# When filling missing values in data to be used in these situations how should approach the two datasets?

# Answer: Apply the measures of central tendency (mean/median etc.) calculated on the train set to both the train and test sets.

# Values calculated on the train test should be applied to both DataFrames.