# If you are confident that the missing values in your dataset are occurring at random, (in other words not being intentionally omitted) 
# The most effective and statistically sound approach to dealing with them is called 'complete case analysis' or listwise deletion.



##                   Listwise Deletion                  ##
# Part 1
# Print the number of rows and columns
print(so_survey_df.shape)
# Output: (999, 11)


# Part 2
# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna()

# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)
# Output: (264, 11)


# Part 3
# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(how = 'any', axis = 1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)
# Output: (999, 7)


# Part 4
# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset = ['Gender'])

# Print the shape of the new DataFrame
print(no_gender.shape)
# output: (693, 11)




##                   Replacing Missing Values with Constants                  ##
# Part 1
# Print the count of occurrences
print(so_survey_df['Gender'].value_counts())

# output:
#     Male                                                                         632
#     Female                                                                        53
#     Female;Male                                                                    2
#     Transgender                                                                    2
#     Female;Male;Transgender;Non-binary. genderqueer. or gender non-conforming      1
#     Male;Non-binary. genderqueer. or gender non-conforming                         1
#     Non-binary. genderqueer. or gender non-conforming                              1
#     Female;Transgender                                                             1
#     Name: Gender, dtype: int64


# part 2
# Replace missing values
so_survey_df['Gender'].fillna(value = 'Not Given', inplace = True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())