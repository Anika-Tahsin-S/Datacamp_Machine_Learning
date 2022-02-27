##                   Dealing with Stray Characters (I)                  ##
# Part 1
# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')

# Part 2
# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$', '')


##                   Dealing with Stray Characters (II)                  ##
# Part 1
# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors = 'coerce')

# Find the indexes of missing values
idx = numeric_vals.isna()

# Print the relevant rows
print(so_survey_df['RawSalary'][idx])
# output:
#     0            NaN
#     2            NaN
#     4      £41671.00
#     6            NaN
#     8            NaN
#              ...    
#     989          NaN
#     990          NaN
#     992          NaN
#     994          NaN
#     997          NaN
#     Name: RawSalary, Length: 401, dtype: object


# Part 2
# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£', '')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')

# Print the column
print(so_survey_df['RawSalary'])
# output:
#     0            NaN
#     1        70841.0
#     2            NaN
#     3        21426.0
#     4        41671.0
#              ...    
#     994          NaN
#     995      58746.0
#     996      55000.0
#     997          NaN
#     998    1000000.0
#     Name: RawSalary, Length: 999, dtype: float64





##                   Method Chaining                  ##
# Method chaining
df['column'] = df['column'].method1().method2().method3()

# Same as 
df['column'] = df['column'].method1()
df['column'] = df['column'].method2()
df['column'] = df['column'].method3()


# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary']\
                              .str.replace(',', '')\
                              .str.replace('$', '')\
                              .str.replace('£', '')\
                              .astype('float')
 
# Print the RawSalary column
print(so_survey_df['RawSalary'])