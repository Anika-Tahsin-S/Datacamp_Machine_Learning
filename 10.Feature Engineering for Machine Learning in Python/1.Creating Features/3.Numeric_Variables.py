##                   Binarizing Columns                  ##
# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df.ConvertedSalary > 0, 'Paid_Job'] = 1

# Print the first five rows of the columns
print(so_survey_df[['Paid_Job', 'ConvertedSalary']].head())




##                   Binarizing Values                  ##
# Part 1
# Bin the continuous variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], 5)

# Print the first 5 rows of the equal_binned column
print(so_survey_df[['equal_binned', 'ConvertedSalary']].head())

# output:
#               equal_binned  ConvertedSalary
#     0  (-2000.0, 400000.0]              0.0
#     1  (-2000.0, 400000.0]          70841.0
#     2  (-2000.0, 400000.0]              0.0
#     3  (-2000.0, 400000.0]          21426.0
#     4  (-2000.0, 400000.0]          41671.0

# Part 2
# Import numpy
import numpy as np

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'], 
                                         bins, labels)

# Print the first 5 rows of the boundary_binned column
print(so_survey_df[['boundary_binned', 'ConvertedSalary']].head())

# Output
#        boundary_binned  ConvertedSalary
# 0      (-inf, 10000.0]              0.0
# 1  (50000.0, 100000.0]          70841.0
# 2      (-inf, 10000.0]              0.0
# 3   (10000.0, 50000.0]          21426.0
# 4   (10000.0, 50000.0]          41671.0