##                   Getting To Know your Data                  ##
# Import pandas
import pandas as pd

# Import so_survey_csv into so_survey_df
so_survey_df = pd.read_csv(so_survey_csv)

# Print the first five rows of the DataFrame
print(so_survey_df.head())

# Print the data type of each column
print(so_survey_df.dtypes)


# Q/A part (Part 4) #
# What type of data is the ConvertedSalary column?
# Numeric



##                   Selecting Specific Data Types                  ##
# Create subset of only the numeric columns
so_numeric_df = so_survey_df.select_dtypes(include = ['int', 'float'])

# Print the column names contained in so_survey_df_num
print(so_numeric_df.columns)