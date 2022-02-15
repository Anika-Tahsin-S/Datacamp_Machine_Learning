import pandas as pd
import numpy as np

running_times_5k = pd.read_csv("running_times_5k.csv")
running_times_5k.head()




##                   Engineering Numerical Features - Taking An Average                  ##
# Create a list of the columns to average
run_columns = ['run1', 'run2', 'run3', 'run4', 'run5']

# Use apply to create a mean column
running_times_5k["mean"] = running_times_5k.apply(lambda row: row[run_columns].mean(), axis = 1)

# Take a look at the results
print(running_times_5k)




##                   Engineering Numerical Features - Datetime                  ##
# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer["start_date_date"])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].apply(lambda row: row.month)

# Take a look at the converted and new month columns
print(volunteer[['start_date_converted', 'start_date_month']].head())