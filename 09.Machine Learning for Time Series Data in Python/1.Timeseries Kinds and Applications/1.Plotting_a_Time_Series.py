##                   Reading in a Time Series with Pandas                  ##
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
data.head()

fig, ax = plt.subplots(figsize = (12, 6))
data.plot('date', 'close', ax = ax)
ax.set(title = "AAPL daily closing price")



##                   Identifying a time series                  ##
# A list of the average length of each class at the school.
# You don't have timestamps for each data point, so it is not a time series.


##                   Plotting a Time Series (I)                  ##
# Print the first 5 rows of data
print(data.head())

# Print the first 5 rows of data2
print(data2.head())

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y = 'data_values', ax=axs[0])
data2.iloc[:1000].plot(y = 'data_values', ax = axs[1])
plt.show()



##                   Plotting a Time Series (II)                  ##
# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x = "time", y = "data_values", ax = axs[0])
data2.iloc[:1000].plot(x = "time", y = "data_values", ax = axs[1])
plt.show()