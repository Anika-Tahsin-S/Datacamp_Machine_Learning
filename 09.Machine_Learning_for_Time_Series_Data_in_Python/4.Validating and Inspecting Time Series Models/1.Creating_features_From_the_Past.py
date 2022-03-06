# Time shifting data with Pandas #
print(df)

# Shift a DataFrame/Series by 3 index values towards the past
print(df.shift(3))
# Positive values roll the data backward, while negative values roll the data forward. 

# Creating a time-shifted DataFrame #
# Here we use a dictionary comprehension that creates several time-lagged versions of the data.

# data is a pandas Series containing time series data
data = pd.Series(...)

# Shifts
shifts = [0, 1, 2, 3, 4, 5, 6, 7]

# Create a dictionary of time-shifted data
many_shifts = pd.DataFrame(many_shifts)

# Convert them into a dataframe
many_shifts = pd.DataFrame(many_shifts)

# Fit the model using these input features
model = Ridge()
model.fit(many_shifts)

# Visualize the fit model coefficients
fig, ax = plt.subplots()
ax.bar(many_shifts.columns, model.coef_)
ax.set(xlabel = 'Coefficient name', ylabel = 'Coefficient value')

# Set formatting so it looks nice
plt.setp(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')




# ------------------------------------------------------------------ #


##                   Creating time-shifted Features                  ##
# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)

# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap = plt.cm.viridis)
prices_perc.iloc[:100].plot(color = 'r', lw = 2)
ax.legend(loc = 'best')
plt.show()





##                   Special Case: Auto-Regressive Models                  ##
# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))

# Fit the model
model = Ridge()
model.fit(X, y)





##                   Visualize Regression Coefficients                  ##
# Part 1
def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel = 'Coefficient name', ylabel = 'Coefficient value')
    
    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    return ax

# Part 2
# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize = (10, 5))
y.loc[:'2011-01'].plot(ax = axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax = axs[1])
plt.show()





##                   Auto-Regression with a Smoother Time Series                  ##
# prices_perc_shifted and model (updated to use a window of 40) are available in your workspace.

# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize = (10, 5))
y.loc[:'2011-01'].plot(ax = axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax = axs[1])
plt.show()
# Smoother