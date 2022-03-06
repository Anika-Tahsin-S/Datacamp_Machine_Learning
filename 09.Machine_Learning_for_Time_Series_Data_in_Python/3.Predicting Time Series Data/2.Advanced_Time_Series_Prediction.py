# Interpolation to fill missing data.
# Interpolation uses the values on either end of a missing window of time to infer what's in-between.

# Interpolation in Pandas #

# Return a boolean that notes where missing values are
missing = prices.isna()

# Interpolate linearly within missing windows
prices_interp = prices.interpolate('linear')

# Plot the interpolated data in red and the data w/ missing values in black
ax = price_interp.plot(c = 'r')
prices.plot(c = 'k', ax = ax, lw = 2)

# Rolling window to tranform data, to standardise its mean and variance over time.

def percent_change(values):
    """Calculates the % change between the last value and the mean of previous values"""
    # Separate the last value and all previous values into variables
    previous_values = values[:-1]
    last_value = values[-1]

    # Calculate the % difference between the last value
    # and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Plot the raw data
fig, axs = plt.subplot(1, 2, figsize = (10, 5))
ax = prices.plot(ax = axs[0])

# Calculate % change and plot
ax = prices.rolling(window = 20).aggregate(percent_change).plot(ax = axs[1])
ax.legend_.set_visible(False)


# Visualizing defination of outliners #
# Idnetify and handling outliers
# Plotting threshold

fig, axs = plt.subplot(1, 2, figsize = (10, 5))
for data, ax in zip([prices, prices_perc_change], axs):
    # Calculate the mean / standard deviation for the data
    this_mean = data.mean()
    this_std = data.std()

    # Plot the data, with a window that is 3 standard deviations around the mean
    data.plot(ax = ax)
    ax.axhline(this_mean + this_std * 3, ls = '--', c = 'r')
    ax.axhline(this_mean - this_std * 3, ls = '--', c = 'r')

# Replace outliers with the median of the remaining values #
# Center the data so the mean is 0
prices_outlier_centered = prices_outliner_perc - prices_outliner_perc.mean()

# Calculate standard deviation
std = prices_outlier_perc.std()

# Use the absolute value of each datapoint to make it easier to find outliers
outliers = np.abs(prices_outlier_centered) > (std * 3)

# Replace outliers with the median values
# Using np.nanmean since there may be nans around the outliers
prices_outlier_fixed = prices_outlier_centered.copy()
prices_outlier_fixed[outliners] = np.nanmedian(prices_outlier_fixed)


fig, axs = plt.subplots(1, 2, figsize = (10, 5))
prices_outlier_centered.plot(ax = axs[0])
prices_outlier_fixed.plot(ax = axs[1])


# -----------------------------------------------------------------------------


##                   Visualizing Messy Data                  ##
# Visualize the dataset
prices.plot(legend = False)
plt.tight_layout()
plt.show()

# Count the missing values of each time series
missing_values = prices.isna().sum()
print(missing_values)
# output:
#    symbol
#    EBAY    273
#    NVDA    502
#    YHOO    232
#    dtype: int64




##                   Imputing Messy Values                  ##
# Part 1
import pandas as pd
# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):

    # Create a boolean mask for missing values
    missing_values = prices.isna()

    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color = 'k', alpha = .6, ax = ax, legend = False)
    
    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax = ax, color = 'r', lw = 3, legend = False)
    plt.show()


# Part 2
# Interpolate using the latest non-missing value
interpolation_type = "zero"
interpolate_and_plot(prices, interpolation_type)

# Part 3
# Interpolate linearly
interpolation_type = 'linear'
interpolate_and_plot(prices, interpolation_type)

# Part 4
# Interpolate with a quadratic function
interpolation_type = "quadratic"
interpolate_and_plot(prices, interpolation_type)





##                   Transforming Raw Data                  ##
# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).apply(percent_change)
prices_perc.loc["2014":"2015"].plot()
plt.show()






##                   Handling Outliers                  ##
def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))
    
    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)
    
    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series

# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.apply(replace_outliers)
prices_perc.loc["2014":"2015"].plot()
plt.show()