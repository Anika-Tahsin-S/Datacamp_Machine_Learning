# Extracting features with rolling windows
# In pandas, the dot-aggregate method can be used to calculate many features of a window at once. 

# VIsualize the raw data
print(prices.head(3))

# Calculate a rolling window, them extract two features
feats = prices.rolling(20).aggregate([np.std, np.max]).dropna()
print(feats.head(3))
# This extracts two features for each column over time. 


# Checking Properties #
# Always plot the features you've extracted over time, as this can give you a clue for how they behave and help you spot noisy data and outliers.

# Using partial() in Python
# If we just take the mean, it returns a single value
a = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
print(np.mean(a))

# We can use the partial function to initialize np.mean with an axis parameter
from functools import partial
mean_over_first_axis = partial(np.mean, axis = 0)

print(mean_over_first_axis(a))


# Summarizing data use np.percentile #
print(np.percentile(np.linspace(0, 200), q = 20))

# Combining np.percentile() with partial functions to calculate a range of percentiles #
data = p.linspace(0, 100)

# Create a list of functions using list comprehension
percentile_funcs = [partial(np.percentile, q = ii) for ii in [20, 40, 60]]

# Calculate the output of each function in the same way
percentiles = [i_func(data) for i_func in percentile_funcs]
print(percentiles)

# Calculate multiple percentiles of a rolling window
data.rolling(20).aggregate(percentiles)

# Calculate 'date-based' features #
# Ensure our index in datatime
prices.index = pd.to_datetime(prices.index)

# extract datetime features
day_of_week_num = prices.index.weekday
print(day_of_week_num[:10])





# ------------------------------------------------------------------ #


##                   Engineering Multiple Rolling Features At Once                  ##
# Define a rolling window with Pandas, excluding the right-most datapoint of the window
prices_perc_rolling = prices_perc.rolling(20, min_periods = 5, closed = 'right')

# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean, np.std]

# Calculate these features for your rolling window object
features = prices_perc_rolling.aggregate(features_to_calculate)

# Plot the results
ax = features.loc[:"2011-01"].plot()
prices_perc.loc[:"2011-01"].plot(ax = ax, color = 'k', alpha = .2, lw = 3)
ax.legend(loc = (1.01, .6))
plt.show()




##                   Percentiles and Partial Functions                  ##
# Import partial from functools
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q = percentile) for percentile in percentiles]

# Calculate each of these quantiles on the data using a rolling window
prices_perc_rolling = prices_perc.rolling(20, min_periods = 5, closed = 'right')
features_percentiles = prices_perc_rolling.aggregate(percentile_functions)

# Plot a subset of the result
ax = features_percentiles.loc[:"2011-01"].plot(cmap = plt.cm.viridis)
ax.legend(percentiles, loc = (1.01, .5))
plt.show()





##                   Using "date" Information                  ##
# Extract date features from the data, add them as columns
prices_perc['day_of_week'] = prices_perc.index.dayofweek
prices_perc['week_of_year'] = prices_perc.index.weekofyear
prices_perc['month_of_year'] = prices_perc.index.month

# Print prices_perc
print(prices_perc)

# output:
#                   EBAY  day_of_week  week_of_year  month_of_year
#     date                                                          
#     2014-01-02  0.017938            3             1              1
#     2014-01-03  0.002268            4             1              1
#     2014-01-06 -0.027365            0             2              1
#     2014-01-07 -0.006665            1             2              1
#     2014-01-08 -0.017206            2             2              1
#     2014-01-09 -0.023270            3             2              1
#     2014-01-10 -0.022257            4             2              1
#     2014-01-13 -0.015901            0             3              1
#     2014-01-14 -0.009002            1             3              1
#     2014-01-15  0.006678            2             3              1
#     2014-01-16  0.004545            3             3              1
#     2014-01-17 -0.006145            4             3              1
#     2014-01-21  0.011801            1             4              1
#     2014-01-22  0.017089            2             4              1
#     2014-01-23  0.027897            3             4              1
#     2014-01-24  0.017463            4             4              1
#     2014-01-27 -0.010723            0             5              1
#     2014-01-28 -0.004210            1             5              1
#     2014-01-29 -0.021560            2             5              1
#     2014-01-30 -0.000356            3             5              1
#     2014-01-31  0.000772            4             5              1
#     2014-02-03 -0.014406            0             6              2
#     2014-02-04 -0.005225            1             6              2
#     2014-02-05  0.006204            2             6              2
#     2014-02-06  0.021023            3             6              2
#     2014-02-07  0.022829            4             6              2
#     2014-02-10  0.005244            0             7              2
#     2014-02-11  0.013913            1             7              2
#     2014-02-12  0.022265            2             7              2
#     2014-02-13  0.022899            3             7              2
#     ...              ...          ...           ...            ...
#     2015-11-18  0.006059            2            47             11
#     2015-11-19  0.004594            3            47             11
#     2015-11-20  0.013584            4            47             11
#     2015-11-23  0.004270            0            48             11
#     2015-11-24  0.008973            1            48             11
#     2015-11-25  0.007589            2            48             11
#     2015-11-27  0.009550            4            48             11
#     2015-11-30  0.024304            0            49             11
#     2015-12-01  0.003618            1            49             12
#     2015-12-02  0.000582            2            49             12
#     2015-12-03 -0.011950            3            49             12
#     2015-12-04  0.015626            4            49             12
#     2015-12-07  0.005337            0            50             12
#     2015-12-08 -0.003425            1            50             12
#     2015-12-09 -0.018376            2            50             12
#     2015-12-10 -0.009308            3            50             12
#     2015-12-11 -0.030131            4            50             12
#     2015-12-14 -0.027313            0            51             12
#     2015-12-15 -0.017158            1            51             12
#     2015-12-16 -0.007877            2            51             12
#     2015-12-17 -0.025614            3            51             12
#     2015-12-18 -0.044852            4            51             12
#     2015-12-21 -0.037511            0            52             12
#     2015-12-22 -0.024807            1            52             12
#     2015-12-23 -0.026665            2            52             12
#     2015-12-24 -0.028684            3            52             12
#     2015-12-28 -0.026797            0            53             12
#     2015-12-29 -0.013726            1            53             12
#     2015-12-30 -0.017296            2            53             12
#     2015-12-31 -0.024640            3            53             12
    
#     [504 rows x 4 columns]