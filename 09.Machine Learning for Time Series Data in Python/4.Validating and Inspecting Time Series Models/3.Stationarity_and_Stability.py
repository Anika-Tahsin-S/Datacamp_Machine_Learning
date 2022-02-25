# Bootstrapping the mean #
from sklearn.utils import resample

# cv_coefficients has shape (n_cv_folds, n_coefficients)
n_boots = 100
bootstrap_means = np.zeros(n_boots, n_coefficients)
for ii in range(n_boots):
    # Generate random indices for our data with replacement then take the sample mean
    random_sample = resample(cv_coefficients)
    bootstrap_means[ii] = random_sample.mean(axis = 0)

# Compute the percentiles of choice for the bootstrapped means
percentiles = np.percentile(bootstrap_means, (2.5, 97.5), axis = 0)

fig, ax = plt.subplots()
ax.scatter(many_shifts.columns, percentiles[0], marker = '_', s = 200)
ax.scatter(many_shifts.columns, percentiles[1], marker = '_', s = 200)

# Model performance over time #
def my_corrcoef(est, X, y):
    """Return the correlation coefficient between model predictions and a validation set."""
    return np.corrcoef(y, est.predict(x))[1, 0]

# Grab the date of the first index of each validation set
first_indices = [data.index[tt[0]] for tr, tt in cv.split(X, y)]

# Calculate the CV scores and convert to a Pandas Series
cv_scores = cross_val_score(model, X, y, cv = cv, scoring = my_corrcoef)
cv_scores = pd.Series(cv_scores, index = first_indices)

fig, ax = plt.subplots(2, 1, figsize = (10, 5), sharex = True)

# Calculate a rolling mean of scores over time
cv_scores_mean = cv_scores.rolling(210, min_periods = 1).mean()
cv_scores.plot(ax = axs[0])
axs[0].set(title = 'Validation scores (correlation)', ylim = [0, 1])

# Plot raw data
data.plot(ax = axs[1])
axs[1].set(title = 'Validation data')
 # There is a clear dip in the middle, probably because the statistics of the data changed.
# Sol 1: Restrict the size of the training window

# Only keep the last 100 datapoints in the training data
window = 100

# Initialize the CV with this window size
cv = TimeSeriesSplit(n_splits = 10, max_train_size = window)







# ------------------------------------------------------------------ #


##                   Stationarity                  ##
# Which of the following time series do you think are not stationary?
# B and C. 
# C begins to trend upward partway through, while B shows a large increase in variance mid-way through, making both of them non-stationary.




##                   Bootstrapping a Confidence Interval                  ##
from sklearn.utils import resample

def bootstrap_interval(data, percentiles = (2.5, 97.5), n_boots = 100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create our empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for our data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis = 0)
        
    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis = 0)
    return percentiles




##                   Calculating Variability in Model Coefficients                  ##
# Part 1
# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits = n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_

# Part 2
# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients)

# Plot it
fig, ax = plt.subplots()
ax.scatter(feature_names, bootstrapped_interval[0], marker = '_', lw = 3)
ax.scatter(feature_names, bootstrapped_interval[1], marker = '_', lw = 3)
ax.set(title = '95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
plt.show()





##                   Visualizing Model Score Variability Over Time                  ##
# Part 1
from sklearn.model_selection import cross_val_score

# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)

# Convert to a Pandas Series object
scores_series = pd.Series(scores, index=times_scores, name='score')

# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles = 2.5))
scores_hi = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles = 97.5))

# Part 2
# Plot the results
fig, ax = plt.subplots()
scores_lo.plot(ax = ax, label = "Lower confidence interval")
scores_hi.plot(ax = ax, label = "Upper confidence interval")
ax.legend()
plt.show()
# You plotted a rolling confidence interval for scores over time. 
# This is useful in seeing when your model predictions are correct.





##                   Accounting For Non-Stationarity                  ##
# Part 1
# Pre-initialize window sizes
window_sizes = [25, 50, 75, 100]

# Create an empty DataFrame to collect the stores
all_scores = pd.DataFrame(index = times_scores)

# Generate scores for each split to see how the model performs over time
for window in window_sizes:
    # Create cross-validation object using a limited lookback window
    cv = TimeSeriesSplit(n_splits=100, max_train_size=window)
    
    # Calculate scores across all CV splits and collect them in a DataFrame
    this_scores = cross_val_score(model, X, y, cv = cv, scoring = my_pearsonr)
    all_scores['Length {}'.format(window)] = this_scores

# Part 2
# Visualize the scores
ax = all_scores.rolling(10).mean().plot(cmap = plt.cm.coolwarm)
ax.set(title = 'Scores for multiple windows', ylabel = 'Correlation (r)')
plt.show()