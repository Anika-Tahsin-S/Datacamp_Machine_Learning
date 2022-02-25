# Cross validation types: KFold #
from sklearn.model_selection import KFold
cv = KFold(n_splits = 5)
for tr, tt in cv.split(X, y):
    ...

# Visualizing model predictions #
fig, axs = plt.subplots(2, 1)

# Plot the indices chosen for calidation on each loop
axs[0].scatter(tt, [0] * len(tt), marker = '_', s = 2, lw = 40)
axs[0].set(ylim = [-.1, .1], title = 'Test set indices (color = CV loop)', xlabel = 'Index pf raw data')

# Plot the model predictions on each iteration
axs[1].plot(model.predict(X[tt]))
axs[1].set(title = 'Test set predictions on each CV loop',
            xlabel = 'Prediction index')


# Shuffling Data #
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits = 3)
for tr, tt in cv.split(X, y):
    ...

# CV iteration #

# Import and initialize the cross-validation iterator
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits = 10)

fig, ax = plt.subplots(figsize = (10, 5))
for ii, (trr, tt) in enumerate(cv.split(X, y)):
    # Plot training and test indices
    l1 = ax.scatter(tr, [ii] * len(tr), c = [plt.cm.coolwarm(.1)],
                    marker = '_', lw = 6)
    l2 = ax.scatter(tt, [ii] * len(tt), c = [plt.cm.coolwarm(.9)],
                    marker = '_', lw = 6)
    ax.set(ylim = [10, -1], title = 'TimeSeriesSplit behavior',
            xlabel = 'data index', ylabel = 'CV iteration')
    ax.legend([l1, l2], ['Training', 'Validation'])

# Custom scoring functions in scikit-learn #

def myfunction(estimator, X, y):
    y_pred = estimator.predict(X)
    my_custom_score = my_custom_function(y_pred, y)
    return my_custom_score

def my_pearsonr(est, X, y):
    # Generate predictions and convert to a vector
    y_pred = est.predict(X).squeeze()

    # Use the numpy "corrcoef" function to calculate a correlation matrix
    my_corrcoef_matrix = np.corrcoef(y_pred, y.squeeze())

    # Return a single correlation value from the matrix
    my_corrcoef = my_corrcoef[1, 0]
    return my_corrcoef





# ------------------------------------------------------------------ #


##                   Cross-Validation with Shuffling                  ##
# Import ShuffleSplit and create the cross-validation object
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits = 10, random_state = 1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X[tt])
    score = r2_score(y[tt], prediction)
    results.append((prediction, score, tt))

# Custom function to quickly visualize predictions
visualize_predictions(results)



##                   Cross-Validation without Shuffling                  ##
# Create KFold cross-validation object
from sklearn.model_selection import KFold
cv = KFold(n_splits = 10, shuffle = False, random_state = 1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data and collect
    prediction = model.predict(X[tt])
    results.append((prediction, tt))
    
# Custom function to quickly visualize predictions
visualize_predictions(results)






##                   Time-based Cross-Validation                  ##
# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits = 10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()