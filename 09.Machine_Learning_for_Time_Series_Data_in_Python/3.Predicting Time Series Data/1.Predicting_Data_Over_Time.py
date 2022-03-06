# Two ways to Compare Timeseries data  #
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)

# Make a line plot for each timeseries
axs[0].plot(x, c = 'k', lw = 3, alpha = .2)
axs[0].plot(y)
axs[0].set(xlabel = 'time', title = 'X values = time')

# Encode time as color in a scatterplot
axs[1].scatter(x_long, y_long, c = np.arange(len(x_long)), cmap = 'virdis')
axs[1].set(xlabel = 'x', ylabel = 'y',  title = 'Color = time')


# Visualize predictions with scikit-learn #
from sklearn.linear_model import LinearRegression, Ridge

model = LinearRegression()
model.fit(X, y)
model.predict(X)

alphas = [.1, 1e2, 1e3]
ax.plot(y_test, color = 'k', alpha = .3, lw = 3)
for ii, alppha in enumarate(alphas):
    y_predicted = Ridge(alpha = alpha).fit(X_train, y_train).predict(X_test)
    ax.plot(y_predict, c = cmap(ii / len(alphas)))
ax.legend(['True values', 'Model 1', 'Model 2', 'Model 3'])
ax.set(xlabel = 'Time')

# Coeeficient of Determination (R^2):
# 1 - [error(model) / variance(testdata)]
# R^2 in scikit-learn
from sklearn.metrics import r2_score
print(r2_score(y_predicted, y_test))





##                   Introducing The Dataset                  ##
# Part 1
# Plot the raw values over time
prices.plot()
plt.show()

# Part 2
# Scatterplot with one company per axis
prices.plot.scatter('EBAY', 'YHOO')
plt.show()

# Part 3
# Scatterplot with color relating to time
prices.plot.scatter('EBAY', 'YHOO', c = prices.index, 
                    cmap = plt.cm.viridis, colorbar = False)
plt.show()





##                   Fitting a Simple Regression Model                  ##
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Use stock symbols to extract training data
X = all_prices[["EBAY", "NVDA", "YHOO"]]
y = all_prices[["AAPL"]]

# Fit and score the model with cross-validation
scores = cross_val_score(Ridge(), X, y, cv = 3)
print(scores)
# output: [-6.09050633 -0.3179172  -3.72957284]




##                   Visualizing Predicted Values                  ##
# Part 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size = .8, shuffle = False, random_state = 1)

# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

# Part 2
# Visualize our predictions along with the "true" values, and print the score
fig, ax = plt.subplots(figsize = (15, 5))
ax.plot(y_test, color = 'k', lw = 3)
ax.plot(predictions, color = 'r', lw = 2)
plt.show()