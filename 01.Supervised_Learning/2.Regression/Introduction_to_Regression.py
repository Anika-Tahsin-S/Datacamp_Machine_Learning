boston = pd.read_csv('boston.csv')
print(boston.head())

#Creating feature and target arrays
X = boston.drop('MEDV', axis = 1).values
y = boston['MEDV'].values

#Predicting house value from a single feature
X_rooms = X[:,5]
type(X_rooms)
type(y)

y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

#Plotting house valuse vs. number of rooms
plt.scattter(X_rooms, y)
plt.ylable('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()

# Fitting regression model
import numpy as no
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linespace(min(X_rooms), max(X_rooms)).reshape(-1, 1)

plt.scatter(X_rooms, y, color = 'blue')
plt.plot(prediction_space, reg.predict(prediction_space), color = 'black', linewidth = 3)
plt.show()