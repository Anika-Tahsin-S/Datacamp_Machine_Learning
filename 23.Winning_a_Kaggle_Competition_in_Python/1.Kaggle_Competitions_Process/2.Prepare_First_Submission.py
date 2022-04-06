import pandas as pd

taxi_train = pd.read_csv('taxi_train.csv')
taxi_train.columns.to_list()

## Problem type
import matplotlib.pyplot as plt

taxi_train.fare_amount.hist(bins = 30, alpha = 0.5)
plt.show()

## Build a model
from sklearn.linear_model import LineraRegression

lr = LineraRegression()

lr.fit(X = taxi_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']], y = taxi_train['fare_amount'])

## Predict on test set
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

taxi_data['fare_amount'] = lr.predict(taxi_test[features])

## Prepare submission
# Read submission file
taxi_sample_sub = pd.read_csv('taxi_sample_submission.csv')
taxi_sample_sub.head(1)

# Prepare submission file
taxi_submission = taxi_test[['key', 'fare_amount']]

# Save submission file as .csv
taxi_submission.to_csv('first_sub.csv', index = False)




## ====================================================================================================== ##