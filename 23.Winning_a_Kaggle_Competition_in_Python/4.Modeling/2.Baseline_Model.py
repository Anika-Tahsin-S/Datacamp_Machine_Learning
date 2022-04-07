## NYC taxi validation
taxi_train = pd.read_csv('taxi_train.csv')
taxi_test = pd.read_csv('taxi_test.csv')

from sklearn.model_selection import train_test_split, validation_curve

validation_train, validation_test = train_test_split(taxi_train, test_size = 0.3, random_state = 123)


## Baseline Model I
import numpy as np

taxi_test['fare_amount'] = np.mean(taxi_trainfare_amount)
taxi_test[['id', 'fare_amount']].to_csv('mean_sub.csv', index = False)



## Baseline Model II
naive_prediction_groups = taxi_train.groupby('passenger_count').fare_amount.mean()
taxi_test['fare_amount'] = taxi_train.passenger_count.map(naive_prediction_groups)
taxi_test[['id', 'fare_amount']].to_csv('mean_group_sub.csv', index = False)




## Baseline Model III
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()
gb.fit(taxi_train[features], taxi_train.fare_amount)
taxi_test['fare_amount'] = gb.predict(taxi_test[features])
taxi_test[['id', 'fare_amount']].to_csv('gb_sub.csv', index = False)




## ====================================================================================================== ##