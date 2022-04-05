# Build Regression Model
from pyspark.ml.regression import LinearRegression

# Create a regression object and train on training data
regression = LinearRegression(labelCol = 'consumption')

regression = regression.fit(flights_train)
predictions = regression.transform(cars_test)


## Calculate the RMSE
from pyspark.ml.evaluation import RegressionEvaluator

RegressionEvaluator(labelCol = 'consumption').evaluate(predictions)

## Examine intercept
regression.intercept

## Examine coefficients
regression.coefficients



# --------------------------------------------------------------------------------------------------------- #
##                  Flight duration model: Just distance                  ##
# Subset from the flights DataFrame:

# +------+--------+--------+
# |km    |features|duration|
# +------+--------+--------+
# |3465.0|[3465.0]|351     |
# |509.0 |[509.0] |82      |
# |542.0 |[542.0] |82      |
# |1989.0|[1989.0]|195     |
# |415.0 |[415.0] |65      |
# +------+--------+--------+
# only showing top 5 rows



from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol = 'duration').fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(flights_test)
predictions.select('duration', 'prediction').show(5, False)

# Calculate the RMSE
RegressionEvaluator(labelCol = 'duration').evaluate(predictions)


# output:
#     +--------+------------------+
#     |duration|prediction        |
#     +--------+------------------+
#     |105     |118.71205377865795|
#     |204     |174.69339409767792|
#     |160     |152.16523695718402|
#     |297     |337.8153345965721 |
#     |105     |113.5132482846978 |
#     +--------+------------------+
#     only showing top 5 rows
    




##                  Interpreting the coefficients                  ##
# Intercept (average minutes on ground)
inter = regression.intercept
print(inter)

# Coefficients
coefs = regression.coefficients
print(coefs)

# Average minutes per km
minutes_per_km = regression.coefficients[0]
print(minutes_per_km)

# Average speed in km per hour
avg_speed = 60 / minutes_per_km
print(avg_speed)

# output:
# intercept       : 44.36345473899361
# coefficients    : [0.07566671399881963]
# Average minutes : 0.07566671399881963
# Average speed   : 792.9510458315392





##                  Flight duration model: Adding origin airport                  ##
# Subset from the flights DataFrame:
# 
# +------+-------+-------------+----------------------+
# |km    |org_idx|org_dummy    |features              |
# +------+-------+-------------+----------------------+
# |3465.0|2.0    |(7,[2],[1.0])|(8,[0,3],[3465.0,1.0])|
# |509.0 |0.0    |(7,[0],[1.0])|(8,[0,1],[509.0,1.0]) |
# |542.0 |1.0    |(7,[1],[1.0])|(8,[0,2],[542.0,1.0]) |
# |1989.0|0.0    |(7,[0],[1.0])|(8,[0,1],[1989.0,1.0])|
# |415.0 |0.0    |(7,[0],[1.0])|(8,[0,1],[415.0,1.0]) |
# +------+-------+-------------+----------------------+
# only showing top 5 rows


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol = 'duration').fit(flights_train)


# Create predictions for the testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
RMSE = RegressionEvaluator(labelCol = 'duration').evaluate(predictions)
print(RMSE)

# output: 11.122486328678256






##                  Interpreting coefficients                  ##
# Average speed in km per hour
avg_speed_hour = 60 / regression.coefficients[0]
print(avg_speed_hour)

# Average minutes on ground at OGG
inter = regression.intercept
print(inter)

# Average minutes on ground at JFK
avg_ground_jfk = inter + regression.coefficients[3]
print(avg_ground_jfk)

# Average minutes on ground at LGA
avg_ground_lga = inter + regression.coefficients[4]
print(avg_ground_lga)

# output:
# avg_speed_hour         : 807.3336599681242
# intercept              : 15.856628374450773
# avg_ground_jfk : 68.53550999587868
# avg_ground_lga speed   : 62.56747182033072



## ====================================================================================================== ##