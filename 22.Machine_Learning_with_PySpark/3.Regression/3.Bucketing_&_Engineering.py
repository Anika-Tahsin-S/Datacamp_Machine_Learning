## RPM buckets
from pyspark.ml.feature import Bucketizer

bucketizer = Bucketizer(splits = [3500, 4500, 6000, 6500], 
                    inputCol = 'rpm', outputCol = 'rpm_bin')

cars = bucketizer.transform(cars)


bucketed.select('rpm', 'rpm_bin').show(5)
cars.groupBy('rpm_bin').coount().show()


# Create a one-hot encoder
onehot = OneHotEncoder(inputCols=['depart_bucket'], outputCols=['depart_dummy'])

# One-hot encode the bucketed departure times
flights_onehot = onehot.fit(bucketed).transform(bucketed)
flights_onehot.select('depart', 'depart_bucket', 'depart_dummy').show(5)

## Engineering Density
cars = cars.withColumn('density_line', cars.mass / cars.length) # Linear density
cars = cars.withColumn('density_quad', cars.mass / cars.length ** 2) # Area density
cars = cars.withColumn('density_cube', cars.mass / cars.length ** 3) # Volume density





# --------------------------------------------------------------------------------------------------------- #
##                  Bucketing departure time                  ##
from pyspark.ml.feature import Bucketizer, OneHotEncoderEstimator

# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(splits = [3 * x for x in range(9)], inputCol = 'depart', outputCol = 'depart_bucket')
# Bucket the departure times
bucketed = buckets.transform(flights)
bucketed.select('depart', 'depart_bucket').show(5)

# Create a one-hot encoder
onehot = OneHotEncoderEstimator(inputCols = ['depart_bucket'], outputCols = ['depart_dummy'])


# One-hot encode the bucketed departure times
flights_onehot = onehot.fit(bucketed).transform(bucketed)
flights_onehot.select('depart', 'depart_bucket', 'depart_dummy').show(5)

# output:
#     +------+-------------+
#     |depart|depart_bucket|
#     +------+-------------+
#     |  9.48|          3.0|
#     | 16.33|          5.0|
#     |  6.17|          2.0|
#     | 10.33|          3.0|
#     |  8.92|          2.0|
#     +------+-------------+
#     only showing top 5 rows
#     
#     +------+-------------+-------------+
#     |depart|depart_bucket| depart_dummy|
#     +------+-------------+-------------+
#     |  9.48|          3.0|(7,[3],[1.0])|
#     | 16.33|          5.0|(7,[5],[1.0])|
#     |  6.17|          2.0|(7,[2],[1.0])|
#     | 10.33|          3.0|(7,[3],[1.0])|
#     |  8.92|          2.0|(7,[2],[1.0])|
#     +------+-------------+-------------+
#     only showing top 5 rows
    






##                  Flight duration model: Adding departure time                  ##
# Feature columns:
# 
#  0 — km
#  1 — ORD
#  2 — SFO
#  3 — JFK
#  4 — LGA
#  5 — SJC
#  6 — SMF
#  7 — TUS
#  8 — 00:00 to 03:00
#  9 — 03:00 to 06:00
# 10 — 06:00 to 09:00
# 11 — 09:00 to 12:00
# 12 — 12:00 to 15:00
# 13 — 15:00 to 18:00
# 14 — 18:00 to 21:00


# Find the RMSE on testing data
from pyspark.ml.evaluation import RegressionEvaluator

RegressionEvaluator(labelCol = 'duration', metricName = 'rmse').evaluate(predictions)

# Average minutes on ground at OGG for flights departing between 21:00 and 24:00
avg_eve_ogg = regression.intercept
print(avg_eve_ogg)

# Average minutes on ground at OGG for flights departing between 00:00 and 03:00
avg_night_ogg = regression.intercept + regression.coefficients[8]
print(avg_night_ogg)

# Average minutes on ground at JFK for flights departing between 00:00 and 03:00
avg_night_jfk = regression.intercept + regression.coefficients[3] + regression.coefficients[8]
print(avg_night_jfk)

# output:
# avg_eve_ogg   : 10.475615792093903
# avg_night_ogg : -4.125122945654926
# avg_night_jfk : 47.580713975630594

    
    




## ====================================================================================================== ##