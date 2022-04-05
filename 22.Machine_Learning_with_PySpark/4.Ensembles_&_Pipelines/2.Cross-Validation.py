cars.select('mass', 'cyl', 'consumption').show(5)

regression = LinearRegression(labelCol = 'consumption')
evaluator = RegressionEvaluator(labelCol = 'consumption')

# Grid and cross-validator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

params = ParamGridBuilder().build()

cv = CrossValidator(estimator = regression, estimatorParamMaps = params, 
                    evaluator = evaluator, numFolds = 10, seed = 13)

cv = cv.fit(cars_train)
cv.avgMetrics
evaluator.evaluate(cv.transform(cars_train))





# --------------------------------------------------------------------------------------------------------- #
##                  Cross validating simple flight duration model                  ##
# Subset from the flights DataFrame:
# 
# +------+--------+--------+
# |km    |features|duration|
# +------+--------+--------+
# |542.0 |[542.0] |82      |
# |885.0 |[885.0] |102     |
# |2317.0|[2317.0]|232     |
# |2943.0|[2943.0]|250     |
# |1765.0|[1765.0]|190     |
# +------+--------+--------+
# only showing top 5 rows

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

flights_train, flights_test = flights.randomSplit([0.8, 0.2])



# Create an empty parameter grid
params = ParamGridBuilder().build()

# Create objects for building and evaluating a regression model
regression = LinearRegression(labelCol = 'duration')
evaluator = RegressionEvaluator(labelCol = 'duration')

# Create a cross validator
cv = CrossValidator(estimator = regression, estimatorParamMaps = params, evaluator = evaluator, numFolds = 5)


# Train and test model on multiple folds of the training data
cv = cv.fit(flights_train)

# NOTE: Since cross-valdiation builds multiple models, the fit() method can take a little while to complete.





##                  Cross validating flight duration model pipeline                  ##
# Create an indexer for the org field
indexer = StringIndexer(inputCol = 'org', outputCol = 'org_idx')

# Create an one-hot encoder for the indexed org field
onehot = OneHotEncoderEstimator(inputCols = ['org_idx'], outputCols = ['org_dummy'])

# Assemble the km and one-hot encoded fields
assembler = VectorAssembler(inputCols = ['km', 'org_dummy'], outputCol = 'features')

# Create a pipeline and cross-validator.
pipeline = Pipeline(stages = [indexer, onehot, assembler, regression])
cv = CrossValidator(estimator = pipeline, estimatorParamMaps = params, evaluator = evaluator)




## ====================================================================================================== ##