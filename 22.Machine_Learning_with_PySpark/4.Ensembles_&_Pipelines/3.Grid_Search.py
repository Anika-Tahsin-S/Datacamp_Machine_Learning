cars.select('mass', 'cyl', 'consumption').show(5)

regression = LinearRegression(labelCol = 'consumption', fitIntercept = True)
regression = regression.fit(cars_train)

# Parameter Grid
from pyspark.ml.tuning import ParamGridBuilder

params = ParamGridBuilder()
params = params.addGrid(regression.fitIntercept, [True, False])

# Build the parameter grid
params = params.build()
print('Number of models to be tested: ', len(params))



cv = CrossValidator(estimator = regression, estimatorParamMaps = params, 
                    evaluator = evaluator)

cv = cv.setNumFolds(10).setSeed(13).fit(cars_train)
cv.avgMetrics
cv.bestModel

predictions = cv.transform(cars_test)
cv.bestModel.explainParam('fitIntercept')

params = ParamGridBuilder().addGrid(regression.fitIntercept, [True, False])\
                           .addGrid(regression.regParam, [0.01, 0.1, 1.0, 10.0])\
                           .addGrid(regression.elasticNetParam, [0.0, 0.5, 1.0])\
                           .build()

print('Number of models to be tested: ', len(params))






# --------------------------------------------------------------------------------------------------------- #
##                  Optimizing flights linear regression                  ##
# Create parameter grid
params = ParamGridBuilder()

# Add grids for two parameters
params = params.addGrid(regression.regParam, [0.01, 0.1, 1.0, 10.0])\
               .addGrid(regression.elasticNetParam, [0.0, 0.5, 1.0])


# Build the parameter grid
params = params.build()
print('Number of models to be tested: ', len(params))

# Create cross-validator
cv = CrossValidator(estimator = pipeline, estimatorParamMaps = params, evaluator = evaluator, numFolds = 5)

# output: Number of models to be tested:  12




##                  Dissecting the best flight duration model                  ##
# The default model has RMSE of 10.614372 on testing data.
# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print(best_model.stages)

# Get the parameters for the LinearRegression object in the best model
best_model.stages[3].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
predictions = best_model.transform(flights_test)
print("RMSE =", evaluator.evaluate(predictions))

# output:
#     [StringIndexer_14299b2d5472, OneHotEncoderEstimator_9a650c117f1d, VectorAssembler_933acae88a6e, LinearRegression_9f5a93965597]
#     RMSE = 10.516377654959923





##                  SMS spam optimised                  ##
# Create parameter grid
params = ParamGridBuilder()

# Add grid for hashing trick parameters
params = params.addGrid(hasher.numFeatures, (1024, 4096, 16384))\
               .addGrid(hasher.binary, (True, False))

# Add grid for logistic regression parameters
params = params.addGrid(logistic.regParam, (0.01, 0.1, 1.0, 10.0))\
               .addGrid(logistic.elasticNetParam, (0.0, 0.5, 1.0))

# Build parameter grid
params = params.build()

print('Number of models to be tested: ', len(params))
# output: Number of models to be tested:  72






##                  How many models for grid search?                  ##
# How many models will be built when the cross-validator below is fit to data?

params = ParamGridBuilder().addGrid(hasher.numFeatures, [1024, 4096, 16384]) \
                           .addGrid(hasher.binary, [True, False]) \
                           .addGrid(logistic.regParam, [0.01, 0.1, 1.0, 10.0]) \
                           .addGrid(logistic.elasticNetParam, [0.0, 0.5, 1.0]) \
                           .build()

cv = CrossValidator(..., estimatorParamMaps=params, numFolds=5)


# Answer: 360
# There are 72 points in the parameter grid and 5 folds in the cross-validator. The product is 360. It takes time to build all of those models, which is why you're not doing it here!


## ====================================================================================================== ##