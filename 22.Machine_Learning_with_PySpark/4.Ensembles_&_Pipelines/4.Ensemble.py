# Create a forest trees
from pyspark.ml.classification import RandomForestClassifier

forest = RandomForestClassifier(numTrees = 5)
forest = forest.fit(cars_train)
forest.trees

forest.featureImportances

# Gradient-Boosted Trees
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(maxIter = 10)
gbt = gbt.fit(cars_train)





# --------------------------------------------------------------------------------------------------------- #
##                  Delayed flights with Gradient-Boosted Trees                  ##
# Subset of data from the flights DataFrame:
# 
# +---+------+--------+-----------------+-----+
# |mon|depart|duration|features         |label|
# +---+------+--------+-----------------+-----+
# |0  |16.33 |82      |[0.0,16.33,82.0] |1    |
# |2  |6.17  |82      |[2.0,6.17,82.0]  |0    |
# |9  |10.33 |195     |[9.0,10.33,195.0]|0    |
# |5  |7.98  |102     |[5.0,7.98,102.0] |0    |
# |7  |10.83 |135     |[7.0,10.83,135.0]|1    |
# +---+------+--------+-----------------+-----+
# only showing top 5 rows

# Import the classes required
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create model objects and train on training data
tree = DecisionTreeClassifier().fit(flights_train)
gbt = GBTClassifier().fit(flights_train)

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(tree.transform(flights_test))
evaluator.evaluate(gbt.transform(flights_test))


# Find the number of trees and the relative importance of features
print(gbt.trees)
print(gbt.featureImportances)

# output:
#     [DecisionTreeRegressionModel (uid=dtr_ae69c3b42761) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_e71e82840b5b) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_8a8de53d49c4) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_c0a331ff241b) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_24702af52e39) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_bcfeeed508ce) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_87f4e0d534ea) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_ed9e8d52b32d) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_5142076564e1) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_d00b8b6221ec) of depth 5 with 61 nodes, DecisionTreeRegressionModel (uid=dtr_a355c03744c6) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_0647208e3795) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_4a073ba497b4) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_58b81f22d284) of depth 5 with 59 nodes, DecisionTreeRegressionModel (uid=dtr_8d4ff9349eab) of depth 5 with 61 nodes, DecisionTreeRegressionModel (uid=dtr_5c7742e51c9c) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_de883a799422) of depth 5 with 61 nodes, DecisionTreeRegressionModel (uid=dtr_9109882bbb2b) of depth 5 with 61 nodes, DecisionTreeRegressionModel (uid=dtr_9053239abdfd) of depth 5 with 63 nodes, DecisionTreeRegressionModel (uid=dtr_45c5efcd0acf) of depth 5 with 63 nodes]
#     (3,[0,1,2],[0.2985254893881606,0.3182156041456896,0.3832589064661497])






##                  Delayed flights with a Random Forest                  ##
# Subset of data from the flights DataFrame:
# 
# +---+------+--------+-----------------+-----+
# |mon|depart|duration|features         |label|
# +---+------+--------+-----------------+-----+
# |9  |10.33 |195     |[9.0,10.33,195.0]|0    |
# |1  |8.0   |232     |[1.0,8.0,232.0]  |0    |
# |11 |7.77  |60      |[11.0,7.77,60.0] |1    |
# |4  |13.25 |210     |[4.0,13.25,210.0]|0    |
# |3  |17.58 |265     |[3.0,17.58,265.0]|1    |
# +---+------+--------+-----------------+-----+
# only showing top 5 rows


# Create a random forest classifier
forest = RandomForestClassifier()

# Create a parameter grid
params = ParamGridBuilder() \
        .addGrid(forest.featureSubsetStrategy, ['all', 'onethird', 'sqrt', 'log2']) \
        .addGrid(forest.maxDepth, [2, 5, 10]) \
        .build()

# Create a binary classification evaluator
evaluator = BinaryClassificationEvaluator()

# Create a cross-validator
cv = CrossValidator(estimator = forest, estimatorParamMaps = params, 
                    evaluator = evaluator, numFolds = 5)





##                  Evaluating Random Forest                  ##
# Average AUC for each parameter combination in grid
print(cv.avgMetrics)

# Average AUC for the best model
print(max(cv.avgMetrics))

# What's the optimal parameter value for maxDepth?
print(cv.bestModel.explainParam('maxDepth'))
# What's the optimal parameter value for featureSubsetStrategy?
print(cv.bestModel.explainParam('featureSubsetStrategy'))

# AUC for best model on testing data
print(evaluator.evaluate(cv.transform(flights_test)))

# output:
#     [0.61550451929848, 0.661275302749083, 0.6832959983649716, 0.6790399103856084, 0.6404890400309002, 0.6659871420567183, 0.6808977119243277, 0.6867946590518151, 0.6414270561540629, 0.6653385916148042, 0.6832494433718275, 0.6851695159338953, 0.6414270561540629, 0.6653385916148042, 0.6832494433718275, 0.6851695159338953]
#     0.6867946590518151
#     maxDepth: Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (current: 20)
#     featureSubsetStrategy: The number of features to consider for splits at each tree node. Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n]. (current: onethird)
#     0.6966021421117832





## ====================================================================================================== ##