## Precision and Recalls
# Precission (Positive)
# TP / (TP + FP)
# Recall (positive)
# TP / (TP + FN)

## Weighted metrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator()
evaluator.evaluate(prediction, {evaluator.metricName: "weightedPrecision"})




# --------------------------------------------------------------------------------------------------------- #
##                  Build a Logistic Regression model                  ##
# First few rows from the flights DataFrame:
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


# Selecting numeric columns
flights_train = flights_train.select("mon", 'depart', 'duration', 'features', 'label')
flights_test = flights_test.select("mon", "depart", "duration", 'features', 'label')



# Import the logistic regression class
from pyspark.ml.classification import LogisticRegression

# Create a classifier object and train on training data
logistic = LogisticRegression().fit(flights_train)

# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(flights_test)
prediction.groupBy("label", "prediction").count().show()

# output:
#     +-----+----------+-----+
#     |label|prediction|count|
#     +-----+----------+-----+
#     |    1|       0.0|  195|
#     |    0|       0.0|  288|
#     |    1|       1.0|  277|
#     |    0|       1.0|  201|
#     +-----+----------+-----+





##                  Evaluate the Logistic Regression model                  ##
# First few predictions from the Logistic Regression model:

# +-----+----------+----------------------------------------+
# |label|prediction|probability                             |
# +-----+----------+----------------------------------------+
# |0    |1.0       |[0.48618640716970973,0.5138135928302903]|
# |1    |0.0       |[0.52242444215606,0.47757555784394007]  |
# |0    |0.0       |[0.5726551829113304,0.4273448170886696] |
# |0    |0.0       |[0.5149292596494213,0.4850707403505788] |
# |1    |0.0       |[0.5426764281965827,0.4573235718034173] |
# +-----+----------+----------------------------------------+
# only showing top 5 rows


from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('precision = {:.2f}\nrecall    = {:.2f}'.format(precision, recall))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(prediction, {binary_evaluator.metricName: "areaUnderROC"})


# output:
#     precision = 0.58
#     recall    = 0.59




## ====================================================================================================== ##