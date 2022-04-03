##                  Train/test split                  ##
# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights.randomSplit([0.8, 0.2], seed = 17)

# Check that training set has around 80% of records
training_ratio = flights_train.count() / flights.count()
print(training_ratio)

# output: 0.7954448701575139





##                  Build a Decision Tree                  ##
# Import the Decision Tree Classifier class
from pyspark.ml.classification import DecisionTreeClassifier

# Create a classifier object and fit to the training data
tree = DecisionTreeClassifier()
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
prediction.select('label', 'prediction', 'probability').show(5, False)


# output:
#     +-----+----------+----------------------------------------+
#     |label|prediction|probability                             |
#     +-----+----------+----------------------------------------+
#     |1    |1.0       |[0.2911010558069382,0.7088989441930619] |
#     |1    |1.0       |[0.3875,0.6125]                         |
#     |1    |1.0       |[0.3875,0.6125]                         |
#     |0    |0.0       |[0.6337448559670782,0.3662551440329218] |
#     |0    |0.0       |[0.9368421052631579,0.06315789473684211]|
#     +-----+----------+----------------------------------------+
#     only showing top 5 rows






##                  Evaluate the Decision Tree                  ##
# Sample of predictions:
# 
# +-----+----------+----------------------------------------+
# |label|prediction|probability                             |
# +-----+----------+----------------------------------------+
# |1    |1.0       |[0.2911010558069382,0.7088989441930619] |
# |1    |1.0       |[0.3875,0.6125]                         |
# |1    |1.0       |[0.3875,0.6125]                         |
# |0    |0.0       |[0.6337448559670782,0.3662551440329218] |
# |0    |0.0       |[0.9368421052631579,0.06315789473684211]|
# +-----+----------+----------------------------------------+
# only showing top 5 rows


# Create a confusion matrix
prediction.groupBy('label', 'prediction').count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label = 1').count()
FP = prediction.filter('prediction = 1 AND label = 0').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN + TP) / (TN + TP + FN + FP)
print(accuracy)

# output:
#     +-----+----------+-----+
#     |label|prediction|count|
#     +-----+----------+-----+
#     |    1|       0.0|  154|
#     |    0|       0.0|  289|
#     |    1|       1.0|  328|
#     |    0|       1.0|  190|
#     +-----+----------+-----+
#     
#     0.6420395421436004




## ====================================================================================================== ##