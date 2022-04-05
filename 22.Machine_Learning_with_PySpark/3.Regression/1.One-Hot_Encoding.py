## One Hot Encoder
from pyspark.ml.feature import OneHotEncoderEstimator

# Create an instance of the one hot encoder
onehot = OneHotEncoderEstimator(inputCols = ['type_idx'], outputCols = ['type_dummy'])

onehot = onehot.fit(cars)
onehot.categorySizes


onehot = onehot.transform(cars)
onehot.select('type', 'type_idx', 'type_dummy').distinct().sort('type_idx').show()



## Dense versus sparse
from pyspark.mllib.linalg import DenseVector, SparseVector

DenseVector([1, 0, 0, 0, 0, 7, 0, 0])

SparseVector(8, [0, 5], [1, 7])







# --------------------------------------------------------------------------------------------------------- #
##                  Encoding flight origin                  ##
# Subset from the flights DataFrame:
# 
# +---+-------+
# |org|org_idx|
# +---+-------+
# |JFK|2.0    |
# |ORD|0.0    |
# |SFO|1.0    |
# |ORD|0.0    |
# |ORD|0.0    |
# +---+-------+
# only showing top 5 rows



# Import the one hot encoder class
from pyspark.ml.feature import OneHotEncoderEstimator

# Create an instance of the one hot encoder
onehot = OneHotEncoderEstimator(inputCols = ['org_idx'], outputCols = ['org_dummy'])

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flights)
flights_onehot = onehot.transform(flights)

# Check the results
flights_onehot.select('org', 'org_idx', 'org_dummy').distinct().sort('org_idx').show()


# output:
#     +---+-------+-------------+
#     |org|org_idx|    org_dummy|
#     +---+-------+-------------+
#     |ORD|    0.0|(7,[0],[1.0])|
#     |SFO|    1.0|(7,[1],[1.0])|
#     |JFK|    2.0|(7,[2],[1.0])|
#     |LGA|    3.0|(7,[3],[1.0])|
#     |SJC|    4.0|(7,[4],[1.0])|
#     |SMF|    5.0|(7,[5],[1.0])|
#     |TUS|    6.0|(7,[6],[1.0])|
#     |OGG|    7.0|    (7,[],[])|
#     +---+-------+-------------+
    







##                  Encoding shirt sizes                  ##
# You have data for a consignment of t-shirts. The data includes the size of the shirt, which is given as either S, M, L or XL.
# 
# Here are the counts for the different sizes:
# 
# +----+-----+
# |size|count|
# +----+-----+
# |   S|    8|
# |   M|   15|
# |   L|   20|
# |  XL|    7|
# +----+-----+

# The sizes are first converted to an index using StringIndexer and then one-hot encoded using OneHotEncoderEstimator.

# Q/A
# Which of the following is not true:

# S shirts get index 2.0 and are one-hot encoded as (3,[2],[1.0])
# M shirts get index 1.0 and are one-hot encoded as (3,[1],[1.0])
# L shirts get index 0.0 and are one-hot encoded as (3,[0],[1.0])
# XL shirts get index 3.0 and are one-hot encoded as (3,[3],[1.0])

# Answer: XL shirts get index 3.0 and are one-hot encoded as (3,[3],[1.0])
# This statement is false: XL is the least frequent size, so it receives an index of 3. However, it is one-hot encoded to (3,[],[]) because it does not get it's own dummy variable. If none of the other dummy variables are true, then this one must be true. So to make a separate dummy variable would be redundant!



## ====================================================================================================== ##