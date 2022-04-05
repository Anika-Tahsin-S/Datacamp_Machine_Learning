## Cars model: Steps
indexer = StringIndexer(inputCol = 'type', outputCol = 'type_idx')

onehot = OneHotEncoderEstimator(inputCols = ['type_idx'],outputCols = ['type_dummy'])

assembler = VectorAssembler(inputCols = ['mass', 'cyl', 'type_dummy'], outputCol='features')

regression = LinearRegression(labelCol = 'consumptino')



## Training data
indexer = indexer.fit(cars_train)
cars_train = indexer.transform(cars_train)

onehot = onehot.fit(cars_train)
cars_train = onehot.transform(cars_train)
cars_train = assemble.transform(cars_train)
regression = regression.fit(cars_train)



## Testing data
cars_test = indexer.transform(cars_test)
cars_test = onehot.transform(cars_test)
cars_test = assemble.transform(cars_test)
predictions = regression.transform(cars_test)



## Cars model: Pipeline
from pyspark.ml import Pipeline

pipeline = Pipeline(stages = [indexer, onehot, assembler, regression])

pipeline = pipeline.fit(cars_train)
predictions = pipeline.transform(cars_test)


## Cars model: Stages
pipeline.stages[3]

print(pipeline.stages[3].intercept)
print(pipeline.stages[3].coefficients)








# --------------------------------------------------------------------------------------------------------- #
##                  Flight duration model: Pipeline stages                  ##
# The first few rows of the flights DataFrame:
# 
# +---+---+---+-------+------+---+------+--------+-----+------+
# |mon|dom|dow|carrier|flight|org|depart|duration|delay|km    |
# +---+---+---+-------+------+---+------+--------+-----+------+
# |11 |20 |6  |US     |19    |JFK|9.48  |351     |null |3465.0|
# |0  |22 |2  |UA     |1107  |ORD|16.33 |82      |30   |509.0 |
# |2  |20 |4  |UA     |226   |SFO|6.17  |82      |-8   |542.0 |
# |9  |13 |1  |AA     |419   |ORD|10.33 |195     |-5   |1989.0|
# |4  |2  |5  |AA     |325   |ORD|8.92  |65      |null |415.0 |
# +---+---+---+-------+------+---+------+--------+-----+------+
# only showing top 5 rows


from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression

# Convert categorical strings to index values
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# One-hot encode index values
onehot = OneHotEncoderEstimator(
    inputCols=['org_idx', 'dow'],
    outputCols=['org_dummy', 'dow_dummy']
)

# Assemble predictors into a single column
assembler = VectorAssembler(inputCols=['km', 'org_dummy', 'dow_dummy'], outputCol='features')

# A linear regression object
regression = LinearRegression(labelCol='duration')





##                  Flight duration model: Pipeline model                  ##
flights_train, flights_test = flights.randomSplit([0.8, 0.2])

# Import class for creating a pipeline
from pyspark.ml import Pipeline

# Construct a pipeline
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])


# Train the pipeline on the training data
pipeline = pipeline.fit(flights_train)

# Make predictions on the testing data
predictions = pipeline.transform(flights_test)




##                  SMS spam pipeline                  ##
# Selected columns from first few rows of the sms DataFrame:
# 
# +---+---------------------------------+-----+
# |id |text                             |label|
# +---+---------------------------------+-----+
# |1  |Sorry I'll call later in meeting |0    |
# |2  |Dont worry I guess he's busy     |0    |
# |3  |Call FREEPHONE now               |1    |
# |4  |Win a cash prize or a prize worth|1    |
# +---+---------------------------------+-----+
# only showing top 4 rows



from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# Break text into tokens at non-word characters
tokenizer = Tokenizer(inputCol= 'text', outputCol = 'words')

# Remove stop words
remover = StopWordsRemover(inputCol = tokenizer.getOutputCol(), outputCol = 'terms')

# Apply the hashing trick and transform to TF-IDF
hasher = HashingTF(inputCol = remover.getOutputCol(), outputCol = 'hash')
idf = IDF(inputCol = hasher.getOutputCol(), outputCol = 'features')

# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()
pipeline = Pipeline(stages=[tokenizer, remover, hasher, idf, logistic])




## ====================================================================================================== ##