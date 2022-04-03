import pyspark
import numpy as np
import pandas as pd

## Dropping Columns
# Either drop the columns you don't want
cars = cars.drop('maker', 'model')

# or Select the columns you want to retain
cars = cars.select('origin', 'type', 'cyl', 'size', 'weight', 'length', 'rpm', 'consumption')

# Filtering out missing data
# How many values missing?
cars.filter('cyl IS NULL').count()

# Drop records with missing values in the cylinder column
cars = cars.filter('cyl IS NOT NULL')

# Drop records with missing values in any column
cars = cars.dropna()





##  Mutating columns
from pyspark.sql.functions import round

# Create a new 'mass' column
cars = cars.withColumn('mass', round(cars.weight / 2.205, 0))
# Covert length to metres
cars = cars.withColumn('length', round(cars.weight * 2.205, 3))

# Indexing categorical data
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol = 'type', outputCol = 'type_idx')
indexer = indexer.fit(cars)
cars = indexer.transform(cars)

# Indexing country of origin
cars = StringIndexer(inputCol = 'origin', outputCol= 'label').fit(cars).transform(cars)

# Assembling columns
from pyspark.ml.feature import VectorAssembler

# Create an assembler object
assembler = VectorAssembler(inputCols=['cyl', 'size'], outputCol = 'features')
assembler.transform(cars)






# --------------------------------------------------------------------------------------------------------- #
##                  Removing columns and rows                  ##
# Remove the 'flight' column
flights_drop_column = flights.drop('flight')

# Number of records with missing 'delay' values
flights_drop_column.filter('delay IS NULL').count()

# Remove records with missing 'delay' values
flights_valid_delay = flights_drop_column.filter('delay IS NOT NULL')

# Remove records with missing values in any column and get the number of remaining rows
flights_none_missing = flights_valid_delay.dropna()
print(flights_none_missing.count())

# output: 47022





##                  Column manipulation                  ##
# Import the required function
from pyspark.sql.functions import round

# Convert 'mile' to 'km' and drop 'mile' column (1 mile is equivalent to 1.60934 km)
flights_km = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                    .drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn('label', (flights_km.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)


# output:
#     +---+---+---+-------+---+------+--------+-----+------+-----+
#     |mon|dom|dow|carrier|org|depart|duration|delay|    km|label|
#     +---+---+---+-------+---+------+--------+-----+------+-----+
#     |  0| 22|  2|     UA|ORD| 16.33|      82|   30| 509.0|    1|
#     |  2| 20|  4|     UA|SFO|  6.17|      82|   -8| 542.0|    0|
#     |  9| 13|  1|     AA|ORD| 10.33|     195|   -5|1989.0|    0|
#     |  5|  2|  1|     UA|SFO|  7.98|     102|    2| 885.0|    0|
#     |  7|  2|  6|     AA|ORD| 10.83|     135|   54|1180.0|    1|
#     +---+---+---+-------+---+------+--------+-----+------+-----+
#     only showing top 5 rows







##                  Categorical columns                  ##
from pyspark.ml.feature import StringIndexer

# Create an indexer
indexer = StringIndexer(inputCol = 'carrier', outputCol = 'carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the other categorical feature
flights_indexed = StringIndexer(inputCol = 'org', outputCol = 'org_idx').fit(flights_indexed).transform(flights_indexed)
flights_indexed.show(5)

# output:
#     +---+---+---+-------+---+------+--------+-----+------+-----+-----------+-------+
#     |mon|dom|dow|carrier|org|depart|duration|delay|    km|label|carrier_idx|org_idx|
#     +---+---+---+-------+---+------+--------+-----+------+-----+-----------+-------+
#     |  0| 22|  2|     UA|ORD| 16.33|      82|   30| 509.0|    1|        0.0|    0.0|
#     |  2| 20|  4|     UA|SFO|  6.17|      82|   -8| 542.0|    0|        0.0|    1.0|
#     |  9| 13|  1|     AA|ORD| 10.33|     195|   -5|1989.0|    0|        1.0|    0.0|
#     |  5|  2|  1|     UA|SFO|  7.98|     102|    2| 885.0|    0|        0.0|    1.0|
#     |  7|  2|  6|     AA|ORD| 10.83|     135|   54|1180.0|    1|        1.0|    0.0|
#     +---+---+---+-------+---+------+--------+-----+------+-----+-----------+-------+
#     only showing top 5 rows







##                  Assembling columns                  ##
# Import the necessary class
from pyspark.ml.feature import VectorAssembler

# Create an assembler object
assembler = VectorAssembler(inputCols = [
    'mon', 'dom', 'dow',
    'carrier_idx', 
    'org_idx',
    'km', 'depart', 'duration'
], outputCol = 'features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights)

# Check the resulting column
flights_assembled.select('features', 'delay').show(5, truncate = False)

# output:
#     +-----------------------------------------+-----+
#     |features                                 |delay|
#     +-----------------------------------------+-----+
#     |[0.0,22.0,2.0,0.0,0.0,509.0,16.33,82.0]  |30   |
#     |[2.0,20.0,4.0,0.0,1.0,542.0,6.17,82.0]   |-8   |
#     |[9.0,13.0,1.0,1.0,0.0,1989.0,10.33,195.0]|-5   |
#     |[5.0,2.0,1.0,0.0,1.0,885.0,7.98,102.0]   |2    |
#     |[7.0,2.0,6.0,1.0,0.0,1180.0,10.83,135.0] |54   |
#     +-----------------------------------------+-----+
#     only showing top 5 rows


## ====================================================================================================== ##