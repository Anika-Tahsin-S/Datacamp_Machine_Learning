##                  Carrier                  ##
# In this exercise you'll create a StringIndexer and a OneHotEncoder to code the carrier column. To do this, you'll call the class constructors with the arguments inputCol and outputCol.

# The inputCol is the name of the column you want to index or encode, and the outputCol is the name of the new column that the Transformer should create.
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol = "carrier", outputCol = "carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol = "carrier_index", outputCol = "carrier_fact")



##                  Destination                  ##
# Create a StringIndexer
dest_indexer = StringIndexer(inputCol = 'dest', outputCol = 'dest_index')

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol = 'dest_index', outputCol = 'dest_fact')



##                  Assemble a vector                  ##
from pyspark.ml.feature import  VectorAssembler
# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols = ["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol = "features")
