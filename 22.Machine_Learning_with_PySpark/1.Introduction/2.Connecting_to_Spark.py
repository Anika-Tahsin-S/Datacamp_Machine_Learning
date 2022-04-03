# Importing pyspark
import pyspark
# Important to check version
pyspark.__version__

## Sub-modules
# Structured Data = pyspark.sql
# Streaming Data = pyspark.streaming
# Machine Learning = pyspark.mllib (deprecated) and pyspark.ml

## Spark URL
# Remote Cluster using Spark URL - spark://<IP address | DNS name>:<port>
# Example, spark://13.59.151.161:7077

# Local Cluster Examples:
# local - only 1 core
# local[4] - 4 cores
# local[*] - all available cores


## Creating a SparkSession
from pyspark.sql import SparkSession

# Creating local cluster using a SparkSession builder
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

# Interact with Spark

# Close connection to Spark
>>> spark.stop()




# --------------------------------------------------------------------------------------------------------- #
##                  Location of Spark master                  ##
# Which of the following is not a valid way to specify the location of a Spark cluster?

# A) spark://13.59.151.161:7077
# B) spark://ec2-18-188-22-23.us-east-2.compute.amazonaws.com:7077
# C) spark://18.188.22.23
# D) local
# E) local[4]
# F) local[*]

# Answer: C; A Spark URL must always include a port number, so this URL is not valid.






##                  Creating a SparkSession                  ##
# Import the SparkSession class
from pyspark.sql import SparkSession

# Create SparkSession object
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('test') \
                    .getOrCreate()

# What version of Spark?
print(spark.version)

# Terminate the cluster
spark.stop()

# output: 2.4.2



## ====================================================================================================== ##