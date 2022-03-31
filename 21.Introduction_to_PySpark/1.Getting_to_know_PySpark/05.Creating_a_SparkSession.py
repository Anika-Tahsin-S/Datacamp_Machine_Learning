# Verify SparkContext
print(sc.SparkContext)

# Print Spark version
print(sc.version)
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 3.2.0
      /_/

Using Python version 3.9.7 (default, Sep 10 2021 00:03:59)
Spark context Web UI available at http://e0646408-03ba-4911-a5d3-7946c3e4ff40.sessions.sessions.svc.cluster.local:4040
Spark context available as 'sc' (master = local[*], app id = local-1648756598233).
SparkSession available as 'spark'.


## ------------------------------------------------------------------- ##
# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)

# output: <pyspark.sql.session.SparkSession object at 0x7ff49fceecd0>
