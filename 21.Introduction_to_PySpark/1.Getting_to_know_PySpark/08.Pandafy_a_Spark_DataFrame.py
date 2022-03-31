import pyspark
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())


# output:
      origin dest    N
    0    SEA  RNO    8
    1    SEA  DTW   98
    2    SEA  CLE    2
    3    SEA  LAX  450
    4    PDX  SEA  144