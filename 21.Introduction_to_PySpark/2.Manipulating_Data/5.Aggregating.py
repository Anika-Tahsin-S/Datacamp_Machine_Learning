##                  Aggregating I                  ##
# All of the common aggregation methods, like .min(), .max(), and .count() are GroupedData methods. These are created by calling the .groupBy() DataFrame method. You'll learn exactly what that means in a few exercises. For now, all you have to do to use these functions is call that method on your DataFrame. For example, to find the minimum value of a column, col, in a DataFrame, df, you could do

# df.groupBy().min("col").show()

# This creates a GroupedData object (so you can use the .min() method), then finds the minimum value in col, and returns it as a DataFrame.

# Now you're ready to do some aggregating of your own!

# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == 'PDX').groupBy().min('distance').show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin == 'SEA').groupBy().max('air_time').show()

# output:
    +-------------+
    |min(distance)|
    +-------------+
    |          106|
    +-------------+
    
    +-------------+
    |max(air_time)|
    +-------------+
    |          409|
    +-------------+







##                  Aggregating II                  ##
# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()

# output:
    +------------------+
    |     avg(air_time)|
    +------------------+
    |188.20689655172413|
    +------------------+
    
    +------------------+
    | sum(duration_hrs)|
    +------------------+
    |25289.600000000126|
    +------------------+