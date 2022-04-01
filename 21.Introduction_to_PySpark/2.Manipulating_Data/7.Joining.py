##                  Joining I                  ##
# A join will combine two different tables along a column that they share. This column is called the key. 
# Examples of keys here include the tailnum and carrier columns from the flights table.

# For example, suppose that you want to know more information about the plane that flew a flight than just the tail number. 
# This information isn't in the flights table because the same plane flies many different flights over the course of two years, so including this information in every row would result in a lot of duplication. 
# To avoid this, you'd have a second table that has only one row for each plane and whose columns list all the information about the plane, including its tail number. 
# You could call this table planes

# When you join the flights table to this table of airplane information, you're adding all the columns from the planes table to the flights table. 
# To fill these columns with information, you'll look at the tail number from the flights table and find the matching one in the planes table, and then use that row to fill out all the new columns.

# Now you'll have a much bigger table than before, but now every row has all information about the plane that flew that flight!

## Which of the following is not true?
# Answer: There is only one kind of join.
# If there were only one kind of join, it would be tough to create some more complicated kinds of tables.



##                  Joining II                  ##
# In PySpark, joins are performed using the DataFrame method .join(). 
# This method takes three arguments. The first is the second DataFrame that you want to join with the first one. 
# The second argument, on, is the name of the key column(s) as a string. The names of the key column(s) must be the same in each table. 
# The third argument, how, specifies the kind of join to perform. In this course we'll always use the value how="leftouter".


# Examine the data
print(airports.show())

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on='dest', how='leftouter')

# Examine the new DataFrame
print(flights_with_airports.show())

# output:
    +---+--------------------+----------------+-----------------+----+---+---+
    |faa|                name|             lat|              lon| alt| tz|dst|
    +---+--------------------+----------------+-----------------+----+---+---+
    |04G|   Lansdowne Airport|      41.1304722|      -80.6195833|1044| -5|  A|
    |06A|Moton Field Munic...|      32.4605722|      -85.6800278| 264| -5|  A|
    |06C| Schaumburg Regional|      41.9893408|      -88.1012428| 801| -6|  A|
    |06N|     Randall Airport|       41.431912|      -74.3915611| 523| -5|  A|
    |09J|Jekyll Island Air...|      31.0744722|      -81.4277778|  11| -4|  A|
    |0A9|Elizabethton Muni...|      36.3712222|      -82.1734167|1593| -4|  A|
    |0G6|Williams County A...|      41.4673056|      -84.5067778| 730| -5|  A|
    |0G7|Finger Lakes Regi...|      42.8835647|      -76.7812318| 492| -5|  A|
    |0P2|Shoestring Aviati...|      39.7948244|      -76.6471914|1000| -5|  U|
    |0S9|Jefferson County ...|      48.0538086|     -122.8106436| 108| -8|  A|
    |0W3|Harford County Ai...|      39.5668378|      -76.2024028| 409| -5|  A|
    |10C|  Galt Field Airport|      42.4028889|      -88.3751111| 875| -6|  U|
    |17G|Port Bucyrus-Craw...|      40.7815556|      -82.9748056|1003| -5|  A|
    |19A|Jackson County Ai...|      34.1758638|      -83.5615972| 951| -4|  U|
    |1A3|Martin Campbell F...|      35.0158056|      -84.3468333|1789| -4|  A|
    |1B9| Mansfield Municipal|      42.0001331|      -71.1967714| 122| -5|  A|
    |1C9|Frazier Lake Airpark|54.0133333333333|-124.768333333333| 152| -8|  A|
    |1CS|Clow Internationa...|      41.6959744|      -88.1292306| 670| -6|  U|
    |1G3|  Kent State Airport|      41.1513889|      -81.4151111|1134| -4|  A|
    |1OH|     Fortman Airport|      40.5553253|      -84.3866186| 885| -5|  U|
    +---+--------------------+----------------+-----------------+----+---+---+
    only showing top 20 rows
    
    None
    +----+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+--------+--------+----+------+--------------------+---------+-----------+----+---+---+
    |dest|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|tailnum|flight|origin|air_time|distance|hour|minute|                name|      lat|        lon| alt| tz|dst|
    +----+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+--------+--------+----+------+--------------------+---------+-----------+----+---+---+
    | LAX|2014|   12|  8|     658|       -7|     935|       -5|     VX| N846VA|  1780|   SEA|     132|     954|   6|    58|    Los Angeles Intl|33.942536|-118.408075| 126| -8|  A|
    | HNL|2014|    1| 22|    1040|        5|    1505|        5|     AS| N559AS|   851|   SEA|     360|    2677|  10|    40|       Honolulu Intl|21.318681|-157.922428|  13|-10|  N|
    | SFO|2014|    3|  9|    1443|       -2|    1652|        2|     VX| N847VA|   755|   SEA|     111|     679|  14|    43|  San Francisco Intl|37.618972|-122.374889|  13| -8|  A|
    | SJC|2014|    4|  9|    1705|       45|    1839|       34|     WN| N360SW|   344|   PDX|      83|     569|  17|     5|Norman Y Mineta S...|  37.3626|-121.929022|  62| -8|  A|
    | BUR|2014|    3|  9|     754|       -1|    1015|        1|     AS| N612AS|   522|   SEA|     127|     937|   7|    54|            Bob Hope|34.200667|-118.358667| 778| -8|  A|
    | DEN|2014|    1| 15|    1037|        7|    1352|        2|     WN| N646SW|    48|   PDX|     121|     991|  10|    37|         Denver Intl|39.861656|-104.673178|5431| -7|  A|
    | OAK|2014|    7|  2|     847|       42|    1041|       51|     WN| N422WN|  1520|   PDX|      90|     543|   8|    47|Metropolitan Oakl...|37.721278|-122.220722|   9| -8|  A|
    | SFO|2014|    5| 12|    1655|       -5|    1842|      -18|     VX| N361VA|   755|   SEA|      98|     679|  16|    55|  San Francisco Intl|37.618972|-122.374889|  13| -8|  A|
    | SAN|2014|    4| 19|    1236|       -4|    1508|       -7|     AS| N309AS|   490|   SEA|     135|    1050|  12|    36|      San Diego Intl|32.733556|-117.189667|  17| -8|  A|
    | ORD|2014|   11| 19|    1812|       -3|    2352|       -4|     AS| N564AS|    26|   SEA|     198|    1721|  18|    12|  Chicago Ohare Intl|41.978603| -87.904842| 668| -6|  A|
    | LAX|2014|   11|  8|    1653|       -2|    1924|       -1|     AS| N323AS|   448|   SEA|     130|     954|  16|    53|    Los Angeles Intl|33.942536|-118.408075| 126| -8|  A|
    | PHX|2014|    8|  3|    1120|        0|    1415|        2|     AS| N305AS|   656|   SEA|     154|    1107|  11|    20|Phoenix Sky Harbo...|33.434278|-112.011583|1135| -7|  N|
    | LAS|2014|   10| 30|     811|       21|    1038|       29|     AS| N433AS|   608|   SEA|     127|     867|   8|    11|      Mc Carran Intl|36.080056| -115.15225|2141| -8|  A|
    | ANC|2014|   11| 12|    2346|       -4|     217|      -28|     AS| N765AS|   121|   SEA|     183|    1448|  23|    46|Ted Stevens Ancho...|61.174361|-149.996361| 152| -9|  A|
    | SFO|2014|   10| 31|    1314|       89|    1544|      111|     AS| N713AS|   306|   SEA|     129|     679|  13|    14|  San Francisco Intl|37.618972|-122.374889|  13| -8|  A|
    | SFO|2014|    1| 29|    2009|        3|    2159|        9|     UA| N27205|  1458|   PDX|      90|     550|  20|     9|  San Francisco Intl|37.618972|-122.374889|  13| -8|  A|
    | SMF|2014|   12| 17|    2015|       50|    2150|       41|     AS| N626AS|   368|   SEA|      76|     605|  20|    15|     Sacramento Intl|38.695417|-121.590778|  27| -8|  A|
    | MDW|2014|    8| 11|    1017|       -3|    1613|       -7|     WN| N8634A|   827|   SEA|     216|    1733|  10|    17| Chicago Midway Intl|41.785972| -87.752417| 620| -6|  A|
    | BOS|2014|    1| 13|    2156|       -9|     607|      -15|     AS| N597AS|    24|   SEA|     290|    2496|  21|    56|General Edward La...|42.364347| -71.005181|  19| -5|  A|
    | BUR|2014|    6|  5|    1733|      -12|    1945|      -10|     OO| N215AG|  3488|   PDX|     111|     817|  17|    33|            Bob Hope|34.200667|-118.358667| 778| -8|  A|
    +----+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+--------+--------+----+------+--------------------+---------+-----------+----+---+---+
    only showing top 20 rows
    
    None