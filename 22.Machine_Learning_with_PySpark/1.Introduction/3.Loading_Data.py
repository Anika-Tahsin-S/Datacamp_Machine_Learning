# Reading data from CSV
cars = spark.read.csv('cars.csv', header = True)

## Optional Arguments:
# header - is first row a header? (default: False)
# sep - field separator (default: a common ',')
# schema - explicit column data type
# inferSchema - deduce column data types from data?
# nullValue - placeholder for missing data


## Inferring column types from data
cars = spark.read.csv('cars.csv', header = True, inferSchema = True)
cars.dtypes

## Dealing with missing data
cars = spark.read.csv('cars.csv', header = True, inferSchema = True, nullValue = 'NA')



# --------------------------------------------------------------------------------------------------------- #
##                  Loading flights data                  ##
# Data dictionary:
# 
#     mon — month (integer between 1 and 12)
#     dom — day of month (integer between 1 and 31)
#     dow — day of week (integer; 1 = Monday and 7 = Sunday)
#     org — origin airport (IATA code)
#     mile — distance (miles)
#     carrier — carrier (IATA code)
#     depart — departure time (decimal hour)
#     duration — expected duration (minutes)
#     delay — delay (minutes)


# First few records from 'flights.csv':

# mon,dom,dow,carrier,flight,org,mile,depart,duration,delay
# 11,20,6,US,19,JFK,2153,9.48,351,NA
# 0,22,2,UA,1107,ORD,316,16.33,82,30
# 2,20,4,UA,226,SFO,337,6.17,82,-8
# 9,13,1,AA,419,ORD,1236,10.33,195,-5
# 4,2,5,AA,325,ORD,258,8.92,65,NA


# Read data from CSV file
flights = spark.read.csv('flights.csv', sep = ',', header = True, inferSchema = True, nullValue = 'NA')

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
print(flights.dtypes)

# output:
#     The data contain 50000 records.
#     +---+---+---+-------+------+---+----+------+--------+-----+
#     |mon|dom|dow|carrier|flight|org|mile|depart|duration|delay|
#     +---+---+---+-------+------+---+----+------+--------+-----+
#     | 11| 20|  6|     US|    19|JFK|2153|  9.48|     351| null|
#     |  0| 22|  2|     UA|  1107|ORD| 316| 16.33|      82|   30|
#     |  2| 20|  4|     UA|   226|SFO| 337|  6.17|      82|   -8|
#     |  9| 13|  1|     AA|   419|ORD|1236| 10.33|     195|   -5|
#     |  4|  2|  5|     AA|   325|ORD| 258|  8.92|      65| null|
#     +---+---+---+-------+------+---+----+------+--------+-----+
#     only showing top 5 rows
#     
#     [('mon', 'int'), ('dom', 'int'), ('dow', 'int'), ('carrier', 'string'), ('flight', 'int'), ('org', 'string'), ('mile', 'int'), ('depart', 'double'), ('duration', 'int'), ('delay', 'int')]








##                  Loading SMS spam data                  ##
# First few records from 'sms.csv':
# 
# 1;Sorry, I'll call later in meeting;0
# 2;Dont worry. I guess he's busy.;0
# 3;Call FREEPHONE 0800 542 0578 now!;1
# 4;Win a 1000 cash prize or a prize worth 5000;1



from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
sms = spark.read.csv("sms.csv", sep = ';', header = False, schema = schema)

# Print schema of DataFrame
sms.printSchema()


# output:
#     root
#      |-- id: integer (nullable = true)
#      |-- text: string (nullable = true)
#      |-- label: integer (nullable = true)





## ====================================================================================================== ##