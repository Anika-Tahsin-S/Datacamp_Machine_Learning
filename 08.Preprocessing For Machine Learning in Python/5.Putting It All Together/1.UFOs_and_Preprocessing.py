import pandas as pd

ufo = pd.read_csv('ufo_sightings_large.csv')
ufo.head()
ufo.shape
ufo.info()



##                   Checking Column Types                  ##
# Check the column types
print(ufo.dtypes)

# Change the type of seconds to float
ufo["seconds"] = ufo["seconds"].astype(float)

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo["date"])

# Check the column types
print(ufo[["seconds", "date"]].dtypes)




##                   Dropping Missing Data                  ##
# Check how many values are missing in the length_of_time, state, and type columns
print(ufo[["length_of_time", "state", "type"]].isnull().sum())

# Keep only rows where length_of_time, state, and type are not null
ufo_no_missing = ufo[ufo["length_of_time"].notnull() & 
          ufo["state"].notnull() & 
          ufo["type"].notnull()]

# Print out the shape of the new dataset
print(ufo_no_missing.shape)
# output:
#    length_of_time    143
#    state             419
#    type              159
#
#    dtype: int64
#    (4283, 4)