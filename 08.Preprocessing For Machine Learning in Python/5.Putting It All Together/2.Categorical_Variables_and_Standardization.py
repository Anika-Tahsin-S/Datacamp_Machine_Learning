import pandas as pd
import re

ufo = ufo_no_missing.copy()
ufo.head()

##                   Extracting Numbers From Strings                  ##
def return_minutes(time_string):

    # Use \d+ to grab digits
    pattern = re.compile(r"\d+")
    
    # Use match on the pattern and column
    num = re.match(pattern, time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(lambda row: return_minutes(row))

# Take a look at the head of both of the columns
print(ufo[["length_of_time", "minutes"]].head())





##                   Identifying Features For Standardization                  ##
import numpy as np

# Check the variance of the seconds and minutes columns
print(ufo[["seconds", "minutes"]].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo["seconds"])

# Print out the variance of just the seconds_log column
print(ufo["seconds_log"].var())
# output:
#    seconds    424087.417474
#    minutes       117.546372
#
#    dtype: float64
#    1.1223923881183004