# --------------------------------------------------------------------------------------------------------- #
##                   Load data using pandas                  ##
# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing.price)

# output:
#     0         221900.0
#     1         538000.0
#     2         180000.0
#     3         604000.0
#     4         510000.0
#     5        1225000.0
#     6         257500.0
#     7         291850.0
#     8         229500.0
#     9         323000.0
#     10        662500.0
#     11        468000.0
#     12        310000.0
#     13        400000.0
#     14        530000.0
#     15        650000.0
#     16        395000.0
#     17        485000.0
#     18        189000.0
#     19        230000.0
#     20        385000.0
#     21       2000000.0
#     22        285000.0
#     23        252700.0
#     24        329000.0
#     25        233000.0
#     26        937000.0
#     27        667000.0
#     28        438000.0
#     29        719000.0
#             ...    
#     21583     399950.0
#     21584     380000.0
#     21585     270000.0
#     21586     505000.0
#     21587     385000.0
#     21588     414500.0
#     21589     347500.0
#     21590    1222500.0
#     21591     572000.0
#     21592     475000.0
#     21593    1088000.0
#     21594     350000.0
#     21595     520000.0
#     21596     679950.0
#     21597    1575000.0
#     21598     541800.0
#     21599     810000.0
#     21600    1537000.0
#     21601     467000.0
#     21602     224000.0
#     21603     507250.0
#     21604     429000.0
#     21605     610685.0
#     21606    1007500.0
#     21607     475000.0
#     21608     360000.0
#     21609     400000.0
#     21610     402101.0
#     21611     400000.0
#     21612     325000.0
#     Name: price, Length: 21613, dtype: float64






##                   Setting the Data Type                  ##
# Import numpy and tensorflow with their standard aliases
import numpy as np
import tensorflow as tf

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)

# output:
#     [221900. 538000. 180000. ... 402101. 400000. 325000.]
#     tf.Tensor([False False False ... False False False], shape=(21613,), dtype=bool)