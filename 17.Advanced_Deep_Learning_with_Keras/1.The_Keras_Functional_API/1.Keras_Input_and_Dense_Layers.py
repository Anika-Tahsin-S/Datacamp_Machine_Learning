# Course Datasets
import pandas as pd
games_season = pd.read_csv('games_season.csv')
games_tourney = pd.read_csv('games_tourney.csv')

# Inputs
from keras.layers import Input
input_tensor = Input(Shape = (1,))
# The shape argument expects a tuple. 
print(input_tensor)

# Outputs
#  Outputs in keras are most commonly a single dense layer, which specifies the shape of the expected output.
from keras.layers import Dense
output_layer = Dense(1)
print(output_layer)
# If you print the output layer, the result is NOT a tensorflow tensor. 
# It is a function, which takes a tensor as input and produces a tensor as output.

##  Layers are used to construct a deep learning model, and tensors are used to define the data flow through the model. 

# Connecting Inputs and Outputs
from keras.layers import Input, Dense
input_tensor = Input(Shape = (1,))
output_layer = Dense(1)
output_tensor = output_layer(input_tensor)
print(output_tensor)
# Now, the final output of our model is a tensor. 







# --------------------------------------------------------------------------------------------------------- #
##                   Input layers                  ##
# Import Input from keras.layers
from keras.layers import Input

# Create an input layer of shape 1
input_tensor = Input(shape = (1,))



##                   Dense layers                  ##
# Load layers
from keras.layers import Input, Dense

# Input layer
input_tensor = Input(shape = (1,))

# Dense layer
output_layer = Dense(1)

# Connect the dense layer to the input_tensor
output_tensor = output_layer(input_tensor)





##                   Output layers                  ##
# Load layers
from keras.layers import Input, Dense

# Input layer
input_tensor = Input(shape = (1,))

# Create a dense layer and connect the dense layer to the input_tensor in one step
# Note that we did this in 2 steps in the previous exercise, but are doing it in one step now
output_tensor = Dense(1)(input_tensor)