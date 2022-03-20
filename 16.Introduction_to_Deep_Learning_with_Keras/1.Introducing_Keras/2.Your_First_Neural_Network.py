# Defining and Adding Activations in Neural Network
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
# Add an input and dense layer
model.add(Dense(2, input_shape = (3,),  activation = 'relu'))
# Add a final 1 neuron layer
model.add(Dense(1))

model.summary()





# --------------------------------------------------------------------------------------------------------- #
##                   Hello nets!                  ##
# Import the Sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape = (2,), activation = "relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()

# output:
#     Model: "sequential_1"
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     dense_1 (Dense)              (None, 10)                30        
#     _________________________________________________________________
#     dense_2 (Dense)              (None, 1)                 11        
#     =================================================================
#     Total params: 41
#     Trainable params: 41
#     Non-trainable params: 0
#     _________________________________________________________________






##                   Counting parameters                  ##
# Part 1
# Instantiate a new Sequential model
model = Sequential()

# Add a Dense layer with five neurons and three inputs
model.add(Dense(5, input_shape = (3,), activation = "relu"))

# Add a final Dense layer with one neuron and no activation
model.add(Dense(1))
# Summarize your model
model.summary()

# output:
#     Model: "sequential_1"
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     dense_1 (Dense)              (None, 5)                 20        
#     _________________________________________________________________
#     dense_2 (Dense)              (None, 1)                 6         
#     =================================================================
#     Total params: 26
#     Trainable params: 26
#     Non-trainable params: 0
#     _________________________________________________________________



# Part 2
# Given the model you just built, which answer is correct regarding the number of weights (parameters) in the hidden layer?
# There are 20 parameters, 15 from the connections of our inputs to our hidden layer and 5 from the bias weight of each neuron in the hidden layer.





##                   Build as shown!                  ##
from keras.models import Sequential
from keras.layers import Dense

# Instantiate a Sequential model
model = Sequential()

# Build the input and hidden layer
model.add(Dense(3, input_shape = (2,), activation = "relu"))

# Add the ouput layer
model.add(Dense(1))