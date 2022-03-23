#  Merge layers allow you to define advanced, non-sequential network topologies.
# There are many kinds of merge layers available in Keras. 
# Add, Subtract, and Multiply layers do simple arithmetic operations by element on the input layers, and require them to be the same shape.



# Merge Layers
from keras.layers import Input, Add

in_tensor_1 = Input((1,))
in_tensor_2 = Input((1,))
out_tensor = Add()([in_tensor_1, in_tensor_2])

in_tensor_3 = Input((1,))
out_tensor = Add()([in_tensor_1, in_tensor_2, in_tensor_3])

# Create the Model
from keras.model import Model
model = Model([in_tensor_1, in_tensor_2], out_tensor)
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')









# --------------------------------------------------------------------------------------------------------- #
##                   Output layer using shared layer                  ##
# Combine the two-team strength lookups you did earlier in the Shared Layer

# Import the Subtract layer from keras
from keras.layers import Subtract

# Create a subtract layer using the inputs from the previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])




##                   Model using two inputs and one output                  ##
# Imports
from keras.layers import Subtract
from keras.models import Model

# Subtraction layer from previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# Create the model
model = Model([team_in_1, team_in_2], score_diff)

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

model.summary()
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Team-1-In (InputLayer)          (None, 1)            0                                            
__________________________________________________________________________________________________
Team-2-In (InputLayer)          (None, 1)            0                                            
__________________________________________________________________________________________________
Team-Strength (Model)           (None, 1)            10888       Team-1-In[0][0]                  
                                                                 Team-2-In[0][0]                  
__________________________________________________________________________________________________
subtract_1 (Subtract)           (None, 1)            0           Team-Strength[1][0]              
                                                                 Team-Strength[2][0]              
==================================================================================================
Total params: 10,888
Trainable params: 10,888
Non-trainable params: 0
__________________________________________________________________________________________________