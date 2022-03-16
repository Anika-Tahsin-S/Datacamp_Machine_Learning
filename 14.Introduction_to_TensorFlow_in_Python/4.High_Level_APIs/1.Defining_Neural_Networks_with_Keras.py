# Building keras sequential
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Dense(16, activation = 'relu', input_shape = (28*28)))

model.add(keras.layers.Dense(8, activation = 'relu'))

model.add(keras.layers.Dense(4, activation = 'softmax'))

# Compile model
model.compile('adam', loss = 'categorical_crossentropy')
print(model.summary())

# For two model joining for same output, we use functional API
import tensorflow as tf

model1_inputs = tf.keras.Input(shape = (28 * 28,))
model2_inputs = tf.keras.Input(shape = (10,))

# Define Layer 1,2 for model 1
model1_layer1 = tf.keras.layers.Dense(12, activation = 'relu')(model1_inputs)
model1_layer2 = tf.keras.layers.Dense(4, activation = 'softmax')(model1_layer1)

# Define Layer 1,2 for model 2
model2_layer1 = tf.keras.layers.Dense(8, activation = 'relu')(model2_inputs)
model2_layer2 = tf.keras.layers.Dense(4, activation = 'softmax')(model2_layer1)

# Merge 
merged = tf.koreas.Model(inputs = [model1_inputs, model2_inputs], outputs = merged)
model.compile('adam', loss = 'categorical_crossentrophy')






# --------------------------------------------------------------------------------------------------------- #
##                   The Sequential Model in Keras                  ##
from tensorflow import keras
# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation = 'relu', input_shape = (784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation = 'relu', input_shape = (784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation = 'softmax'))

# Print the model architecture
print(model.summary())

# output:
#     Model: "sequential"
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     dense (Dense)                (None, 16)                12560     
#     _________________________________________________________________
#     dense_1 (Dense)              (None, 8)                 136       
#     _________________________________________________________________
#     dense_2 (Dense)              (None, 4)                 36        
#     =================================================================
#     Total params: 12,732
#     Trainable params: 12,732
#     Non-trainable params: 0
#     _________________________________________________________________
#     None








##                   Compiling a Sequential Model                  ##
# Define the first dense layer
model.add(keras.layers.Dense(16, activation = 'sigmoid', input_shape = (784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4, activation = 'softmax'))

# Compile the model
model.compile('adam', loss = 'categorical_crossentropy')

# Print a model summary
print(model.summary())

# output:
#     Model: "sequential"
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     dense (Dense)                (None, 16)                12560     
#     _________________________________________________________________
#     dropout (Dropout)            (None, 16)                0         
#     _________________________________________________________________
#     dense_1 (Dense)              (None, 4)                 68        
#     =================================================================
#     Total params: 12,628
#     Trainable params: 12,628
#     Non-trainable params: 0
#     _________________________________________________________________
#     None







##                   Defining a Multiple Input Model                  ##
# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation = 'sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation = 'softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation = 'relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation = 'softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs = [m1_inputs, m2_inputs], outputs = merged)

# Print a model summary
print(model.summary())

# output:
#     Model: "model"
#     __________________________________________________________________________________________________
#     Layer (type)                    Output Shape         Param #     Connected to                     
#     ==================================================================================================
#     input_1 (InputLayer)            [(None, 784)]        0                                            
#     __________________________________________________________________________________________________
#     input_2 (InputLayer)            [(None, 784)]        0                                            
#     __________________________________________________________________________________________________
#     dense (Dense)                   (None, 12)           9420        input_1[0][0]                    
#     __________________________________________________________________________________________________
#     dense_2 (Dense)                 (None, 12)           9420        input_2[0][0]                    
#     __________________________________________________________________________________________________
#     dense_1 (Dense)                 (None, 4)            52          dense[0][0]                      
#     __________________________________________________________________________________________________
#     dense_3 (Dense)                 (None, 4)            52          dense_2[0][0]                    
#     __________________________________________________________________________________________________
#     add (Add)                       (None, 4)            0           dense_1[0][0]                    
#                                                                      dense_3[0][0]                    
#     ==================================================================================================
#     Total params: 18,944
#     Trainable params: 18,944
#     Non-trainable params: 0
#     __________________________________________________________________________________________________
#     None