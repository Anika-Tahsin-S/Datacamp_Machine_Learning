# Compiling and fitting Model
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

pred = np.loadtxt('predictions_data.csv', delimiter = ',')
n_cols = pred.shape[1]

model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (n_cols, ))
model.add(Dense(100, activation = 'relu')
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(pred, target)





# --------------------------------------------------------------------------------------------------------- #
##                   Compiling the Model                  ##
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)
# output: Loss function: mean_squared_error




##                   Fitting the Model                  ##
# The data to be used as predictive features is loaded in a NumPy matrix called predictors and the data to be predicted is stored in a NumPy matrix called target. 
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit the model
model.fit(predictors, target)

# output:
#     Epoch 1/10
    
#  32/534 [>.............................] - ETA: 23s - loss: 146.0927320/534 
#         [================>.............] - ETA: 1s - loss: 100.8006534/534 
#         [==============================] - 1s - loss: 77.7563      
#     Epoch 2/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 84.7528512/534 
#         [===========================>..] - ETA: 0s - loss: 29.6869534/534 
#         [==============================] - 0s - loss: 30.2410     
#     Epoch 3/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 21.0520534/534 
#         [==============================] - 0s - loss: 27.0648     
#     Epoch 4/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 16.8140534/534 
#         [==============================] - 0s - loss: 25.1152     
#     Epoch 5/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 23.1975534/534 
#         [==============================] - 0s - loss: 24.0130     
#     Epoch 6/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 13.4018480/534 
#         [=========================>....] - ETA: 0s - loss: 23.4701534/534 
#         [==============================] - 0s - loss: 23.1953     
#     Epoch 7/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 28.1658534/534 
#         [==============================] - 0s - loss: 22.4476     
#     Epoch 8/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 11.3793534/534 
#         [==============================] - 0s - loss: 22.0779     
#     Epoch 9/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 21.9343534/534   
#         [==============================] - 0s - loss: 21.7463     
#     Epoch 10/10
    
#  32/534 [>.............................] - ETA: 0s - loss: 5.4732534/534   
#         [==============================] - 0s - loss: 21.5527    