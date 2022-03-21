# Compiling, Training, Predicting in Neural Network
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
# Add an input and dense layer
model.add(Dense(2, input_shape = (3,),  activation = 'relu'))
# Add a final 1 neuron layer
model.add(Dense(1))

# Compiling
model.compile(optimizer = 'adam', loss = 'mse')

# Training
model.fit(X_train, y_train, epochs = 5)

# Predicting
preds = model.predict(X_test)
print(preds)





# --------------------------------------------------------------------------------------------------------- #
##                   Specifying a model                  ##
# This data is stored in two numpy arrays: one called time_steps , what we call features, and another called y_positions, with the labels. 

import numpy as np

# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape = (1,), activation = 'relu'))

# Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))

# End your model with a Dense layer and no activation
model.add(Dense(1))

# It's important to note we aren't using an activation function in our output layer since y_positions aren't bounded and they can take any value. 
# The model is built to perform a regression task.






##                   Training                  ##
# Compile your model
model.compile(optimizer = 'adam', loss = 'mse')

print("Training started..., this can take a while:")

# Fit your model on your data for 30 epochs
model.fit(time_steps, y_positions, epochs = 30)

# Evaluate your model 
print("Final loss value:",model.evaluate(time_steps, y_positions))

# output:
#     Training started..., this can take a while:
#     Epoch 1/30
    
#       32/2000 [..............................] - ETA: 18s - loss: 2465.2439
#      768/2000 [==========>...................] - ETA: 0s - loss: 1839.5495 
#     1504/2000 [=====================>........] - ETA: 0s - loss: 1596.0035
#     2000/2000 [==============================] - 0s 220us/step - loss: 1367.8088
#     Epoch 2/30
    
#       32/2000 [..............................] - ETA: 0s - loss: 513.3908
#      800/2000 [===========>..................] - ETA: 0s - loss: 276.4149
#     1504/2000 [=====================>........] - ETA: 0s - loss: 218.2295
#     2000/2000 [==============================] - 0s 68us/step - loss: 198.7367
# ......................................................
#     Epoch 30/30
    
#       32/2000 [..............................] - ETA: 0s - loss: 0.1128
#      704/2000 [=========>....................] - ETA: 0s - loss: 0.1957
#     1472/2000 [=====================>........] - ETA: 0s - loss: 0.1681
#     2000/2000 [==============================] - 0s 68us/step - loss: 0.1644
    
#       32/2000 [..............................] - ETA: 1s
#     1664/2000 [=======================>......] - ETA: 0s
#     2000/2000 [==============================] - 0s 41us/step
#     Final loss value: 0.1136323138239095






##                   Predicting the orbit!                  ##
# Remember np.arange(x,y) produces a range of values from x to y-1. That is the [x, y) interval.
# plot_orbit source code
def plot_orbit(model_preds):
  axeslim = int(len(model_preds)/2)
  plt.plot(np.arange(-axeslim, axeslim + 1),np.arange(-axeslim, axeslim + 1)**2,color="mediumslateblue")
  plt.plot(np.arange(-axeslim, axeslim + 1),model_preds,color="orange")
  plt.axis([-40, 41, -5, 550])
  plt.legend(["Scientist's Orbit", 'Your orbit'],loc="lower left")
  plt.title("Predicted orbit vs Scientist's Orbit")
  plt.show()

# Part 1
# Predict the twenty minutes orbit
twenty_min_orbit = model.predict(np.arange(-10, 11))

# Plot the twenty minute orbit 
plot_orbit(twenty_min_orbit)

# Part 2
# Predict the eighty minute orbit
eighty_min_orbit = model.predict(np.arange(-40, 41))

# Plot the eighty minute orbit 
plot_orbit(eighty_min_orbit)