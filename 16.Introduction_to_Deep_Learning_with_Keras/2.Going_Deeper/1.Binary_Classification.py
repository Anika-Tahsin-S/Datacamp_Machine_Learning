# Pairplots
import seaborn as sns
sns.pairplot(circles, hue = 'target')

# The sigmoid function
# The sigmoid activation function squashes the neuron output of the second to last layer to a floating point number between 0 and 1. 

from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
# Add an input and dense layer
model.add(Dense(4, input_shape = (2,),  activation = 'tanh'))
# Add a final 1 neuron layer
model.add(Dense(1, activation = 'sigmoid'))

# Compiling
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy')

# Training
model.fit(coordinates, labels, epochs = 20)

# Predicting
preds = model.predict(coordinates)
print(preds)






# --------------------------------------------------------------------------------------------------------- #
##                   Exploring dollar bills                  ##
# The dataset comes with 4 features: variance,skewness,kurtosis and entropy. These features are calculated by applying mathematical operations over the dollar bill images. The labels are found in the dataframe's class column.
# A pandas DataFrame named banknotes is ready to use, let's do some data exploration!

# Import seaborn
import seaborn as sns
import pandas

# Use pairplot and set the hue to be our class column
sns.pairplot(banknotes, hue = 'class') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations per class
print('Observations per class: \n', banknotes['class'].value_counts())

# output:
#     Dataset stats: 
#              variance   skewness   curtosis    entropy
#     count  96.000000  96.000000  96.000000  96.000000
#     mean   -0.057791  -0.102829   0.230412   0.081497
#     std     1.044960   1.059236   1.128972   0.975565
#     min    -2.084590  -2.621646  -1.482300  -3.034187
#     25%    -0.839124  -0.916152  -0.415294  -0.262668
#     50%    -0.026748  -0.037559  -0.033603   0.394888
#     75%     0.871034   0.813601   0.978766   0.745212
#     max     1.869239   1.634072   3.759017   1.343345
#     Observations per class: 
#      real    53
#     fake    43
#     Name: class, dtype: int64







##                   A binary classification model                  ##
# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()

# Add a dense layer 
model.add(Dense(4, input_shape = (2,),  activation = 'sigmoid'))

# Compile your model
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# Display a summary of your model
model.summary()

# output:
#     Model: "sequential_1"
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     dense_1 (Dense)              (None, 1)                 5         
#     =================================================================
#     Total params: 5
#     Trainable params: 5
#     Non-trainable params: 0
#     _________________________________________________________________







##                   Is this dollar bill fake ?                  ##
# The dataset has already been partitioned into features: X_train & X_test, and labels: y_train & y_test.

# Train your model for 20 epochs
model.fit(X_train, y_train, epochs = 20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

# output:
#     Epoch 1/20
    
#      32/960 [>.............................] - ETA: 6s - loss: 0.7641 - acc: 0.5000
#     768/960 [=======================>......] - ETA: 0s - loss: 0.6658 - acc: 0.6523
#     960/960 [==============================] - 0s 285us/step - loss: 0.6655 - acc: 0.6531
#     Epoch 2/20
#     
#      32/960 [>.............................] - ETA: 0s - loss: 0.6201 - acc: 0.5938
#     768/960 [=======================>......] - ETA: 0s - loss: 0.6339 - acc: 0.6745
#     960/960 [==============================] - 0s 74us/step - loss: 0.6429 - acc: 0.6698
#   ..........................................................................................
#     Epoch 20/20
#     
#      32/960 [>.............................] - ETA: 0s - loss: 0.4655 - acc: 0.7812
#     672/960 [====================>.........] - ETA: 0s - loss: 0.4299 - acc: 0.8214
#     960/960 [==============================] - 0s 90us/step - loss: 0.4266 - acc: 0.8250
#     
#      32/412 [=>............................] - ETA: 0s
#     412/412 [==============================] - 0s 101us/step
#     Accuracy: 0.8252427167105443