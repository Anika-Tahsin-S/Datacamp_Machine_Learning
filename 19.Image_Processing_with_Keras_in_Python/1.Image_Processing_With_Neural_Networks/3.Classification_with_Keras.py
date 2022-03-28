# Keras for image classification
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
train_data.shape
model.add(Dense(10, activation = 'relu', input_shape = (784,)))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metric = ['accuracy'])
train_data = train_data.reshape((50, 784))
model.fit(train_data, train_labels, validation_split = 0.2, epochs = 3)

test_data = test_data.reshape((10, 784))
model.evaluate(test_data, test_labels)







# --------------------------------------------------------------------------------------------------------- #
##                  Build a neural network                  ##
# Imports components from Keras
from keras.models import Sequential
from keras.layers import Dense

# Initializes a sequential model
model = Sequential()

# First layer
model.add(Dense(10, activation = 'relu', input_shape = (784,)))

# Second layer
model.add(Dense(10, activation = 'relu'))

# Output layer
model.add(Dense(3, activation = 'softmax'))

model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 10)                7850      
_________________________________________________________________
dense_1 (Dense)              (None, 10)                110       
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 33        
=================================================================
Total params: 7,993
Trainable params: 7,993
Non-trainable params: 0
_________________________________________________________________







##                  Compile a neural network                  ##
# Compile the model
model.compile(optimizer = 'adam', 
           loss = 'categorical_crossentropy', 
           metrics = ['accuracy'])





##                  Fitting a neural network model to clothing data                  ##
# Collected #
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_data = train_data[(train_labels >= 0) & (train_labels < 3)][0:50].reshape(-1, 28, 28, 1)
train_labels = train_labels[(train_labels >= 0) & (train_labels < 3)][0:50]
train_labels = pd.get_dummies(train_labels).to_numpy()

test_data = test_data[(test_labels >= 0) & (test_labels < 3)][0:10].reshape(-1, 28, 28, 1)
test_labels = test_labels[(test_labels >= 0) & (test_labels < 3)][0:10]
test_labels = pd.get_dummies(test_labels).to_numpy()
# ==================================================================================================== #


train_data

array([[[[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]],

.....................

        [[0.],
         [0.],
         [0.],
         ...,
         [0.],
         [0.],
         [0.]]]], dtype=float32)

train_labels

array([[0., 1., 0.],
       [0., 1., 0.],
........................
       [1., 0., 0.],
       [1., 0., 0.]])


# Reshape the data to two-dimensional array
train_data = train_data.reshape(50, 784)

# Fit the model
model.fit(train_data, train_labels, validation_split = 0.2, epochs = 3)

# output:
#    Epoch 1/3
#   
# 1/2 [==============>...............] - ETA: 1s - loss: 1.1410 - accuracy: 0.18752/2 
#     [==============================] - 2s 555ms/step - loss: 1.1085 - accuracy: 0.2750 - val_loss: 1.1623 - val_accuracy: 0.3000
#     Epoch 2/3
#     
# 1/2 [==============>...............] - ETA: 0s - loss: 0.9680 - accuracy: 0.62502/2 
#     [==============================] - 0s 26ms/step - loss: 0.9684 - accuracy: 0.6250 - val_loss: 1.0922 - val_accuracy: 0.3000
#     Epoch 3/3
#     
# 1/2 [==============>...............] - ETA: 0s - loss: 0.8964 - accuracy: 0.62502/2 
#     [==============================] - 0s 22ms/step - loss: 0.8759 - accuracy: 0.6500 - val_loss: 1.0566 - val_accuracy: 0.3000






##                  Cross-validation for neural network evaluation                  ##
# Reshape test data
test_data = test_data.reshape(10, 784)

# Evaluate the model
model.evaluate(test_data, test_labels)

# output:
#     1/1 [==============================] - ETA: 0s - loss: 0.8370 - accuracy: 0.70001/1 
#     [==============================] - 0s 25ms/step - loss: 0.8370 - accuracy: 0.7000