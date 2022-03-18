##                   Building your own digit recognition model                  ##
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation = 'relu', input_shape = (784, )))

# Add the second hidden layer
model.add(Dense(50, activation = 'relu', input_shape = (784,)))

# Add the output layer
model.add(Dense(10, activation = 'softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model
model.fit(X, y, validation_split = 0.3)

# output:
#     Train on 1750 samples, validate on 750 samples
#     Epoch 1/10
    
#   32/1750 [..............................] - ETA: 103s - loss: 2.1979 - acc: 0.218852/1750 
#           [=====>........................] - ETA: 7s - loss: 2.2179 - acc: 0.2017  640/1750 
#           [=========>....................] - ETA: 3s - loss: 2.1065 - acc: 0.2719960/1750 
#           [===============>..............] - ETA: 1s - loss: 1.9770 - acc: 0.32811312/1750 
#           [=====================>........] - ETA: 0s - loss: 1.8433 - acc: 0.38721664/1750 
#           [===========================>..] - ETA: 0s - loss: 1.7080 - acc: 0.45011750/1750 
#           [==============================] - 2s - loss: 1.6770 - acc: 0.4646 - val_loss: 1.0119 - val_acc: 0.7680
# .............................................................................
#     Epoch 10/10
    
#   32/1750 [..............................] - ETA: 0s - loss: 0.1536 - acc: 0.9688384/1750 
#           [=====>........................] - ETA: 0s - loss: 0.1110 - acc: 0.9714736/1750 
#           [===========>..................] - ETA: 0s - loss: 0.1097 - acc: 0.97011056/1750 
#           [=================>............] - ETA: 0s - loss: 0.0996 - acc: 0.97731408/1750 
#           [=======================>......] - ETA: 0s - loss: 0.1003 - acc: 0.97801750/1750 
#           [==============================] - 0s - loss: 0.0966 - acc: 0.9789 - val_loss: 0.3127 - val_acc: 0.6


# You should see better than 90% accuracy recognizing handwritten digits, even while using a small training set of only 1750 images!