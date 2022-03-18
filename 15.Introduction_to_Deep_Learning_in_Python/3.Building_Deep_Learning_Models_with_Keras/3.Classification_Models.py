# For classification we set the loss function as 'categorical_crossentropy' instead of 'mean_squared_error'.
# The argument "metrics equals accuracy" means that I want to print out the accuracy score at the end of each epoch, which makes it easier to see and understand the models progress. 
# Also change the activation function to softmax. The softmax activation function ensures the predictions sum to 1, so they can be interpreted like probabilities. 

from keras.utils.np_utils import to_categorical
import pandas as pd

data = pd.read_csv('basketball_shot_log.csv')
pred = data.drop(['shot_results'], axis = 1).as_matrix()
target = to_categorical(data.shot_result)

model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (n_cols, ))
model.add(Dense(100, activation = 'relu')
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(pred, target)








# --------------------------------------------------------------------------------------------------------- #
##                   Understanding your Classification Data                  ##
# The data is pre-loaded in a pandas DataFrame called df.

# It's smart to review the maximum and minimum values of each variable to ensure the data isn't misformatted or corrupted. What was the maximum age of passengers on the Titanic?
df.describe()

         survived      pclass         age       sibsp       parch        fare        male  embarked_from_cherbourg  embarked_from_queenstown  embarked_from_southampton
count  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000               891.000000                891.000000                 891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208    0.647587                 0.188552                  0.086420                   0.722783
std      0.486592    0.836071   13.002015    1.102743    0.806057   49.693429    0.477990                 0.391372                  0.281141                   0.447876
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000    0.000000                 0.000000                  0.000000                   0.000000
25%      0.000000    2.000000   22.000000    0.000000    0.000000    7.910400    0.000000                 0.000000                  0.000000                   0.000000
50%      0.000000    3.000000   29.699118    0.000000    0.000000   14.454200    1.000000                 0.000000                  0.000000                   1.000000
75%      1.000000    3.000000   35.000000    1.000000    0.000000   31.000000    1.000000                 0.000000                  0.000000                   1.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200    1.000000                 1.000000                  1.000000                   1.000000

# Answer: 80. The maximum age in the data is 80.





##                   Last steps in Classification Models                  ##
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model
model.fit(predictors, target)

# output:
#     Epoch 1/10
    
#  32/891 [>.............................] - ETA: 33s - loss: 7.6250 - acc: 0.2188448/891 
#         [==============>...............] - ETA: 1s - loss: 3.0348 - acc: 0.5893 864/891 
#         [============================>.] - ETA: 0s - loss: 2.5594 - acc: 0.5926891/891 
#         [==============================] - 1s - loss: 2.5604 - acc: 0.5903     
#     Epoch 2/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 1.4954 - acc: 0.3438480/891 
#         [===============>..............] - ETA: 0s - loss: 1.8984 - acc: 0.5271891/891 
#         [==============================] - 0s - loss: 1.4070 - acc: 0.5993     
#     Epoch 3/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 1.7638 - acc: 0.5000448/891 
#         [==============>...............] - ETA: 0s - loss: 0.8898 - acc: 0.6272864/891 
#         [============================>.] - ETA: 0s - loss: 0.7538 - acc: 0.6632891/891 
#         [==============================] - 0s - loss: 0.7524 - acc: 0.6577     
#     Epoch 4/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 0.7166 - acc: 0.5938480/891 
#         [===============>..............] - ETA: 0s - loss: 0.6494 - acc: 0.6708891/891 
#         [==============================] - 0s - loss: 0.7053 - acc: 0.6712     
#     Epoch 5/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 0.6150 - acc: 0.5938512/891 
#         [================>.............] - ETA: 0s - loss: 0.6368 - acc: 0.6602891/891 
#         [==============================] - 0s - loss: 0.6526 - acc: 0.6532     
#     Epoch 6/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 0.4687 - acc: 0.7500544/891 
#         [=================>............] - ETA: 0s - loss: 0.6032 - acc: 0.6857891/891 
#         [==============================] - 0s - loss: 0.6179 - acc: 0.6880     
#     Epoch 7/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 0.6612 - acc: 0.5938448/891 
#         [==============>...............] - ETA: 0s - loss: 0.6469 - acc: 0.6719704/891 
#         [======================>.......] - ETA: 0s - loss: 0.6457 - acc: 0.6747891/891 
#         [==============================] - 0s - loss: 0.6332 - acc: 0.6824     
#     Epoch 8/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 0.5294 - acc: 0.7500544/891 
#         [=================>............] - ETA: 0s - loss: 0.6524 - acc: 0.6691891/891 
#         [==============================] - 0s - loss: 0.6315 - acc: 0.6813     
#     Epoch 9/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 0.6901 - acc: 0.6562448/891 
#         [==============>...............] - ETA: 0s - loss: 0.6311 - acc: 0.6741891/891 
#         [==============================] - 0s - loss: 0.6013 - acc: 0.6936     
#     Epoch 10/10
    
#  32/891 [>.............................] - ETA: 0s - loss: 0.4896 - acc: 0.75000448/891 
#         [==============>...............] - ETA: 0s - loss: 0.6317 - acc: 0.7031891/891 
#         [==============================] - 0s - loss: 0.6526 - acc: 0.6779     

# This simple model is generating an accuracy of 68!