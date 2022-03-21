# When compiling your model, instead of binary cross-entropy as we used before, we now use categorical cross-entropy or log loss.
# Categorical cross-entropy measures the difference between the predicted probabilities and the true label of the class we should have predicted.
# So if we should have predicted 1 for a given class, taking a look at the graph we see we would get high loss values for predicting close to 0 (since we'd be very wrong) and low loss values for predicting closer to 1 (the true label). 

# Preparing a dataset
import pandas as pd
from keras.utils import to_categorical

df = pd.read_csv('data.csv')

# Turn response variable into labeled codes
df.response = pd.Categorical(df.response)
df.response = df.response.cat.codes

# Turn response variable into one-hot response vector
y = to_categorical(df.response)






# --------------------------------------------------------------------------------------------------------- #
##                   A multi-class model                  ##
# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape = (2,), activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation = 'softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])






##                   Prepare your Dataset                  ##
import pandas as pd

darts.head()

     xCoord    yCoord competitor
0  0.196451 -0.520341      Steve
1  0.476027 -0.306763      Susan
2  0.003175 -0.980736    Michael
3  0.294078  0.267566       Kate
4 -0.051120  0.598946      Steve

darts.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 800 entries, 0 to 799
Data columns (total 3 columns):
xCoord        800 non-null float64
yCoord        800 non-null float64
competitor    800 non-null object
dtypes: float64(2), object(1)
memory usage: 18.8+ KB


# Part 1
# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Print the label encoded competitors
print('Label encoded competitors: \n',darts.competitor.head())

# output:
#     Label encoded competitors: 
#      0    2
#     1    3
#     2    1
#     3    0
#     4    2
#     Name: competitor, dtype: int8

# Part 2
# Import to_categorical from keras utils module
from keras.utils import to_categorical

coordinates = darts.drop(['competitor'], axis=1)
# Use to_categorical on your labels
competitors = to_categorical(darts.competitor)

# Now print the one-hot encoded labels
print('One-hot encoded competitors: \n',competitors)

# output:
#     One-hot encoded competitors: 
#      [[0. 0. 1. 0.]
#      [0. 0. 0. 1.]
#      [0. 1. 0. 0.]
#      ...
#      [0. 1. 0. 0.]
#      [0. 1. 0. 0.]
#      [0. 0. 0. 1.]]








##                   Training on dart throwers                  ##
# The coordinates features and competitors labels you just transformed have been partitioned into coord_train,coord_test and competitors_train,competitors_test.
# Fit your model to the training data for 200 epochs
model.fit(coord_train, competitors_train, epochs = 200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

# output:
#     Epoch 1/200
#     
#      32/640 [>.............................] - ETA: 4s - loss: 1.3952 - acc: 0.3750
#     640/640 [==============================] - 0s 458us/step - loss: 1.3863 - acc: 0.3047
#   ................................................................
#     Epoch 200/200
#     
#      32/640 [>.............................] - ETA: 0s - loss: 0.3470 - acc: 0.8125
#     544/640 [========================>.....] - ETA: 0s - loss: 0.5331 - acc: 0.7941
#     640/640 [==============================] - 0s 126us/step - loss: 0.5125 - acc: 0.8000
#     
#      32/160 [=====>........................] - ETA: 0s
#     160/160 [==============================] - 0s 378us/step
#     Accuracy: 0.84375









##                   Softmax predictions                  ##
# Part 1
# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# output:
#     Raw Model Predictions                         | True labels
#     [0.34438723 0.00842557 0.63167274 0.01551455] | [0. 0. 1. 0.]
#     [0.0989717  0.00530467 0.07537904 0.8203446 ] | [0. 0. 0. 1.]
#     [0.33512568 0.00785374 0.28132284 0.37569773] | [0. 0. 0. 1.]
#     [0.8547263  0.01328656 0.11279515 0.01919206] | [1. 0. 0. 0.]
#     [0.3540977  0.00867271 0.6223853  0.01484426] | [0. 0. 1. 0.]




# Part 2
import numpy as np

# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# Extract the position of highest probability from each pred vector
preds_chosen = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds_chosen):
  print("{:25} | {}".format(pred,competitors_small_test[i]))

# output:
#     Raw Model Predictions                         | True labels
#     [0.34438723 0.00842557 0.63167274 0.01551455] | [0. 0. 1. 0.]
#     [0.0989717  0.00530467 0.07537904 0.8203446 ] | [0. 0. 0. 1.]
#     [0.33512568 0.00785374 0.28132284 0.37569773] | [0. 0. 0. 1.]
#     [0.8547263  0.01328656 0.11279515 0.01919206] | [1. 0. 0. 0.]
#     [0.3540977  0.00867271 0.6223853  0.01484426] | [0. 0. 1. 0.]
#     Rounded Model Predictions | True labels
#                             2 | [0. 0. 1. 0.]
#                             3 | [0. 0. 0. 1.]
#                             3 | [0. 0. 0. 1.]
#                             0 | [1. 0. 0. 0.]
#                             2 | [0. 0. 1. 0.]