# Stacking models requires given 2 datasets 
import pandas as pd
games_season = pd.read_csv('games_season.csv')
games_tourney = pd.read_csv('games_tourney.csv')

# Enrich the tournament data
in_data_1 = games_tourney['team_1']
in_data_2 = games_tourney['team_2']
in_data_3 = games_tourney['home']
pred = regular_season_model.predict([in_data_1, in_data_2, in_data_3])

games_tourney['pred'] = pred
games_tourney.head()

# 3 input model with pure numeric data
games_tourney[['home', 'seed_diff', 'pred']].head()


# A huge advantage of this approach is simplicity. 
# You can create a model with a single input tensor and an output tensor, and fit it using a single dataset. 

from keras.layers import Input, Dense

in_tensor = Input(shape = (3,))
out_tensor = Dense(1)(in_tensor)

# Create the Model
from keras.model import Model
model = Model(in_tensor, out_tensor)
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
train_X = train_data[['home', 'seed_diff', 'pred']]
train_y = train_data['score_diff']

model.fit(train_X, train_y, epochs = 10, validation_split = 0.10)

test_X = test_data[['home', 'seed_diff', 'pred']]
test_y = test_data['score_diff']
model.evaluate(test_X, test_y)

# To recap: stacking keras models means using the predictions from one model as an input to a second model.
# When stacking, it's important to use different datasets for each model.









# --------------------------------------------------------------------------------------------------------- #
##                   Add the model predictions to the tournament data                  ##
# Predict
games_tourney['pred'] = model.predict([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']])



##                   Create an input layer with multiple columns                  ##
# Create an input layer with 3 columns
input_tensor = Input((3,))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')


model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 3)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 4         
=================================================================
Total params: 4
Trainable params: 4
Non-trainable params: 0
_________________________________________________________________





##                   Fit the model                  ##
# Fit the model
model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'],
          epochs = 1,
          verbose = True)

# output:
#     Epoch 1/1
#     
#       32/3168 [..............................] - ETA: 13s - loss: 21.2439
#      800/3168 [======>.......................] - ETA: 0s - loss: 18.5027 
#     1568/3168 [=============>................] - ETA: 0s - loss: 18.2717
#     2336/3168 [=====================>........] - ETA: 0s - loss: 18.1234
#     3136/3168 [============================>.] - ETA: 0s - loss: 17.8214
#     3168/3168 [==============================] - 0s 107us/step - loss: 17.8147





##                   Evaluate the model                  ##
# Evaluate the model on the games_tourney_test dataset
# Evaluate the model on the games_tourney_test dataset
print(model.evaluate(games_tourney_test[['home', 'seed_diff', 'prediction']], games_tourney_test['score_diff'], verbose = False))

# output: 9.07315489498804