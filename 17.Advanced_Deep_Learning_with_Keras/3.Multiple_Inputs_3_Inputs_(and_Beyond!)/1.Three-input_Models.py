# Simple model with 3 inputs
from keras.layers import Input, Concatenate, Dense

in_tensor_1 = Input((1,))
in_tensor_2 = Input((1,))
in_tensor_3 = Input((1,))
out_tensor = Concatenate()([in_tensor_1, in_tensor_2, in_tensor_3])
output_tensor = Dense(1)(out_tensor)

# Shared model with 3 inputs
shared_layer = Dense(1)
shared_tensor_1 = shared_layer(in_tensor_1)
shared_tensor_2 = shared_layer(in_tensor_2)
out_tensor = Concatenate()([shared_tensor_1, shared_tensor_2, in_tensor_3])
output_tensor = Dense(1)(out_tensor)

# Create the Model
from keras.model import Model
model = Model([in_tensor_1, in_tensor_2, in_tensor_3], out_tensor)
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

model.fit([[train['col_1'], train['col_2'], train['col_3']], train_data['target'])

model.evaluate([[test['col_1'], test['col_2'], test['col_3']], test['target'])









# --------------------------------------------------------------------------------------------------------- #
##                   Make an input layer for home vs. away                  ##
# Create an Input for each team
team_in_1 = Input(shape = (1,), name = 'Team-1-In')
team_in_2 = Input(shape = (1,), name = 'Team-2-In')

# Create an input for home vs away
home_in = Input(shape = (1,), name = 'Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer, then add a Dense layer
out = Concatenate()([team_1_strength, team_2_strength, home_in])
output = Dense(1)(out)







##                   Make a model and compile it                  ##
# Import the model class
from keras.models import Model

# Make a Model
model = Model([team_in_1, team_in_2, home_in], out)

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')





##                   Fit the model and evaluate                  ##
# Fit the model to the games_season dataset
model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'],
          epochs = 1,
          verbose = True,
          validation_split = 0.1,
          batch_size = 2048)

# Evaluate the model on the games_tourney dataset
print(model.evaluate([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']],
          games_tourney['score_diff'], verbose = False))

# output:
#     Train on 280960 samples, validate on 31218 samples
#     Epoch 1/1
#     
#       2048/280960 [..............................] - ETA: 27s - loss: 12.0596
#      20480/280960 [=>............................] - ETA: 3s - loss: 12.0037 
#      49152/280960 [====>.........................] - ETA: 1s - loss: 12.0264
#      73728/280960 [======>.......................] - ETA: 0s - loss: 12.0126
#      98304/280960 [=========>....................] - ETA: 0s - loss: 11.9921
#     120832/280960 [===========>..................] - ETA: 0s - loss: 12.0180
#     147456/280960 [==============>...............] - ETA: 0s - loss: 12.0082
#     176128/280960 [=================>............] - ETA: 0s - loss: 12.0169
#     204800/280960 [====================>.........] - ETA: 0s - loss: 12.0192
#     227328/280960 [=======================>......] - ETA: 0s - loss: 12.0223
#     249856/280960 [=========================>....] - ETA: 0s - loss: 12.0115
#     278528/280960 [============================>.] - ETA: 0s - loss: 11.9989
#     280960/280960 [==============================] - 1s 3us/step - loss: 12.0003 - val_loss: 12.3391
#     11.683796553325697



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
Home-In (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 3)            0           Team-Strength[1][0]              
                                                                 Team-Strength[2][0]              
                                                                 Home-In[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            4           concatenate_1[0][0]              
==================================================================================================
Total params: 10,892
Trainable params: 10,892
Non-trainable params: 0
__________________________________________________________________________________________________