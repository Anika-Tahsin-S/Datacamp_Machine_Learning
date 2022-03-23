# Keras models with multiple inputs work just like Keras models with a single input. They use the same fit, evaluate, and predict methods. 
# The only difference is that all of these methods take lists of inputs, rather a single input. 

# Fit with Multiple inputs
model.fit([data_1, data_2], target)

# Predict with Multiple inputs
model.predict([np.array([[1]]), np.array([[2]])])
array([[3.]]), dtype = float32)

model.predict([np.array([[42]]), np.array([[119]])])
array([[161.]]), dtype = float32)

# Evaluate with Multiple inputs
model.evaluate([np.array([[-1]]), np.array([[-2]])], np.array([[-3]]))










# --------------------------------------------------------------------------------------------------------- #
##                   Fit the model to the regular season training data                  ##
# Get the team_1 column from the regular season data
input_1 = games_season['team_1']

# Get the team_2 column from the regular season data
input_2 = games_season['team_2']

# Fit the model to input 1 and 2, using score diff as a target
model.fit([input_1, input_2], games_season['score_diff'], 
          epochs = 1, batch_size = 2048,
          validation_split = 0.1, verbose = True)

# output:
#     Train on 280960 samples, validate on 31218 samples
#     Epoch 1/1
#     
#       2048/280960 [..............................] - ETA: 14s - loss: 12.0254
#      30720/280960 [==>...........................] - ETA: 1s - loss: 12.2530 
#      61440/280960 [=====>........................] - ETA: 0s - loss: 12.1837
#      92160/280960 [========>.....................] - ETA: 0s - loss: 12.1353
#     126976/280960 [============>.................] - ETA: 0s - loss: 12.1233
#     161792/280960 [================>.............] - ETA: 0s - loss: 12.1343
#     192512/280960 [===================>..........] - ETA: 0s - loss: 12.1235
#     227328/280960 [=======================>......] - ETA: 0s - loss: 12.1257
#     264192/280960 [===========================>..] - ETA: 0s - loss: 12.1269
#     280960/280960 [==============================] - 1s 2us/step - loss: 12.1203 - val_loss: 11.8384







##                   Evaluate the model on the tournament test data                  ##
# Get team_1 from the tournament data
input_1 = games_tourney['team_1']

# Get team_2 from the tournament data
input_2 = games_tourney['team_2']

# Evaluate the model using these inputs
print(model.evaluate([input_1, input_2], games_tourney['score_diff'], verbose = False))

# output: 11.649009290595183