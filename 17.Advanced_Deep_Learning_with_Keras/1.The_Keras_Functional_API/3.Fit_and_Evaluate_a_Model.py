# Fit the model
from pandas import read_csv
games = read_csv('games_tourney.csv')
model.fit(games['seed_deff'], games['score_diff'], batch_size = 64, validation_split = 0.20, verbose = True)

# Evaluate model
model.evaluate(games_test['seed_deff'], games_test['score_diff'])







# --------------------------------------------------------------------------------------------------------- #
##                   Fit the model to the tournament basketball data                  ##
# Now fit the model
model.fit(games_tourney_train['seed_diff'], games_tourney_train['score_diff'],
          epochs = 1,
          batch_size = 128,
          validation_split = 0.10,
          verbose = True)

# output:
#     Train on 3087 samples, validate on 343 samples
#     Epoch 1/1
#     
#      128/3087 [>.............................] - ETA: 4s - loss: 12.6147
#     1536/3087 [=============>................] - ETA: 0s - loss: 12.8213
#     3087/3087 [==============================] - 0s 96us/step - loss: 12.6617 - val_loss: 11.8747





##                   Evaluate the model on a test set                  ##
# Load the X variable from the test data
X_test = games_tourney_test['seed_diff']

# Load the y variable from the test data
y_test = games_tourney_test['score_diff']

# Evaluate the model on the test data
print(model.evaluate(X_test, y_test, verbose = False))
# output: 10.06973339669147