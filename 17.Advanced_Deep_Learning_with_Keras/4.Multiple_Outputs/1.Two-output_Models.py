# Simple model with 2 outputs
from keras.layers import Input, Concatenate, Dense

in_tensor = Input(shape = (1,))
out_tensor = Dense(2)(in_tensor)

# Create the Model
from keras.model import Model
model = Model(in_tensor, out_tensor)
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

games_tourney[['seed_diff', 'score_1', 'score_2']].head()

X = games_tourney_train[['seed_diff']]
y = games_tourney_train[['score_1', 'score_2']]
model.fit(X, y, epochs = 500)

# Inspecting a 2 output model
model.get_weights()


# Evaluating a model 2 outputs
X = games_tourney_test[['seed_diff']]
y = games_tourney_test[['score_1', 'score_2']]
model.evaluate(X, y)









# --------------------------------------------------------------------------------------------------------- #
##                   Simple two-output model                  ##
# Define the input
input_tensor = Input((2,))

# Define the output
output_tensor = Dense(2)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')






##                   Fit a model with two outputs                  ##
# Fit the model
model.fit(games_tourney_train[['seed_diff', 'pred']],
  		  games_tourney_train[['score_1', 'score_2']],
  		  verbose = True,
  		  epochs = 100,
  		  batch_size = 16384)

# output:
#     Epoch 1/100
#     
#     3430/3430 [==============================] - 0s 29us/step - loss: 71.6213
# ...............................................................................
# Epoch 100/100
#     
#     3430/3430 [==============================] - 0s 2us/step - loss: 9.7125







##                   Inspect the model (I)                  ##
# Print the model's weights
print(model.get_weights())
# output:
#     [array([[ 0.13067713, -0.10371894], [ 0.38644195, -0.35632333]], dtype=float32), array([72.38115, 72.38473], dtype=float32)]


# Print the column means of the training data
print(games_tourney_train.mean())

# output:
#     season        1.998074e+03
#     team_1        5.556771e+03
#     team_2        5.556771e+03
#     home          0.000000e+00
#     seed_diff     0.000000e+00
#     score_diff    0.000000e+00
#     score_1       7.162128e+01
#     score_2       7.162128e+01
#     won           5.000000e-01
#     pred         -1.625470e-14
#     dtype: float64







##                   Evaluate the model                  ##
# Evaluate the model on the tournament test data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']], games_tourney_test[['score_1', 'score_2']], verbose = False))

# output: 8.986760239102948