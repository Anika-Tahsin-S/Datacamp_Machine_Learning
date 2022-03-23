# Build a simple regressor/classifier
from keras.layers import Input, Dense

in_tensor = Input(shape = (1,))
out_tensor_reg = Dense(1)(in_tensor)
out_tensor_class = Dense(1, activation = 'sigmoid')(out_tensor_reg)

# Create the Model
from keras.model import Model
model = Model(in_tensor, [out_tensor_reg, out_tensor_class])
model.compile(optimizer = 'adam', loss = ['mean_absolute_error', 'binary_crossentropy'])

# Fit the combination classifier/regressor
X = games_tourney_train[['seed_diff']]
y_reg = games_tourney_train[['score_diff']]
y_class = games_tourney_train[['won']]

model.fit(X, [y_reg, y_class], epochs = 100)

model.get_weights()

# You can manually calculate the final layer in the model for some example data, to get an understanding of how the model has learned to relate score difference to win probabilities.
# First, multiply 1 by the weight for the final layer in the model: 0.14. Add the bias for the final layer: 0.0007.
from scipy.special import expit as sigmoid
print(sigmoid(1 * 0.13870609 + 0.00734114))

# Evaluate
X = games_tourney_test[['seed_diff']]
y_reg = games_tourney_test[['score_diff']]
y_class = games_tourney_test[['won']]

model.evaluate(X, [y_reg, y_class])









# --------------------------------------------------------------------------------------------------------- #
##                   Classification and regression in one model                  ##
# Create an input layer with 2 columns
input_tensor = Input(shape = (2,))

# Create the first output
output_tensor_1 = Dense(1, activation = 'linear', use_bias = False)(input_tensor)

# Create the second output (use the first output as input here)
output_tensor_2 = Dense(1, activation = 'sigmoid', use_bias = False)(output_tensor_1)
# Create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1, output_tensor_2])





##                   Compile and fit the model                  ##
# Import the Adam optimizer
from keras.optimizers import Adam

# Compile the model with 2 losses and the Adam optimzer with a higher learning rate
model.compile(loss = ['mean_absolute_error', 'binary_crossentropy'], optimizer = Adam(lr = 0.01))

# Fit the model to the tournament training data, with 2 inputs and 2 outputs
model.fit(games_tourney_train[['seed_diff', 'pred']],
          [games_tourney_train[['score_diff']], games_tourney_train[['won']]],
          epochs = 10,
          verbose = True,
          batch_size = 16384)

# output:
#     Epoch 1/10
#     
#     3430/3430 [==============================] - 0s 47us/step - loss: 9.5718 - dense_1_loss: 8.9779 - dense_2_loss: 0.5939
# ....................................................................................................
#     Epoch 10/10
#     
#     3430/3430 [==============================] - 0s 1us/step - loss: 9.4562 - dense_1_loss: 8.9190 - dense_2_loss: 0.5372







##                   Inspect the model (II)                  ##
# Part 1
# Print the model weights
print(model.get_weights())
# output:
#     [array([[0.9695341 ], [0.22126554]], dtype=float32), 
#     array([[0.1428871]], dtype=float32)]
    

# Print the training data means
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




# Part 2
# Import the sigmoid function from scipy
from scipy.special import expit as sigmoid

# Weight from the model
weight = 0.14

# Print the approximate win probability predicted close game
print(sigmoid(1 * weight))

# Print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))

# output:
#     0.5349429451582145
#     0.8021838885585818

# So sigmoid(1 * 0.14) is 0.53, which represents a pretty close game and sigmoid(10 * 0.14) is 0.80, which represents a pretty likely win. 
# In other words, if the model predicts a win of 1 point, it is less sure of the win than if it predicts 10 points.








##                   Evaluate on new data with two metrics                  ##
# Evaluate the model on new data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']],
               [games_tourney_test[['score_diff']], games_tourney_test[['won']]], verbose = False))

# output: [9.685116468970456, 9.11585681592647, 0.5692595753503676]