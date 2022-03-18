# Stochastic Gradient Descent
def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_shape = (n_cols,)))
    model.add(Dense(2, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))
    return(model)

lr_to_test = [0.000001, 0.01, 1]

# loop for learning rate
for lr in lr_to_test:
    model = get_new_model()
    optimizer = SGD(lr = lr)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy')
    model.fit(pred, target)





# --------------------------------------------------------------------------------------------------------- #
##                   Diagnosing optimization problems                  ##
# Which of the following could prevent a model from showing an improved loss in its first few epochs?
# Learning rate too low.
# Learning rate too high.
# Poor choice of activation function.
# All of the above.


# Answer: All of the above
# All the options listed could prevent a model from showing an improved loss in its first few epochs.






##                   Changing optimization parameters                  ##
# IPython  Shell
# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[1:]

# print predicted_prob_true
print(predicted_prob_true)
# ...................................... #


# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr = lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)

# output:   
#     Testing model with learning rate: 0.000001
    
#     Epoch 1/10
    
#  32/891 [>.............................] - ETA: 34s - loss: 3.6053480/891 
#         [===============>..............] - ETA: 1s - loss: 3.6363891/891 
#         [==============================] - 1s - loss: 3.6057     

    
    
#     Testing model with learning rate: 0.010000
    
#     Epoch 1/10
    
#  32/891 [>.............................] - ETA: 38s - loss: 1.0910448/891 
#         [==============>...............] - ETA: 1s - loss: 2.1088891/891 
#         [==============================] - 1s - loss: 1.4059     

    
    
#     Testing model with learning rate: 1.000000
    
#     Epoch 1/10
    
#  32/891 [>.............................] - ETA: 34s - loss: 1.0273480/891 
#         [===============>..............] - ETA: 1s - loss: 5.5083891/891 
#         [==============================] - 1s - loss: 5.9885     