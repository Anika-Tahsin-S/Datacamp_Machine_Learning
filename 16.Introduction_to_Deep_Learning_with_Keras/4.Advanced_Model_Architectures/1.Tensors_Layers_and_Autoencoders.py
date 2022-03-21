## Accessing Keras Layer ##
first_layer = model.layers[0]
print(first_layer.input)
print(first_layer.output)
print(first_layer.weights)


# Import Keras Backend
import keras.backend as K
inp = model.layers[0].input
out = model.layers[0].output

inp_to_out = K.function([inp], [out])
print(inp_to_out)


##                   New Architecture: Autoencoders                   ##
# Building a simple autoencoder
autoencoder = Sequential()
autoencoder.add(Dense(4, input_shape = (100,), activation = 'relu'))
autoencoder.add(Dense(100, activation = 'sigmoid'))
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# Breaking it into an encoder
encoder = Sequential()
autoencoder.add(autoencoder.layers[0])
encoder.predict(X_test)







# --------------------------------------------------------------------------------------------------------- #
##                   It's a flow of tensors                  ##
# Import keras backend
import keras.backend as K

# Input tensor from the 1st layer of the model
inp = model.layers[0].input

# Output tensor from the 1st layer of the model
out = model.layers[0].output

# Define a function from inputs to outputs
inp_to_out = K.function([inp], [out])

# Print the results of passing X_test through the 1st layer
print(inp_to_out([X_test]))

# output:
#     [array([[7.77682841e-01, 0.00000000e+00],
#            [0.00000000e+00, 0.00000000e+00],
#   ........................................
#            [0.00000000e+00, 1.50589132e+00],
#            [1.23799539e+00, 1.65470392e-01],
#            [0.00000000e+00, 1.54111814e+00]], dtype=float32)]







##                   Neural separation                  ##
# IPython Shell
# Creates a model given an activation and learning rate
def create_model(learning_rate, activation):
  
  	# Create an Adam optimizer with the given learning rate
  	opt = Adam(lr = learning_rate)
  	
  	# Create your binary classification model  
  	model = Sequential()
  	model.add(Dense(128, input_shape = (30,), activation = activation))
  	model.add(Dense(256, activation = activation))
  	model.add(Dense(1, activation = 'sigmoid'))
  	
  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
  	return model

# Plot
def plot():
  fig, ax = plt.subplots()
  plt.scatter(layer_output[:, 0], layer_output[:, 1],c = y_test,edgecolors='none')
  plt.title('Epoch: {}, Test Accuracy: {:3.1f} %'.format(i+1, test_accuracy * 100.0))
  plt.show()
# -----------------------------------------------------------------

for i in range(0, 21):
  	# Train model for 1 epoch
    h = model.fit(X_train, y_train, batch_size = 16, epochs = 1, verbose = 0)
    if i%4 == 0: 
      # Get the output of the first layer
      layer_output = inp_to_out([X_test])[0]
      
      # Evaluate model accuracy for this epoch
      test_accuracy = model.evaluate(X_test, y_test)[1] 
      
      # Plot 1st vs 2nd neuron output
      plot()


# If you take a look at the graphs you can see how the neurons are learning to spread out the inputs based on whether they are fake or legit dollar bills. (A single fake dollar bill is represented as a purple dot in the graph) At the start the outputs are closer to each other, the weights are learned as epochs go by so that fake and legit dollar bills get a different, further and further apart output. Click in between the graphs fast, it's like a movie!






##                   Building an autoencoder                  ##
# Start with a sequential model
autoencoder = Sequential()

# Add a dense layer with input the original image pixels and neurons the encoded representation
autoencoder.add(Dense(32, input_shape = (784, ), activation = "relu"))

# Add an output layer with as many neurons as the orginal image pixels
autoencoder.add(Dense(784, activation = "sigmoid"))

# Compile your model with adadelta
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

# Summarize your model structure
autoencoder.summary()

# output:
#     Model: "sequential_1"
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     dense_1 (Dense)              (None, 32)                25120     
#     _________________________________________________________________
#     dense_2 (Dense)              (None, 784)               25872     
#     =================================================================
#     Total params: 50,992
#     Trainable params: 50,992
#     Non-trainable params: 0
#     _________________________________________________________________








##                   De-noising like an autoencoder                  ##
# Part 1
# Build your encoder by using the first layer of your autoencoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the noisy images and show the encodings for your favorite number [0-9]
encodings = encoder.predict(X_test_noise)
show_encodings(encodings, number = 1)


# Part 2
# Build your encoder by using the first layer of your autoencoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the noisy images and show the encodings for your favorite number [0-9]
encodings = encoder.predict(X_test_noise)
show_encodings(encodings, number = 1)

# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(X_test_noise)

# Plot noisy vs decoded images
compare_plot(X_test_noise, decoded_imgs)

# Amazing! The noise is gone now! You could get a better reconstruction by using a convolutional autoencoder.