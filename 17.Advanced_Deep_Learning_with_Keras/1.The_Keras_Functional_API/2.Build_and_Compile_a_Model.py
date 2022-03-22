# Keras Models
from keras.layers import Input, Dense

input_tensor = Input(shape = (1,))
output_tensor = Dense(1)(input_tensor)

# Building Models
from keras.models import Model
model = Model(input_tensor, output_tensor)

# Compile a model
model.compile(optimizer = 'adam', loss = 'mae')

model.summary()

# Plotting
from keras.utils import plot_model
plot_model(model, to_file = 'model.png')

from matplotlib import pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()








# --------------------------------------------------------------------------------------------------------- #
##                   Build a model                  ##
# Input/dense/output layers
from keras.layers import Input, Dense
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)

# Build the model
from keras.models import Model
model = Model(input_tensor, output_tensor)





##                   Build a model                  ##
# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')



##                   Visualize a model                  ##
# Import the plotting function
from keras.utils import plot_model
import matplotlib.pyplot as plt

# Summarize the model
model.summary()

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


# Sumarry output:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Input-Layer (InputLayer)     (None, 1)                 0         
_________________________________________________________________
Output-Layer (Dense)         (None, 1)                 2         
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________