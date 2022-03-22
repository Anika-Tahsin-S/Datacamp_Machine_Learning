from keras.layers import Input, Dense

input_tensor_1 = Input((1,))
input_tensor_2 = Input((1,))

shared_layer = Dense(1)
output_tensor_1 = shared_layer(input_tensor_1)
output_tensor_2 = shared_layer(input_tensor_2)

# Sharing Multiple layers as model
from keras.layers import Embedding

input_tensor = Input(shape = (1,))
n_teams = 10887
embed_layer = Embedding(input_dim = n_teams, input_length = 1, output_dim = 1, name = 'Team-Strength-Lookup')
embed_tensor = embed_layer(input_tensor)

flatten_tensor = Flatten()(embed_tensor)
model = Model(input_tensor, flatten_tensor)

input_tensor_1 = Input((1,))
input_tensor_2 = Input((1,))

output_tensor_1 = model(input_tensor_1)
output_tensor_2 = model(input_tensor_2)









# --------------------------------------------------------------------------------------------------------- #
##                   Defining two inputs                  ##
# Load the input layer from keras.layers
from keras.layers import Input

# Input layer for team 1
team_in_1 = Input((1,), name = "Team-1-In")

# Separate input layer for team 2
team_in_2 = Input((1,), name = "Team-2-In")






##                   Lookup both inputs in the same model                  ##
# Lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# Lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)