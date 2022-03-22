# Embedding Layer
from keras.layers import Embedding
input_tensor = Input(shape = (1,))
n_teams = 10887
embed_layer = Embedding(input_dim = n_teams, input_length = 1, output_dim = 1, name = 'Team-Strength-Lookup')
embed_tensor = embed_layer(input_tensor)

# Flattening
from keras.layers import Flatten
flatten_tensor = Flatten()(embed_tensor)
#  Flatten layers are an advanced layer for deep learning models and can be used to transform data from multiple dimensions back down to two dimensions. 
# They are useful for dealing with time series data, text data, and images. 

# All together
input_tensor = Input(shape = (1,))
n_teams = 10887
embed_layer = Embedding(input_dim = n_teams, input_length = 1, output_dim = 1, name = 'Team-Strength-Lookup')
embed_tensor = embed_layer(input_tensor)

flatten_tensor = Flatten()(embed_tensor)
model = Model(input_tensor, flatten_tensor)








# --------------------------------------------------------------------------------------------------------- #
##                   Define team lookup                  ##
# Imports
from keras.layers import Embedding
from numpy import unique

# Count the unique number of teams
n_teams = unique(games_season['team_1']).shape[0]

# Create an embedding layer
team_lookup = Embedding(input_dim = n_teams,
                        output_dim = 1,
                        input_length = 1,
                        name = 'Team-Strength')





##                   Define team model                  ##
# Imports
from keras.layers import Input, Embedding, Flatten
from keras.models import Model

# Create an input layer for the team ID
teamid_in = Input(shape = (1,))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name = 'Team-Strength-Model')