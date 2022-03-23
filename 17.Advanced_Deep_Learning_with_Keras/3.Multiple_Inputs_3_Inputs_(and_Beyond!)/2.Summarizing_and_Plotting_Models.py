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







##                   Model summaries                  ##
# Part 1
# How many total parameters does this model have?
# Answer : 10,892


# Part 2
# How many trainable parameters does this model have?
# Answer : 10,892


# Part 3
# Which layer of your model has the most trainable parameters?
# Answer : Team-Strength (Model)








##                   Plotting models                  ##
# Part 1
# Imports
import matplotlib.pyplot as plt
from keras.utils import plot_model

# Plot the model
plot_model(model, to_file = 'model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


# Part 2
# How many inputs does this model have?
# Answer : 3


# Part 3
# How many outputs does this model have?
# Answer : 1


# Part 4
# Which layer is shared between 2 inputs?
# Answer : Team-Strength