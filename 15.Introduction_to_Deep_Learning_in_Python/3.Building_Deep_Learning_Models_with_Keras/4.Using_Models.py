# Saving, Reloading and using your model
from keras import load_model
model.save('model_file.h5')
model1 = load_model('model1.h5')
pred = model1.predict(data_to_predict_with)
probability_true = pred[:,1]
model1.summary()




# --------------------------------------------------------------------------------------------------------- #
##                   Making Predictions                  ##
# New data to make predictions is stored in a NumPy array as pred_data. 
import numpy
# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)

# output:
#     Epoch 1/10
    
#  32/800 [>.............................] - ETA: 26s - loss: 5.3679 - acc: 0.3438544/800 
#         [===================>..........] - ETA: 0s - loss: 2.7597 - acc: 0.5699800/800         
#           [==============================] - 1s - loss: 2.3259 - acc: 0.5825     
#    Epoch 2/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 3.0064 - acc: 0.3438448/800 
#         [===============>..............] - ETA: 0s - loss: 1.5986 - acc: 0.5893800/800 
#         [==============================] - 0s - loss: 1.2419 - acc: 0.6088     
#     Epoch 3/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 0.5868 - acc: 0.7500576/800 
#         [====================>.........] - ETA: 0s - loss: 0.7521 - acc: 0.6788800/800 
#         [==============================] - 0s - loss: 0.7359 - acc: 0.6663     
#     Epoch 4/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 0.5872 - acc: 0.6875544/800 
#         [===================>..........] - ETA: 0s - loss: 0.8137 - acc: 0.6801800/800 
#         [==============================] - 0s - loss: 0.7642 - acc: 0.6575     
#     Epoch 5/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 0.5793 - acc: 0.7812512/800 
#         [==================>...........] - ETA: 0s - loss: 0.6055 - acc: 0.7012800/800 
#         [==============================] - 0s - loss: 0.6219 - acc: 0.6850     
#     Epoch 6/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 0.6313 - acc: 0.6875576/800 
#         [====================>.........] - ETA: 0s - loss: 0.6295 - acc: 0.6892800/800 
#         [==============================] - 0s - loss: 0.6423 - acc: 0.6900     
#     Epoch 7/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 0.5596 - acc: 0.7188640/800 
#         [=======================>......] - ETA: 0s - loss: 0.6778 - acc: 0.6719800/800 
#         [==============================] - 0s - loss: 0.6516 - acc: 0.6900     
#     Epoch 8/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 0.6163 - acc: 0.6562672/800 
#         [========================>.....] - ETA: 0s - loss: 0.6494 - acc: 0.6667800/800 
#         [==============================] - 0s - loss: 0.6471 - acc: 0.6687     
#     Epoch 9/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 0.6607 - acc: 0.6562608/800 
#         [=====================>........] - ETA: 0s - loss: 0.6318 - acc: 0.6711800/800 
#         [==============================] - 0s - loss: 0.6257 - acc: 0.6837     
#     Epoch 10/10
    
#  32/800 [>.............................] - ETA: 0s - loss: 0.7275 - acc: 0.5938640/800 
#         [=======================>......] - ETA: 0s - loss: 0.6037 - acc: 0.6922800/800 
#         [==============================] - 0s - loss: 0.6128 - acc: 0.6837    

#     [0.2666508  0.43373662 0.81286263 0.5527454  0.23296797 0.20595169
#      0.1349096  0.3632123  0.21433544 0.5624285  0.25882646 0.32837358
#      0.21516892 0.44281405 0.2148334  0.18588893 0.30253404 0.47543657
#      0.1254332  0.43917575 0.6550583  0.26118502 0.14015315 0.36253673
#      0.4660681  0.21710011 0.5746518  0.6193853  0.22835338 0.5584776
#      0.48270977 0.48246554 0.22026658 0.29262072 0.36464027 0.6735897
#      0.3329101  0.21490133 0.58109105 0.47156096 0.33093986 0.410923
#      0.5038511  0.18834975 0.38575104 0.13447838 0.41894975 0.19238003
#      0.48430887 0.7470138  0.4185302  0.03536972 0.48878586 0.59497565
#      0.28728792 0.41621014 0.9166899  0.2513105  0.4683839  0.22026658
#      0.16530544 0.35302365 0.27502537 0.4424924  0.36241814 0.19305041
#      0.34972182 0.5554562  0.23718026 0.45410004 0.2589697  0.48514014
#      0.18687625 0.11483733 0.46909463 0.4278682  0.37025085 0.34319168
#      0.2127945  0.59108806 0.48987415 0.19187023 0.36710712 0.29073486
#      0.25541103 0.49985334 0.33763576 0.5368187  0.42267123 0.49387228
#      0.21134487]