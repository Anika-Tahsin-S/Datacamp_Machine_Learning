# Inspecting
import pandas as pd
# Read in the data
data = pd.read_csv('prices.csv', index_col = 0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())



##                   Many repetitions of sounds                  ##
fig, axs = plt.subplots(3, 2, figsize = (15, 7), sharex = True, sharey = True)

# Calculate the time array
time = np.arange(0, len(normal)) / sfreq

# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T

# Loop through each audio file / ax object and plot
# .T.ravel() transposes the array, then unravels it into a 1-D vector for looping
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
show_plot_and_make_titles()




##                   Invariance in Time                  ##
# Average across the audio files of each DataFrame
mean_normal = np.mean(normal, axis = 1)
mean_abnormal = np.mean(abnormal, axis = 1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 3), sharey = True)
ax1.plot(time, mean_normal)
ax1.set(title = "Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title = "Abnormal Data")
plt.show()



##                   Build a Classification Model                  ##
from sklearn.svm import LinearSVC

# Initialize and fit the model
model = LinearSVC()
model.fit(X_train, y_train)

# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test.squeeze()) / len(y_test))
# output: 0.555555555556