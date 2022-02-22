##                   Calculating the Envelope of Sound                  ##
# Part 1
# Plot the raw data first
audio.plot(figsize = (10, 5))
plt.show()

# Part 2
# Rectify the audio signal
audio_rectified = audio.apply(np.abs)

# Plot the result
audio_rectified.plot(figsize = (10, 5))
plt.show()

# Part 3
# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(50).mean()

# Plot the result
audio_rectified_smooth.plot(figsize = (10, 5))
plt.show()



##                   Calculating Features From the Envelope                  ##
import pandas as pd

# Calculate stats
means = np.mean(audio_rectified_smooth, axis=0)
stds = np.std(audio_rectified_smooth, axis=0)
maxs = np.max(audio_rectified_smooth, axis=0)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(model, X, y, cv = 5)
print(np.mean(percent_score))
# output: 0.716666666667



##                   Derivative Features: The Tempogram                  ##
import librosa as lr

# Part 1
# Calculate the tempo of the sounds
tempos = []
for col, i_audio in audio.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr = sfreq, hop_length = 2**6, aggregate = None))

# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)

# Calculate statistics of each tempo
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)


# Part 2
# Create the X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv = 5)
print(np.mean(percent_score))
# Output: 0.533333333333