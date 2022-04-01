##                  Transform the data                  ##
# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)


##                  Split the data                  ##
# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])


# =============================================================== #