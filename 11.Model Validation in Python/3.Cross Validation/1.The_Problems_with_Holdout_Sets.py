##                   Two Samples                  ##
# Part 1
# Create two different samples of 200 observations 
sample1 = tic_tac_toe.sample(200, random_state = 1111)
sample2 = tic_tac_toe.sample(200, random_state = 1171)

# Part 2
# Print the number of common observations 
print(len([index for index in sample1.index if index in sample2.index]))
# output: 40

# Part 3
# Print the number of observations in the Class column for both samples 
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())
# output:
#     40
#     positive    134
#     negative     66
#     Name: Class, dtype: int64
#     positive    123
#     negative     77
#     Name: Class, dtype: int64




##                   Potential Problems                  ##
# Which of the following statements are TRUE regarding potential problems with holdout samples:

#     A: Using different data splitting methods may lead to varying data in the final holdout samples.
#     B: If you have limited data, your holdout accuracy may be misleading.
#     C: There are no problems. Creating a single train and test sample is the only way to validate models.
#     D: You shouldn't use holdout samples with limited data because you are limiting the potential training data.

# Answer: A & B
# If our models are not generalizing well or if we have limited data, we should be careful using a single training/validation split. 