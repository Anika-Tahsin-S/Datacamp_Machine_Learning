##                   Cleaning Up Your Text                  ##
# Part 1
# Print the first 5 rows of the text column
print(speech_df['text'].head())
# output:
#     0    Fellow-Citizens of the Senate and of the House...
#     1    Fellow Citizens:  I AM again called upon by th...
#     2    WHEN it was first perceived, in early times, t...
#     3    Friends and Fellow-Citizens:  CALLED upon to u...
#     4    PROCEEDING, fellow-citizens, to that qualifica...
#     Name: text, dtype: object

# Part 2
# Replace all non letter characters with a whitespace
speech_df['text_clean'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ')

# Change to lower case
speech_df['text_clean'] = speech_df['text_clean'].str.lower()

# Print the first 5 rows of the text_clean column
print(speech_df['text_clean'].head())






##                   High Level Text Features                  ##
# Find the length of each text
speech_df['char_cnt'] = speech_df['text_clean'].str.len()

# Count the number of words in each text
speech_df['word_cnt'] = speech_df['text_clean'].str.split().str.len()

# Find the average length of word
speech_df['avg_word_length'] = speech_df['char_cnt'] / speech_df['word_cnt']
# Print the first 5 rows of these columns
print(speech_df[['text_clean', 'char_cnt', 'word_cnt', 'avg_word_length']])
# output:
#                                                text_clean  char_cnt  word_cnt  avg_word_length
#     0   fellow citizens of the senate and of the house...      8616      1432            6.017
#     1   fellow citizens   i am again called upon by th...       787       135            5.830
#     2   when it was first perceived  in early times  t...     13871      2323            5.971
#     3   friends and fellow citizens   called upon to u...     10144      1736            5.843