##                   Readability of 'The Myth of Sisyphus'                  ##
f = open("sisyphus_essay.txt"); sisyphus_essay = f.readline(); f.close()

# Import Textatistic
from textatistic import Textatistic

# Compute the readability scores 
readability_scores = Textatistic(sisyphus_essay).scores

# Print the flesch reading ease score
flesch = readability_scores['flesch_score']
print("The Flesch Reading Ease is %.2f" % (flesch))
# output:
#     The Flesch Reading Ease is 81.67




##                   Readability of Various Publications                  ##
f1 = open("forbes.txt"); forbes = f.readline(); f.close()
f2 = open("harvard_law.txt"); harvard_law = f.readline(); f.close()
f3 = open("r_digest.txt"); r_digest = f.readline(); f.close()
f4 = open("time_kids.txt"); time_kids = f.readline(); f.close()


# Import Textatistic
from textatistic import Textatistic

# List of excerpts
excerpts = [forbes, harvard_law, r_digest, time_kids]

# Loop through excerpts and compute gunning fog index
gunning_fog_scores = []
for excerpt in excerpts:
  readability_scores = Textatistic(excerpt).scores
  gunning_fog = readability_scores['gunningfog_score']
  gunning_fog_scores.append(gunning_fog)

# Print the gunning fog indices
print(gunning_fog_scores)
# output:
#     [14.436002482929858, 20.735401069518716, 11.085587583148559, 5.926785009861934]