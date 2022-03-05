import polyglot.text import Text

##                   French NER with Polyglot I                  ##

# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)
    
# Print the type of ent
print(type(ent))


# output:
#     ['Charles', 'Cuvelliez']
#     ['Charles', 'Cuvelliez']
#     ['Bruxelles']
#     ['l’IA']
#     ['Julien', 'Maldonato']
#     ['Deloitte']
#     ['Ethiquement']
#     ['l’IA']
#     ['.']
#     <class 'polyglot.text.Chunk'>






##                   French NER with Polyglot II                  ##
# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
print(entities)
# output:
#     [('I-PER', 'Charles Cuvelliez'), ('I-PER', 'Charles Cuvelliez'), ('I-ORG', 'Bruxelles'), ('I-PER', 'l’IA'), ('I-PER', 'Julien Maldonato'), ('I-ORG', 'Deloitte'), ('I-PER', 'Ethiquement'), ('I-LOC', 'l’IA'), ('I-PER', '.')]







##                   Spanish NER with Polyglot                  ##
# Initialize the count variable: count
count = 0

# Iterate over all the entities
for ent in txt.entities:
    # Check whether the entity contains 'Márquez' or 'Gabo'
    if 'Márquez' in ent or 'Gabo' in ent:
        # Increment count
        count += 1

# Print count
print(count)

# Calculate the percentage of entities that refer to "Gabo": percentage
percentage = count / len(txt.entities)
print(percentage)
# output:
#     29
#     0.29591836734693877