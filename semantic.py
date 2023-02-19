import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# cat is more similar to monkey than banana because they are animals
# monkey is more similar to banana than cat is to banana because monkeys eat bananas
# this is interesting because it knows certain animals eat certain foods making the words more similar

tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# using my own examples of different words to see the differences
tokens = nlp('dog bone wolf banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# running the above example using the language model 'en_core_web_sm' gives the following warning
# The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the
# tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the
# small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors.
# You can always add your own word vectors, or use one of the larger models instead if available.

# the similarity was not as accurate as the md model because it is smaller
