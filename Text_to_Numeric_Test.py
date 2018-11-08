# Test Technique 1 #
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["This is the sample text. This and the are appeared multiple times."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

print(vectorizer.vocabulary_)