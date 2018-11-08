# Test Technique 1 #
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["This is the sample text. This and the are appeared multiple times."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
#print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
#print(vector.shape)
#print(type(vector))
#print(vector.toarray())

# Test Technique 2 #
import pandas as pd
#import xlrd
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
#text = pd.read_excel("C:/Users/mmishra/Downloads/Code/R_n_D/Data/SampleData.xlsx")
#data = pd.DataFrame(text)
#print(data)
text =  ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())