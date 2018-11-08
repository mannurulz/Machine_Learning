# Test Technique 1 #
#from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
#text = ["This is the sample text. This and the are appeared multiple times."]
# create the transform
#vectorizer = CountVectorizer()
# tokenize and build vocab
#vectorizer.fit(text)
# summarize
#print(vectorizer.vocabulary_)
# encode document
#vector = vectorizer.transform(text)
# summarize encoded vector
#print(vector.shape)
#print(type(vector))
#print(vector.toarray())

# Test Technique 2 #
import pandas as pd
import numpy as np
#import xlrd
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text1 = pd.read_excel("C:/Users/mmishra/Downloads/Code/R_n_D/Data/SampleData.xlsx")
data = pd.DataFrame(text1)
#print(data)

# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text1)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform(text1)
# summarize encoded vector
print(vector.shape)
print(vector)
#print(vector.toarray())

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(vector)
print(kmeans.labels_)
print(kmeans.cluster_centers_)