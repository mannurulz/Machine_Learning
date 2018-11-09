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
import matplotlib.pyplot as plt
# list of text documents
text1 = pd.read_excel("C:/Users/mmishra/Downloads/Code/R_n_D/Data/SampleData.xlsx")
data = pd.DataFrame(text1)
#print(data)

# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text1)
# summarize
#print(vectorizer.vocabulary_)
#print(vectorizer.idf_)
# encode document
vector = vectorizer.transform(text1)
# summarize encoded vector
print(vector.shape)
print(vector)
#print(vector.toarray())

#from sklearn.ensemble import RandomForestClassifier

#rfc = RandomForestClassifier()
#rfc.fit(vector)
#print(rfc.score)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(vector)
print(kmeans.labels_)
K_labevector= kmeans.labels_
print(vector.groupby(K_label).K_label.count())
#print(kmeans.cluster_centers_)
y_kmeans = kmeans.predict(vector)
plt.scatter(vector[:, 0], vector[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
