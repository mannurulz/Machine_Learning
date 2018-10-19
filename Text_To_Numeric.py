# import numpy as np
# import re
# 
# def tokenize_sentences(sentences):
#     words = []
#     for sentence in sentences:
#         w = extract_words(sentence)
#         words.extend(w)
#         
#     words = sorted(list(set(words)))
#     return words
# 
# def extract_words(sentence):
#     ignore_words = ['a']
#     words = re.sub("[^\w]", " ",  sentence).split() #nltk.word_tokenize(sentence)
#     words_cleaned = [w.lower() for w in words if w not in ignore_words]
#     return words_cleaned    
#     
# def bagofwords(sentence, words):
#     sentence_words = extract_words(sentence)
#      frequency word count
#     bag = np.zeros(len(word s))
#     for sw in sentence_words:
#         for i,word in enumerate(words):
#             if word == sw: 
#                 bag[i] += 1
#                 
#     return np.array(bag)
# 
# sentences = ["Machine learning is great","Natural Language Processing is a complex field","Natural Language Processing is used in machine learning"]
# vocabulary = tokenize_sentences(sentences)
# bagofwords("Machine learning is great", vocabulary)
# 
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
# train_data_features = vectorizer.fit_transform(sentences)
# vectorizer.transform(["Machine learning is great"]).toarray()


''''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
data_corpus = ["Manmohan likes to watch movies. Mary likes movies too.", 
"John also likes to watch football games."]
X = vectorizer.fit_transform(data_corpus) 
print(X.toarray())
print(vectorizer.get_feature_names())
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import timeit
data = pd.read_csv("C:/Users/mmishra/Downloads/Code/R_n_D/Data/SampleData.csv")
print(data)


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)
X_train_counts.shape
print(X_train_counts)
print("****************")
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
print(X_train_tfidf)


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf)#, data.status)
print(clf.predict(X))
'''

#df = pd.DataFrame(data,columns=['Type','Message Status','Severity','Sent','Policy','Matches','Subject','Sender','Recipients','Status','Has Attachment','Contract Type','Department','Country'])
df = pd.DataFrame(data,columns=['Message Status','Policy','Matches','Subject','Sender','Recipients','Status','Has Attachment','Contract Type','Department Code','Department','Country'])
print(df)

#df.convert_objects(convert_numeric=True)
#print(df)
df.fillna(0,inplace=True)
print(df)

#from sklearn.feature_extraction import CountVectorizer



def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
print(df)

X = np.array(df.drop(['Status'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['Status'])


clf = KMeans(n_clusters=2)
clf.fit(X)
clf.predict(X)
labels = clf.labels_
print(clf.score(X))
print(labels)

fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig)#, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)

plt.title("K Means", fontsize=14)
plt.show()
'''