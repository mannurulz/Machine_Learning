import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pickle
#import timeit
data = pd.read_csv("C:/Users/mmishra/Downloads/Code/R_n_D/Data/SampleData.csv")
df = pd.DataFrame(data,columns=['Type','Message Status','Severity','Sent','Policy','Matches','Subject','Sender','Recipients','Status','Has Attachment','Contract Type','Department','Country'])

print(df.head())

#df.convert_objects(convert_numeric=True)
#print(df)
df.fillna(0,inplace=True)
print(df)

from sklearn.feature_extraction import CountVectorizer

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
print(df.head())

X = np.array(df.drop(['Status'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['Status'])

#print(timeit.timeit())
clf = KMeans(n_clusters=2)
clf.fit(X)
#print(timeit.timeit())
# Saving model in file
model_file = 'KMeans_Model.pkl'
pickle.dump(clf, open(model_file,'wb'))


# Load Model from file
loaded_model = pickle.load(open(model_file,'rb'))
#result = loaded_model.score(X,y)
#print(result)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = loaded_model.predict(predict_me)
    print(prediction)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))



#import matplotlib.pyplot as plt

 

#plt.plot(X)
#plt.plot(y, X['Severity'],color="k")
#plt.show()