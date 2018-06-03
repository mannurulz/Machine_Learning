import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

df = pd.read_csv('/Users/mannu/Desktop/INF553/fwdassignmenthelp/sample_data.txt',sep=',')

#Make a copy of DF
df_tr = df
#print(df_tr)
#Transsform the timeOfDay to dummies
#df_tr = pd.get_dummies(df_tr, columns=['timeOfDay'])

#Standardize
clmns = ['Col1', 'Col2']
df_tr_std = stats.zscore(df_tr[clmns])
print(df_tr_std)
#Cluster the data
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_tr_std)
labels = kmeans.labels_

#Glue back to originaal data
df_tr['clusters'] = labels

#Add the column into our list
clmns.extend(['clusters'])

#Lets analyze the clusters
print df_tr[clmns].groupby(['clusters']).mean()

'''