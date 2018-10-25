'''
Developed by Manmohan Mishra.

1. Read data from XLSX File
2. Plot data based on status (target) column for analysis
    a.  Data is not balanced - then upsample/resample for minority class to avoid biased model training
3. Clean Data
4. Preprocessing & Split for Training & Test
5. Initiate Model, Fit (train)
6. Predict with test data
7. Accuracy Validation & Testing
8. Plot Confusion Matrix
9. Export ML Model to file
10. Use ML Model to predict with actual data

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

import itertools
from sklearn.metrics import confusion_matrix
from pandas.tests.groupby.test_function import test_size
from sklearn.cluster.tests.test_k_means import n_samples

data = pd.read_excel("C:/Users/mmishra/Desktop/SampleData.xlsx")
df = pd.DataFrame(data, columns=['Policy','matches','Status','Subject','has attachment','VIP','department code'])
df.fillna(0, inplace=True)

columns = df.columns.values

#Source data analysis & Plotting

fig = plt.figure(figsize=(8,6))
df.groupby('Status').Status.count().plot.bar(ylim=0)
plt.show()


#Re-Sampling the data as classed are not balanced
from sklearn.utils import resample

df_esc = df[df.Status == 'Escalated']
df_dis = df[df.Status == 'Dismissed']

# Escalated class belong to minority, as the number of alerts escalated are less.
# n_samples should be equal to majority class - this is to have proper balance data
df_esc_enh = resample(df_esc, replace=True, n_samples=60000, random_state=123)

df_balanced = pd.concat([df_dis, df_esc_enh])


# Preprocessing
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

df = handle_non_numerical_data(df_balanced)


# Create Target
Y = df[['Status']].copy()
#Dropping Target variable from feature dataset
X = np.array(df.drop(['Status'],1).astype(float))

print(X.shape, Y.shape)

# Split data for training & testing 
x_tr, x_te, y_tr, y_te = train_test_split(X,Y, test_size=0.2)

RFC = RandomForestClassifier()
RFC.fit(x_tr,y_tr)

print(RFC.feature_importances_)
Yp = RFC.predict(x_te)
RFC.score(x_te, y_te)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


'''
Model Export in file for direct usage
'''

import pickle

model_file = 'RFC.pickle'
pickle.dump(RFC, open(model_file,'wb'))


'''
Using model from file direct on new test data
Note: Model is not trained or tested using this data. This will be new prediction.
'''

TData = pd.read_excel("My_Prod_Test_File")
Tdf = pd.DataFrame(TData, columns=['Policy','matches','Status','Subject','has attachment','VIP','department code'])
Tdf.fillna(0, inplace=True)

Tdf = handle_non_numerical_data(Tdf)
TX = np.array(Tdf.drop(['Status'],1).astype(float))
print(TX)


my_model = pickle.load(open(model_file,'rb'))
my_model.predict(TX)
