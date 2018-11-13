'''
Developed by Manmohan Mishra.

1. Read data from XLSX File
2. Plot data based on status (target) column for analysis
    a.  Data is not balanced - then upsample/resample for minority class to avoid biased model training
3. Clean Data
4. Preprocessing & Split for Training & Test
5. Initiate Model, Fit (train, target)
6. Predict with test data
7. Accuracy Validation & Testing
8. Plot Confusion Matrix
9. Export ML Model to file
10. Use ML Model to predict with actual data
11. Check for ML Model - Learning Curve


Additional

1. Fitting Logestic Regression & Logestic Regression CV
2. Plot learning curve for LR & LR-CV
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
import os


path = r'path_of_folder'
xl_files = os.listdir(path)
df = pd.DataFrame()

for x in xl_files:
    data = pd.read_excel(path+x)
    data_df = pd.DataFrame(data, columns=['Col1','Col2','Status','Col3','Col4','Col5','Col6'])
    df = df.append(data_df, sort=True)

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
# Predict Target variable value based on Test Data - x_te
Yp = RFC.predict(x_te)

# Print prediction probability (from both the classes A & B) along with each prediction.
Ypp = RFC.predict_proba(x_te)

for x in range(len(x_te)):
    print(Yp[x], Ypp[x])

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



# Model Export in file, can be posted in UAT or PROD for use.

import pickle
model_file = 'RFC.pickle'
pickle.dump(RFC, open(model_file,'wb'))



# Import model from file and perform prediction.
# Note: Model has not been trained or tested using below data. This will be a new prediction.

TData = pd.read_excel("My_Prod_Test_File")
Tdf = pd.DataFrame(TData, columns=['Col1','Col2','Status','Col3','Col4','Col5','Col6'])
Tdf.fillna(0, inplace=True)

Tdf = handle_non_numerical_data(Tdf)
TX = np.array(Tdf.drop(['Status'],1).astype(float))
print(TX)

# Model will be loaded from the file for prediction
my_model = pickle.load(open(model_file,'rb'))
my_model.predict(TX)


# Check ML Model - Learning Curve

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, Y, ylim=None, cv=None, n_jobs=None, train_size=np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Samples")
    plt.ylabel("Äccuracy / Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, Y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color ="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross Validation Score")
    plt.legend(loc='best')
    return plt

title = "Learning Curve - RFC Model"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = RandomForestClassifier()
plot_learning_curve(estimator, title, X, Y, (0.7, 1.01), cv=cv, n_jobs=4)
plt.show()


## Extending code for Logestic Regression
from sklearn.linear_model import LogesticRegression
from sklearn.linear_model import LogesticRegressionCV

lr = LogesticRegression(solver='lbfgs')
lr_cv = LogesticRegressionCV(cv=5, multi_class='multinomial')
lr.fit(x_tr, y_tr)
lr_yp = lr.predict(x_te)
print(lr.score(x_te, y_te))

## Extending code for Logestic Regression CV
lr_cv = LogesticRegressionCV(cv=5, multi_class='multinomial')
lr_cv.fit(x_tr, y_tr)
lr_yp = lr_cv.predict(x_te)
print(lr_cv.score(x_te, y_te))

## Plotting learning curve for Linear Regression  
title = "Learning Curve - Linear Regression Model"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# Calling the model for regression with multinomial classification
estimator = LogesticRegression(multi_class='multinomial')
plot_learning_curve(estimator, title, X, Y, (0.4, 1.01), cv=cv, n_jobs=4)
plt.show()


## Plotting learning curve for Linear Regression  
title = "Learning Curve - Linear Regression Model"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# Suffling the data and equally splitting for ploting the graph
estimator = LogesticRegressionCV(cv=5, multi_class='multinomial')
plot_learning_curve(estimator, title, X, Y, (0.4, 1.01), cv=cv, n_jobs=4)
plt.show()