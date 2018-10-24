'''
Developed by Manmohan Mishra.

1. Read data from XLSX File
2. Plot data based on status (target) column for analysis
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

data = pd.read_excel("C:/Users/mmishra/Desktop/SampleData.xlsx")
df = pd.DataFrame(data, columns=['Policy','matches','Status','Subject','has attachment','VIP','department code'])
df.fillna(0, inplace=True)

columns = df.columns.values

#Source data analysis & Plotting

fig = plt.figure(figsize=(8,6))
df.groupby('Status').Status.count().plot.bar(ylim=0)
plt.show()


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

df = handle_non_numerical_data(df)


# Create Target
Y = df[['Status']].copy()
#Dropping Target variable from feature dataset
X = np.array(df.drop(['Status'],1).astype(float))
