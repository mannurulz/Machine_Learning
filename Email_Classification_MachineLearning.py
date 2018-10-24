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

#Source data analysis

fig = plt.figure(figsize=(8,6))
df.groupby('Status').Status.count().plot.bar(ylim=0)
plt.show()