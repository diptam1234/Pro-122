import pandas as pd
import cv2
import seaborn as snb
import numpy as nm
import matplotlib.pyplot as mp

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")
print( pd.Series(data).value_counts() )

classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

sample_per_class = 5 

figure = mp.figure( figsize=(nclasses*2, (1+sample_per_class*2)) )

idx_cls = 0

for cls in classes:
    idxs = nm.flatnonzero(y == cls)
    idxs = nm.random.choice(idxs, sample_per_class, replace=False)
    i = 0
    for idx in idxs:
        plt_idx = i * nclasses + idx_cls + 1
        p = mp.subplot(sample_per_class, nclasses, plt_idx)
        p = mp.heatmap(nm.array(X.loc[idx]).reshape(28,28), cmap=mp.cm.gray ,
                xticklabels=False, yticklabels=False, cbar=False)
        p = mp.axis('off')
        i += 1
    idx_cls += 1

idxs = nm.flatnonzero(y == '0')
print(nm.array(X.loc[idxs[0]]))

X_train , X_test , y_train , y_test = train_test_split(X,y , random_state=9 , test_size = 2500 , train_size = 7500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

lr = LogisticRegression(solver = "saga" , multi_class = 'multinominal').fit(X_train_scaled, y_train)

y_predict = lr.predict(X_test_scaled)

accuracy = accuracy_score(y_test,y_predict)

print("Accuracy--->",accuracy)
 
cm = pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted'])

p = mp.figure(figsize=(10,10))
p = snb.heatmap(cm, annot=True, fmt="d", cbar=False)
