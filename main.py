import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from collections import Counter
import autokeras as ak
import codecs


data = pd.read_csv('train/feats.csv',header=0, sep=',').values
x_train = np.delete(data, (0,4), axis= 1)
y = data[:,-1]
ids = data[:,0]
x_train = np.array(x_train)
y = np.array(y)
clf = ak.StructuredDataClassifier(num_classes=4,objective='val_accuracy',max_trials=50)
clf.fit(x_train, y)


test_data = pd.read_csv('test/feats.csv',header=0, sep=',').values
x_test = np.delete(test_data, 0, axis= 1)
predicted_y = clf.predict(x_test)
predicted_y = predicted_y.reshape((87))
ids = test_data[:,0]
result = []
for i in range(len(x_test)):
    label = predicted_y[i]
    result.append([ids[i], label])
c = Counter(predicted_y)

with codecs.open('result.csv', 'w', encoding='utf-8') as wf:
    for i in range(len(result)):
        wf.write(result[i][0]+',')
        wf.write(str(result[i][1]))
        wf.write('\n')
print('done')

