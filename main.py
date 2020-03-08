import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from collections import Counter
import autokeras as ak
import codecs
import random



data = pd.read_csv('train/feats.csv',header=0, sep=',').values
x_train1 = np.delete(data, (0,4), axis= 1)
x_train2 = np.delete(data, (1,2,3), axis=1)
y = x_train2[:,-1]
ids = data[:,0]
images = []
labels = []
x_train = []
for i in range(len(x_train1)):
    path = 'train/images/' + ids[i]
    label = data[i][4]
    if os.listdir(path) == []:
        continue
    j = os.listdir(path)[0]
    images.append(cv2.resize(cv2.imread(path +'/'+ j, 0), (640,640)))
    x_train.append(x_train1[i])
    labels.append(label)

images = np.array(images)
labels = np.array(labels)
x_train = np.array(x_train)


clf = ak.StructuredDataClassifier(num_classes=4)
clf.fit(x_train, labels)


test_data = pd.read_csv('test/feats.csv',header=0, sep=',').values
x_test1 = np.delete(test_data, 0, axis= 1)
ids = test_data[:,0]
y_pred = clf.predict(x_test1)
y_pred = y_pred.reshape(len(y_pred))
result = []
for i in range(len(x_test1)):
    label = y_pred[i]
    result.append([ids[i], label])
with codecs.open('result.csv', 'w', encoding='utf-8') as wf:
    for i in range(len(result)):
        wf.write(result[i][0]+',')
        wf.write(str(result[i][1]))
        wf.write('\n')
print('done')

