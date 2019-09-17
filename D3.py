#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.bayes.naive_bayes import NaiveBayes
from skmultiflow.drift_detection.eddm import EDDM
import time
import sys


# In[2]:


def drift_detector(S,T,threshold = 0.75):
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)
    T['in_target'] = 0 # in target set
    S['in_target'] = 1 # in source set
    ST = pd.concat( [T, S], ignore_index=True, axis=0)
    labels = ST['in_target'].values
    ST = ST.drop('in_target', axis=1).values
    clf = LogisticRegression(solver='liblinear')
    predictions = np.zeros(labels.shape)
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(ST, labels):
        X_train, X_test = ST[train_idx], ST[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    auc_score = AUC(labels, predictions)
    if auc_score > threshold:
        return True
    else:
        return False


# In[5]:


class D3():
    def __init__(self, w, rho, dim, auc):
        self.size = int(w*(1+rho))
        self.win_data = np.zeros((self.size,dim))
        self.win_label = np.zeros(self.size)
        self.w = w
        self.rho = rho
        self.dim = dim
        self.auc = auc
        self.drift_count = 0
        self.window_index = 0
    def addInstance(self,X,y):
        if(self.isEmpty()):
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.window_index = self.window_index + 1
        else:
            print("Error: Buffer is full!")
    def isEmpty(self):
        return self.window_index < self.size
    def driftCheck(self):
        if drift_detector(self.win_data[:self.w], self.win_data[self.w:self.size], auc): #returns true if drift is detected
            self.window_index = int(self.w * self.rho)
            self.win_data = np.roll(self.win_data, -1*self.w, axis=0)
            self.win_label = np.roll(self.win_label, -1*self.w, axis=0)
            self.drift_count = self.drift_count + 1
            return True
        else:
            self.window_index = self.w
            self.win_data = np.roll(self.win_data, -1*(int(self.w*self.rho)), axis=0)
            self.win_label =np.roll(self.win_label, -1*(int(self.w*self.rho)), axis=0)
            return False
    def getCurrentData(self):
        return self.win_data[:self.window_index]
    def getCurrentLabels(self):
        return self.win_label[:self.window_index]


# In[3]:


def select_data(x):
    df = pd.read_csv(x)
    scaler = MinMaxScaler()
    df.iloc[:,0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:,0:df.shape[1]-1])
    return df


# In[4]:


def check_true(y,y_hat):
    if(y==y_hat):
        return 1
    else:
        return 0


# In[6]:


df = select_data(sys.argv[1])
stream = DataStream(df)
stream.prepare_for_use()
stream_clf = HoeffdingTree()
w = int(sys.argv[2])
rho = float(sys.argv[3])
auc = float(sys.argv[4])


# In[ ]:


D3_win = D3(w,rho,stream.n_features,auc)
stream_acc = []
stream_record = []
stream_true= 0

i=0
start = time.time()
X,y = stream.next_sample(int(w*rho))
stream_clf.partial_fit(X,y, classes=stream.target_values)
while(stream.has_more_samples()):
    X,y = stream.next_sample()
    if D3_win.isEmpty():
        D3_win.addInstance(X,y)
        y_hat = stream_clf.predict(X)
        stream_true = stream_true + check_true(y, y_hat)
        stream_clf.partial_fit(X,y)
        stream_acc.append(stream_true / (i+1))
        stream_record.append(check_true(y, y_hat))
    else:
        if D3_win.driftCheck():             #detected
            #print("concept drift detected at {}".format(i))
            #retrain the model
            stream_clf.reset()
            stream_clf.partial_fit(D3_win.getCurrentData(), D3_win.getCurrentLabels(), classes=stream.target_values)
            #evaluate and update the model
            y_hat = stream_clf.predict(X)
            stream_true = stream_true + check_true(y, y_hat)
            stream_clf.partial_fit(X,y)
            stream_acc.append(stream_true / (i+1))
            stream_record.append(check_true(y, y_hat))
            #add new sample to the window
            D3_win.addInstance(X,y)
        else:
            #evaluate and update the model
            y_hat = stream_clf.predict(X)
            stream_true = stream_true + check_true(y, y_hat)
            stream_clf.partial_fit(X,y)
            stream_acc.append(stream_true / (i+1))
            stream_record.append(check_true(y, y_hat))
            #add new sample to the window
            D3_win.addInstance(X,y)
    i = i+1  

elapsed = format(time.time() - start, '.4f')
acc = format((stream_acc[-1]*100), '.4f')
final_accuracy = "Final accuracy: {}, Elapsed time: {}".format(acc,elapsed)
print(final_accuracy)


# In[7]:


def window_average(x,N):
    low_index = 0
    high_index = low_index + N
    w_avg = []
    while(high_index<len(x)):
        temp = sum(x[low_index:high_index])/N
        w_avg.append(temp)
        low_index = low_index + N
        high_index = high_index + N
    return w_avg


# In[8]:


a=int(len(df)/30)
ddd_acc2 = window_average(stream_record, a)


# In[ ]:


x = np.linspace(0, 100, len(ddd_acc2), endpoint=True)

f = plt.figure()
plt.plot(x, ddd_acc2, 'r', label='D3', marker="*") 

plt.xlabel('Percentage of data', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(True)
plt.legend(loc='lower left')

plt.show()

f.savefig("preq.pdf", bbox_inches='tight')


# In[ ]:




