'''
Created on 2019年7月3日

@author: ch
'''
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data

namelist = ["George", "Michael", "Sam", "Wendy", "Rose", "Vicky"]
# namelist = ["George"]
for i in range(len(namelist)):
    name = namelist[i]
    dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/' + name + '-bid-seq-train.csv', header=0, index_col=0)
    scaler = MinMaxScaler(feature_range=(0,1))
    X, y = get_data(dataset, scaler, fit=True)
    with open('/Users/ch/git/tf/resources/bp/minmax/' + name + '-minmax.pk', 'wb') as fid:
        pickle.dump(scaler, fid)
    clf = MLPClassifier(solver='adam', max_iter=1500, alpha=1e-5,
                        activation='tanh',
                        verbose=10,
                        learning_rate_init=.1,
                        tol=1e-4,
                        early_stopping=False,
                        hidden_layer_sizes=(5,5), random_state=1)
    clf.fit(X, y)
    with open('/Users/ch/git/tf/resources/bp/model/' + name + '.model', 'wb') as fid:
        pickle.dump(clf, fid)
