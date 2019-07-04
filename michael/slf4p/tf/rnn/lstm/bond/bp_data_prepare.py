'''
Created on 2019年7月3日

@author: ch
'''
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import split_data

namelist = ["Michael", "Wendy", "Vicky", "Sam", "George", "Rose"]
for i in namelist:
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/bp/data/train-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/bp/data/' + i + '-bid-seq-train.csv'
    split_data(filepath, topath)
    
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/bp/data/val-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/bp/data/' + i + '-bid-seq-val.csv'
    split_data(filepath, topath)
    
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/bp/data/test-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/bp/data/' + i + '-bid-seq-test.csv'
    split_data(filepath, topath)
