'''
Created on 2019年6月26日

@author: ch
'''
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import split_data

namelist = ["Michael", "Wendy", "Vicky", "Sam", "George", "Rose"]
for i in namelist:
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/train-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/data/' + i + '-bid-seq-train.csv'
    split_data(filepath, topath)
    
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/val-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/data/' + i + '-bid-seq-val.csv'
    split_data(filepath, topath)
    
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/test-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/data/' + i + '-bid-seq-test.csv'
    split_data(filepath, topath)

