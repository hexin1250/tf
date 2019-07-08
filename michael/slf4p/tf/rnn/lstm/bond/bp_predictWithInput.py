'''
Created on 2019年7月4日

@author: ch
'''
import pickle
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import consist_data

arr = []
# args = sys.argv[1:]
yield_v = 0.001392662
price_v = 100.07637840000001
rating = 6
args = [yield_v * price_v, rating, price_v, yield_v]

namelist = ["George", "Michael", "Sam", "Wendy", "Rose", "Vicky"]
for i in range(len(namelist)):
    name = namelist[i]
    # integer encode direction
    dataset = consist_data(sys_args=args, order=i)
    minmaxFile = '/Users/ch/git/tf/resources/bp/minmax/' + name + '-minmax.pk'
    with open(minmaxFile, 'rb') as fid:
        scaler = pickle.load(fid)
    val_X, val_y = get_data(dataset, scaler)
    modelFileName = '/Users/ch/git/tf/resources/bp/model/' + name + '.model'
    with open(modelFileName, 'rb') as fid:
        clf = pickle.load(fid)
    result = clf.predict(val_X)
    result_proba = clf.predict_proba(val_X)
    print(result_proba[0][1])
#     print(name, result_proba[0][1])
