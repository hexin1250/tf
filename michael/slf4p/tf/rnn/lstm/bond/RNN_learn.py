'''
Created on 2019年6月26日

@author: ch
'''
import pandas as pd
import keras
import pickle
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_XY

times=1
namelist = ["George", "Michael", "Sam", "Wendy", "Rose", "Vicky"]
# namelist = ["Vicky"]
for name in namelist:
    dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/' + name + '-bid-seq-train.csv', header=0, index_col=0)
    scaler = MinMaxScaler(feature_range=(0,1))
    values_train_X, train_y = get_data(dataset, scaler, fit=True)
    train_X = get_XY(values_train_X)
    with open('/Users/ch/git/tf/resources/rnn/minmax/' + name + '-minmax.pk', 'wb') as fid:
        pickle.dump(scaler, fid)
    
    validation_ds = pd.read_csv('/Users/ch/git/tf/resources/data/' + name + '-bid-seq-val.csv', header=0, index_col=0)
    values_test_X, test_y = get_data(validation_ds, scaler)
    test_X = get_XY(values_test_X)
#     
#     model = keras.models.Sequential()
#     model.add(keras.layers.SimpleRNN(units=10, input_shape=(train_X.shape[1], train_X.shape[2])))
#     model.add(keras.layers.Dense(1, activation='sigmoid'))
#     model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])

    model = load_model(filepath='/Users/ch/git/tf/resources/rnn/model/' + name + '-bond.md')
    model.load_weights(filepath='/Users/ch/git/tf/resources/rnn/model/' + name + '-bond_weights.md')
    print("current train client:", name)
#     batch_size = int(len(train_y) / 3)
    batch_size = 100
    history = model.fit(train_X, train_y, epochs=times, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    
    # save model
    model.save(filepath='/Users/ch/git/tf/resources/rnn/model/' + name + '-bond_weights.md')
    model.save(filepath='/Users/ch/git/tf/resources/rnn/model/' + name + '-bond.md')
