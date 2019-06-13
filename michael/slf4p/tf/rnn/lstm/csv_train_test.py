'''
Created on 2019年6月13日

@author: ch
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from codecs import ignore_errors
from jupyter_core.troubleshoot import get_data

train_date_url = "File:/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/train.data"
train_file_path = tf.keras.utils.get_file("train.csv", train_date_url)

# np.set_printoptions(precision=3, suppress=True)
# with open(train_file_path, 'r') as f:
#     names_row = f.readline();
#     csv_columns = names_row.rstrip('\n').split(',')
#     print(csv_columns)

def get_dataset(file_path):
    dataset_s = tf.data.experimental.CsvDataset(
        filenames=file_path,
        record_defaults=[tf.string, tf.float32, tf.string, tf.float32, tf.float32, tf.int32],
        header=True,
        field_delim=",",
        select_cols=[0,1,2,3,4,5]
    )
#     columns = ['Name','cpn','rate','price','yield','result']
#     columns = ['Name','cpn','rate']
#     csv_columns = [col for col in columns]
#     dataset_s = tf.data.experimental.make_csv_dataset(
#         file_pattern=file_path,
#         batch_size=12,
#         select_columns=csv_columns
#     )
    return dataset_s

raw_train_data = get_dataset(train_file_path)
print(raw_train_data)
# examples, labels = next(iter(raw_train_data))
# print("Examples:\n", examples, "\n")
# print("Labels:\n", labels)

