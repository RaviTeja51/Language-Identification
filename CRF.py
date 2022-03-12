import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_crf import CRFModel
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, SGD
from sklearn.metrics import classification_report, f1_score, confusion_matrix

emb = 64
max_seq_len = 22
batch_size = 128
epochs = 30

X_train_eng = pad_sequences(eng_train_tokens,maxlen=max_seq_len,padding="post",truncating="post") 
X_dev_eng = pad_sequences(eng_dev_tokens,maxlen=max_seq_len,padding="post",truncating="post")
    
X_train_hin = pad_sequences(hin_train_tokens,maxlen=max_seq_len,padding="post",truncating="post") 
X_dev_hin = pad_sequences(hin_dev_tokens,maxlen=max_seq_len,padding="post",truncating="post")

train_y = pad_sequences(train_y_token,maxlen=max_seq_len,padding="post",truncating="post")
dev_y = pad_sequences(dev_y_token,maxlen=max_seq_len,padding="post",truncating="post")

dp1 = 0.3
units = 64
filters =  5
kernel = 16

hin_X = tf.keras.layers.Input((None,))
eng_X = tf.keras.layers.Input((None,))
hin = tf.keras.layers.Embedding(hin_num_words,
                    EMBEDDING_DIM,
                    embeddings_initializer=tf.keras.initializers.Constant(hin_embedding_matrix),
                    trainable=False,
                    mask_zero=True,name="hin_emb")(hin_X)
eng = tf.keras.layers.Embedding(eng_num_words,
                    EMBEDDING_DIM,
                    embeddings_initializer=tf.keras.initializers.Constant(eng_embedding_matrix),
                    trainable=False,
                    mask_zero=True,name="eng_emb")(eng_X)

eng_proj = tf.keras.layers.Dense(units=emb,use_bias=False,activation="tanh",name="eng_proj")(eng)
hin_proj = tf.keras.layers.Dense(units=emb,use_bias=False,activation="tanh",name="hin_proj")(hin)

hin_w = tf.keras.layers.Dense(units=emb,use_bias=False,name="hin_weighted")(hin_proj)
eng_w = tf.keras.layers.Dense(units=emb,use_bias=False,name="eng_weighted")(eng_proj)

meta = tf.keras.layers.Add()([hin_w,eng_w])


meta = tf.keras.layers.SpatialDropout1D(dp1)(meta)



seq_model = tf.keras.Model(inputs=[hin_X, eng_X],
                        outputs=meta)
model =  CRFModel(seq_model, num_classes)

lr = 0.0003
optim = Adam(learning_rate = lr)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = Adam(learning_rate = lr),
              metrics=['accuracy'])
history = model.fit(x=[X_train_hin, X_train_eng], y=train_y,batch_size = batch_size,
                    epochs = epochs,
                    validation_data = ([X_dev_hin, X_dev_eng], dev_y),
                    verbose=2)