import json
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import pandas as pd
import re
import jieba
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
# from attention import Attention

epoch = 6
state = 456
batchsize = 4
maxlen = 510
nfold = 10
learning_rate = 2e-5

config_path = '../pretrain_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../pretrain_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../pretrain_model/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

file_path = '../log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp + '.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

def read_data(file_path, id, name):
    train_id = []
    train_title = []
    train_text = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for idx, line in enumerate(f):
            line = line.strip().split(',')
            train_id.append(line[0].replace('\'', '').replace(' ', ''))
            train_title.append(line[1])
            train_text.append('，'.join(line[2:]))
    output = pd.DataFrame(dtype=str)
    output[id] = train_id
    output[name + '_title'] = train_title
    output[name + '_content'] = train_text
    return output

train_interrelation = pd.read_csv('../data/Train_Interrelation.csv', dtype=str)
print(len(train_interrelation))

Train_Achievements = read_data('../data/Train_Achievements.csv', 'Aid', 'Achievements')
print(len(Train_Achievements))

Requirements = read_data('../data/Requirements.csv', 'Rid', 'Requirements')
print(len(Requirements))

TestPrediction = pd.read_csv('../data/TestPrediction.csv', dtype=str)
print(len(TestPrediction))

Test_Achievements = read_data('../data/Test_Achievements.csv', 'Aid', 'Achievements')

train = pd.merge(train_interrelation, Train_Achievements, on='Aid', how='left')
train = pd.merge(train, Requirements, on='Rid', how='left')

test = pd.merge(TestPrediction, Test_Achievements, on='Aid', how='left')
test = pd.merge(test, Requirements, on='Rid', how='left')

#使用titile + content
train['Achievements_title'] = train['Achievements_title'] + '_' + train['Achievements_content']
train['Requirements_title'] = train['Requirements_title'] + '_' + train['Requirements_content']
test['Achievements_title'] = test['Achievements_title'] + '_' + test['Achievements_content']
test['Requirements_title'] = test['Requirements_title'] + '_' + test['Requirements_content']

train_achievements = train['Achievements_title'].values
train_requirements = train['Requirements_title'].values

labels = train['Level'].astype(int).values - 1
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)

test_achievements = test['Achievements_title'].values
test_requirements = test['Requirements_title'].values

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    def __init__(self, data, batch_size=batchsize):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2, y = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y = [], [], []
            for c, i in enumerate(idxs):
                achievements = X1[i]
                requirements = X2[i]
                t, t_ = tokenizer.encode(first=achievements, second=requirements, max_len=maxlen)
                T.append(t)
                T_.append(t_)
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = seq_padding(T)
                    T_ = seq_padding(T_)
                    Y = seq_padding(Y)
                    yield [T, T_], Y
                    T, T_, Y = [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import *
from keras.layers import Conv1D, GlobalMaxPooling1D, Concatenate, Bidirectional, GRU

def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)
    output = Dense(64, activation='relu')(T)
    output = Dense(4, activation='softmax')(output)
    model = Model([T1, T2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model

def predict(data):
    prob = []
    val_x1, val_x2 = data
    for i in tqdm(range(len(val_x1))):
        achievements = val_x1[i]
        requirements = val_x2[i]

        t1, t1_ = tokenizer.encode(first=achievements, second=requirements, max_len=maxlen)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob

num_model_seed = 1
predict_cat = np.zeros((len(test), 4), dtype=np.float32)
oof = np.zeros((len(train), 4), dtype=np.float32)
for model_seed in range(num_model_seed):
    oof_train = np.zeros((len(train), 4), dtype=np.float32)
    oof_test = np.zeros((len(test), 4), dtype=np.float32)
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=state)
    for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
        logger.info('================     fold {}        ==============='.format(fold))
        x1 = train_achievements[train_index]
        x2 = train_requirements[train_index]
        y = labels_cat[train_index]

        val_x1 = train_achievements[valid_index]
        val_x2 = train_requirements[valid_index]
        val_y = labels[valid_index]
        val_cat = labels_cat[valid_index]

        model = get_model()
        early_stopping = EarlyStopping(monitor='val_acc', patience=2)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)
        checkpoint = ModelCheckpoint('../model_save/model3/' + str(fold) + '.hdf5', monitor='val_acc',
                                     verbose=2, save_best_only=True, mode='max', save_weights_only=True)
        train_D = data_generator([x1, x2, y])
        valid_D = data_generator([val_x1, val_x2, val_cat])

        model.fit_generator(train_D.__iter__(),
                            steps_per_epoch=len(train_D),
                            epochs=epoch,
                            validation_data=valid_D.__iter__(),
                            validation_steps=len(valid_D),
                            callbacks=[early_stopping, plateau, checkpoint]
                            )
        oof_train[valid_index] = predict([val_x1, val_x2])
        score = 1.0 / (1 + mean_absolute_error(labels[valid_index] + 1, np.argmax(oof_train[valid_index], axis=1) + 1))
        acc = accuracy_score(labels[valid_index] + 1, np.argmax(oof_train[valid_index], axis=1) + 1)
        f1 = f1_score(labels[valid_index] + 1, np.argmax(oof_train[valid_index], axis=1) + 1, average='macro')
        logger.info('score: %.4f, acc: %.4f, f1: %.4f\n' % (score, acc, f1))
        oof_test += predict([test_achievements, test_requirements])
        K.clear_session()

    oof_test /= nfold
    np.savetxt('../model_save/model3/model3_train_bert.txt', oof_train)
    np.savetxt('../model_save/model3/model3_test_bert.txt', oof_test)
    cv_score = 1.0 / (1 + mean_absolute_error(labels + 1, np.argmax(oof_train, axis=1) + 1))
    # print(cv_score)
    predict_cat += oof_test / num_model_seed
    oof += oof_train / num_model_seed
cv_score = 1.0 / (1 + mean_absolute_error(labels + 1, np.argmax(oof, axis=1) + 1))
print(cv_score)
test['Level'] = np.argmax(predict_cat, axis=1) + 1
test[['Guid', 'Level']].to_csv('../submit/model3_bert_{}.csv'.format(cv_score), index=False)