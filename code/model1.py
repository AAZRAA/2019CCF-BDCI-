import json
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam, Adamax, Nadam
from keras.backend.tensorflow_backend import set_session
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import pandas as pd
import re
import jieba
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

epoch = 6
learning_rate = 5e-5
min_learning_rate = 1e-5
maxlen = 47
state = 456
nfold = 30
batchsize = 16

#预训练模型
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

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
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

#只用title
train_achievements = train['Achievements_title'].values
train_requirements = train['Requirements_title'].values

labels = train['Level'].astype(int).values - 1
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)

test_achievements = test['Achievements_title'].values
test_requirements = test['Requirements_title'].values

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
                    T = np.array(T)
                    T_ = np.array(T_)
                    Y = np.array(Y)
                    yield [T, T_], Y
                    T, T_, Y = [], [], []

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback

def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)

    output = Dense(4, activation='softmax')(T)

    model = Model([T1, T2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model


class Evaluate(Callback):
    def __init__(self, val_data, val_index):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            model.save_weights('../model_save/model1/model1_bert{}.w'.format(fold))
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, score: %.4f, acc: %.4f, f1: %.4f,best: %.4f\n' % (self.lr, epoch, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_x2, val_y, val_cat = self.val_data
        for i in tqdm(range(len(val_x1))):
            achievements = val_x1[i]
            requirements = val_x2[i]

            t1, t1_ = tokenizer.encode(first=achievements, second=requirements)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0]+1)
            prob.append(_prob[0])

        score = 1.0 / (1 + mean_absolute_error(val_y+1, self.predict))
        acc = accuracy_score(val_y+1, self.predict)
        f1 = f1_score(val_y+1, self.predict, average='macro')
        return score, acc, f1

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=state)

def predict(data):
    prob = []
    val_x1, val_x2 = data
    for i in tqdm(range(len(val_x1))):
        achievements = val_x1[i]
        requirements = val_x2[i]

        t1, t1_ = tokenizer.encode(first=achievements, second=requirements)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob

oof_train = np.zeros((len(train), 4), dtype=np.float32)
oof_test = np.zeros((len(test), 4), dtype=np.float32)
for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
    logger.info('================     fold {}        ==============='.format(fold))
    x1 = train_achievements[train_index]
    x2 = train_requirements[train_index]
    y = labels_cat[train_index]

    val_x1 = train_achievements[valid_index]
    val_x2 = train_requirements[valid_index]
    val_y = labels[valid_index]
    val_cat = labels_cat[valid_index]

    train_D = data_generator([x1, x2, y])
    evaluator = Evaluate([val_x1, val_x2, val_y, val_cat], valid_index)

    model = get_model()
    model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=epoch,
                        callbacks=[evaluator]
                       )
    model.load_weights('../model_save/model1/model1_bert{}.w'.format(fold))
    oof_test += predict([test_achievements, test_requirements])
    K.clear_session()

oof_test /= nfold
np.savetxt('../model_save/model1/model1_train_bert.txt', oof_train)
np.savetxt('../model_save/model1/model1_test_bert.txt', oof_test)

cv_score = 1.0 / (1 + mean_absolute_error(labels+1, np.argmax(oof_train, axis=1) + 1))
print(cv_score)
test['Level'] = np.argmax(oof_test, axis=1) + 1
test[['Guid', 'Level']].to_csv('../submit/model1_bert_{}.csv'.format(cv_score), index=False)