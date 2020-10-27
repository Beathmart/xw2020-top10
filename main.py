import numpy as np
import pandas as pd
from tqdm import tqdm
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.metrics import classification_report


BASE_PATH = 'data/'
TRAIN_PATH = 'sensor_train.csv'
TEST_PATH = 'sensor_test.csv'
MODEL_PATH = 'models/'
MEAN = [8.03889039e-03, -6.41381949e-02, 2.37856977e-02, 8.64949391e-01, 2.80964889e+00, 7.83041714e+00, 6.44853358e-01, 9.78580749e+00]
STD = [0.6120893, 0.53693888, 0.7116134, 3.22046385, 3.01195336, 2.61300056, 0.87194132, 0.68427254]
TRAIN_COLS = ['acc_x', 'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg', 'mod', 'modg']
NUM_CLASSES = 19
NUM_SAMPLES = 60
NUM_FEATURES = len(TRAIN_COLS)
BLEND_COEF = [0.3, 0.15, 0.35, 0.2]
SEED = 42
mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
    4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
    8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
    12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
    16: 'C_2', 17: 'C_5', 18: 'C_6'}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def resample(df, num_resample=NUM_SAMPLES):
    print('Resampling...')
    m = len(df.fragment_id.unique())
    x = np.zeros((m, num_resample, NUM_FEATURES, 1))
    for i in df.fragment_id.unique():
        tmp = df[df.fragment_id==i][:num_resample]
        x[i,:,:, 0] = signal.resample(tmp[TRAIN_COLS], num_resample, np.array(tmp.time_point))[0]
    assert x.shape == (m, 60, 8, 1)
    return x

def get_score(y, y_pred):
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred:
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]:
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]:
        return 1.0/3
    else:
        return 0.0

def get_score_matrix(score_func):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(19):
        for j in range(19):
            matrix[i,j] = score_func(i,j)
    return matrix

SCORE_MATRIX = get_score_matrix(get_score)

def keras_acc_combo(y_true, y_pred):
    ## custom metric for keras
    y_true = K.eval(K.argmax(y_true, axis=1))
    y_pred = K.eval(K.argmax(y_pred, axis=1))
    scores = SCORE_MATRIX[y_true, y_pred]
    return np.mean(scores)

def acc_combo(y, y_pred):
    y = _check_dtype(y)
    y_pred = _check_dtype(y_pred)
    score = 0
    for i in range(len(y)):
        score += get_score(y[i], y_pred[i])
    return score/len(y)

def acc(y, y_pred, argmax=True):
    y = _check_dtype(y)
    y_pred = _check_dtype(y_pred)
    score = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            score += 1
    return score/len(y)

def _check_dtype(x):
    if 'values' in dir(x):
        x = x.values

    if len(x.shape) > 1 and x.shape[1] != 1:
        x = x.argmax(axis=1)
    else:
        x = x.reshape(-1)
    return x

def jitter(x, snr_db, dim=1):
    snr = 10 ** (snr_db / 10)
    Xp = np.sum(x ** 2, axis=dim, keepdims=True) / x.shape[dim]
    Np = Xp / snr
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)
    xn = x + n
    return xn

def scaling(x, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,NUM_FEATURES))
    myNoise = np.matmul(np.ones((NUM_SAMPLES,1)), scalingFactor)
    print(myNoise[0])
    myNoise = myNoise.reshape((1, NUM_SAMPLES, NUM_FEATURES, 1))
    return x*myNoise

def slide_window(ts, step=10, stride=1):
    m = ts.shape[0]
    n = ts.shape[1]
    result = np.zeros(((m-step)//stride+1, step, n))
    for i, s in enumerate(range(0, m-step+1, stride)):
        result[i] = ts[s:s+step, :]
    return result

def prepare_lstm_data(data, step=10, stride=1, verbose=True):
    print('Preparing data for LSTM...')
    samples = data.shape[0]
    m = data.shape[1]
    n = data.shape[2]
    result = np.zeros((samples, (m-step)//stride+1, step, n))
    for i in range(samples):
        result[i] = slide_window(data[i], step, stride)

    result = result.reshape((-1, step, n))
    if verbose:
        print(f'\nfinal shape: {result.shape}')
    return result

def cnn():
    input_shape = (NUM_SAMPLES, NUM_FEATURES, 1)
    model = Sequential()
    model.add(
        Conv2D(filters=256,
               kernel_size=(8, 3),
            #    activation='relu',
               padding='same',
               input_shape=input_shape,
            #    kernel_regularizer=l2(0.001),
               use_bias=False)
        )
    ## bn
    model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))


    model.add(
        Conv2D(filters=256,
               kernel_size=(5, 3),
               #    activation='relu',
               padding='same',
            #    dilation_rate=(2,2),
            #    kernel_regularizer=l2(0.001),
               use_bias=False)
        )
    ## bn
    model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))


    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(
        Conv2D(filters=512,
               kernel_size=(3, 3),
            #    activation='relu',
               padding='same',
            #    dilation_rate=(2,2),
            #    kernel_regularizer=l2(0.001),
               use_bias=False)
        )
    ## bn
    model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))


    model.add(
        Conv2D(filters=512,
               kernel_size=(3, 3),
            #    activation='relu',
               padding='same',
            #    dilation_rate=(2,2),
            #    kernel_regularizer=l2(0.001),
               use_bias=False)
        )
    ## bn
    model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D()) #
    model.add(Dropout(0.2))

    model.add(Dense(1024, use_bias=False))
    ## bn
    model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # model.add(Dense(1024, use_bias=False))
    # ## bn
    # model.add(BatchNormalization(scale=False))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(NUM_CLASSES, use_bias=False))
    ## bn
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

def lstmfcn(padding='same'):
    input_shape = (NUM_SAMPLES, NUM_FEATURES)
    input = Input(shape=input_shape)
    c = Conv1D(
        128,
        8,
        padding=padding,
        use_bias=False,
        # kernel_initializer='he_uniform'
        )(input)
    c = BatchNormalization(scale=False)(c)
    c = Activation('relu')(c)

    c = Conv1D(
        256,
        5,
        padding=padding,
        use_bias=False,
        # kernel_initializer='he_uniform'
        )(c)
    c = BatchNormalization(scale=False)(c)
    c = Activation('relu')(c)

    c = Conv1D(
        128,
        3,
        padding=padding,
        use_bias=False,
        # kernel_initializer='he_uniform'
        )(c)
    c = BatchNormalization(scale=False)(c)
    c = Activation('relu')(c)

    c = GlobalAveragePooling1D()(c)
    # c = Dropout(0.2)(c)

    l = LSTM(
        8,
        input_shape=input_shape,
        # return_sequences=True
    )(input)
    l = Dropout(0.8)(l)

    x = concatenate([l, c])

    x = Dense(NUM_CLASSES, use_bias=False)(x)
    x = BatchNormalization()(x)
    out = Activation('softmax')(x)

    model = Model(input, out)
    return model

def lstm(step):
    model = Sequential()
    model.add(LSTM(128, input_shape=(step,NUM_FEATURES), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(19, activation='softmax'))
    return model

def BLOCK(seq, filters, kernal_size):
    cnn = Conv1D(filters, 1, padding='SAME', activation='relu')(seq)
    cnn = LayerNormalization()(cnn)

    cnn = Conv1D(filters, kernal_size, padding='SAME', activation='relu')(cnn)
    cnn = LayerNormalization()(cnn)

    cnn = Conv1D(filters, 1, padding='SAME', activation='relu')(cnn)
    cnn = LayerNormalization()(cnn)

    seq = Conv1D(filters, 1)(seq)
    seq = Add()([seq, cnn])
    return seq

def BLOCK2(seq, filters=128, kernal_size=5):
    seq = BLOCK(seq, filters, kernal_size)
    seq = MaxPooling1D(2)(seq)
    seq = SpatialDropout1D(0.3)(seq)
    seq = BLOCK(seq, filters//2, kernal_size)
    seq = GlobalAveragePooling1D()(seq)
    return seq

def conv1d():
    inputs = Input(shape=(NUM_SAMPLES, NUM_FEATURES))
    seq_3 = BLOCK2(inputs, kernal_size=3)
    seq_5 = BLOCK2(inputs, kernal_size=5)
    seq_7 = BLOCK2(inputs, kernal_size=7)
    seq = concatenate([seq_3, seq_5, seq_7])
    seq = Dense(512, activation='relu')(seq)
    seq = Dropout(0.3)(seq)
    seq = Dense(128, activation='relu')(seq)
    seq = Dropout(0.3)(seq)
    outputs = Dense(NUM_CLASSES, activation='softmax')(seq)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def cnn_inference(x):
    print('CNN model inference...')
    model = cnn()
    t = x.reshape((-1, NUM_SAMPLES, NUM_FEATURES, 1))
    cnn_proba_t = np.zeros((len(t), NUM_CLASSES))
    for fold in tqdm(range(5)):
        model.load_weights(MODEL_PATH + f'cnn_fold{fold}.h5')
        cnn_proba_t += model.predict(t) / 5
    assert  cnn_proba_t.shape == (7500, 19)
    return cnn_proba_t

def lstmfcn_inference(x):
    print('LSTM-FCN model inference...')
    model = lstmfcn()
    t = x.reshape((-1, NUM_SAMPLES, NUM_FEATURES))
    lstmfcn_proba_t = np.zeros((len(t), NUM_CLASSES))
    for fold in tqdm(range(5)):
        model.load_weights(MODEL_PATH + f'lstmfcn_fold{fold}.h5')
        lstmfcn_proba_t += model.predict(t) / 5
    assert lstmfcn_proba_t.shape == (7500, 19)
    return lstmfcn_proba_t

def lstm_inference(x):
    print('LSTM model inference...')
    t = x.reshape((-1, NUM_SAMPLES, NUM_FEATURES))
    step = 20
    stride = 5
    npersamp = (NUM_SAMPLES - step) // stride + 1
    assert npersamp == 9

    t_l = prepare_lstm_data(t, step, stride, verbose=False)
    model = lstm(step)
    lstm_proba_t = np.zeros((len(t_l), NUM_CLASSES))
    for fold in tqdm(range(5)):
        model.load_weights(MODEL_PATH + f'lstm_fold{fold}.h5')
        lstm_proba_t += model.predict(t_l) / 5
    t_frag = lstm_proba_t.reshape((len(t), npersamp, NUM_CLASSES))
    t_frag = np.sum(t_frag, axis=1).reshape((len(t), NUM_CLASSES))
    lstm_proba_t = t_frag / npersamp
    assert lstm_proba_t.shape == (7500, 19)
    return lstm_proba_t

def conv1d_inference(x):
    print('conv1d model inference...')
    model = conv1d()
    t = x.reshape((-1, NUM_SAMPLES, NUM_FEATURES))
    conv1d_proba_t = np.zeros((len(t), NUM_CLASSES))
    for fold in tqdm(range(5)):
        model.load_weights(MODEL_PATH + f'conv1d_fold{fold}.h5')
        conv1d_proba_t += model.predict(t) / 5
    assert conv1d_proba_t.shape == (7500, 19)
    return conv1d_proba_t

def cnn_train(x, y):
    X = x.reshape((len(x), NUM_SAMPLES, NUM_FEATURES, 1))
    kfold = StratifiedKFold(5, shuffle=True, random_state=SEED)
    seed_everything(SEED)

    for fold, (trn, val) in enumerate(kfold.split(X[:7292], y[:7292])):
        trn_aug1 = trn + 7292
        # val_aug1 = val + 7292

        trn_aug2 = trn + 7292 * 2
        # val_aug2 = val + 7292 * 2

        trn_aug3 = trn + 7292 * 3
        # val_aug3 = val + 7292 * 3

        trn_aug4 = trn + 7292 * 4


        trn = np.concatenate((trn,
                              trn_aug1,
                              trn_aug2,
                              trn_aug3,
                              trn_aug4,
                              ))
        # val = np.concatenate((val, val_aug1, val_aug2, val_aug3))

        y_ = to_categorical(y, num_classes=NUM_CLASSES)
        model = cnn()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['acc',
                              # keras_acc_combo
                               ],
                     # run_eagerly=True
                      )
        plateau = ReduceLROnPlateau(monitor='val_acc',  # 'val_keras_acc_combo',
                                    verbose=0,
                                    mode='max',
                                    factor=0.5,
                                    patience=5,
                                    min_lr=10 ** -5)
        early_stopping = EarlyStopping(monitor='val_acc',  # 'val_keras_acc_combo',
                                       verbose=0,
                                       mode='max',
                                       patience=10)
        checkpoint = ModelCheckpoint(MODEL_PATH+f'cnn_fold{fold}.h5',
                                     monitor='val_acc',  # 'val_keras_acc_combo',
                                     verbose=0,
                                     mode='max',
                                     save_best_only=True)

        if fold == 0:
            print(model.summary())

        print(fold)
        model.fit(
            X[trn], y_[trn],
            epochs=200,
            batch_size=128,
            verbose=1,
            shuffle=True,
            validation_data=(X[val], y_[val]),
            callbacks=[plateau, early_stopping, checkpoint]
        )

        print('-' * 50)
        print('-' * 50)


def lstmfcn_train(x, y):
    X = x.reshape((len(x), NUM_SAMPLES, NUM_FEATURES))
    kfold = StratifiedKFold(5, shuffle=True, random_state=SEED)
    seed_everything(SEED)


    for fold, (trn, val) in enumerate(kfold.split(X[:7292], y[:7292])):
        trn_aug1 = trn + 7292
        # val_aug1 = val + 7292

        trn_aug2 = trn + 7292 * 2
        # val_aug2 = val + 7292 * 2

        trn_aug3 = trn + 7292 * 3
        # val_aug3 = val + 7292 * 3

        trn_aug4 = trn + 7292 * 4

        trn = np.concatenate((trn,
                              trn_aug1,
                              trn_aug2,
                              trn_aug3,
                              trn_aug4,
                              ))
        # val = np.concatenate((val, val_aug1, val_aug2, val_aug3))

        y_ = to_categorical(y, num_classes=NUM_CLASSES)
        model = lstmfcn()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['acc',
                              # keras_acc_combo
                               ],
                      #run_eagerly=True
                      )
        plateau = ReduceLROnPlateau(monitor='val_acc',  # val_keras_acc_combo',
                                    verbose=0,
                                    mode='max',
                                    factor=0.5,
                                    patience=5,
                                    min_lr=10 ** -5)
        early_stopping = EarlyStopping(monitor='val_acc',  # val_keras_acc_combo',
                                       verbose=0,
                                       mode='max',
                                       patience=10)
        checkpoint = ModelCheckpoint(MODEL_PATH+f'lstmfcn_fold{fold}.h5',
                                     monitor='val_acc',  # val_keras_acc_combo',
                                     verbose=0,
                                     mode='max',
                                     save_best_only=True)

        if fold == 0:
            print(model.summary())

        print(fold)
        model.fit(
            X[trn], y_[trn],
            epochs=200,
            batch_size=128,
            verbose=1,
            shuffle=True,
            validation_data=(X[val], y_[val]),
            callbacks=[plateau, early_stopping, checkpoint]
        )

        print('-' * 50)
        print('-' * 50)


def lstm_train(x, y):
    X = x.reshape((len(x), NUM_SAMPLES, NUM_FEATURES))
    step = 20
    stride = 5
    npersamp = (NUM_SAMPLES - step) // stride + 1
    X_l = prepare_lstm_data(X, step, stride)

    X_frag = X_l.reshape(-1, npersamp, step, NUM_FEATURES)

    kfold = StratifiedKFold(5, shuffle=True, random_state=SEED)
    seed_everything(SEED)

    for fold, (trn, val) in enumerate(kfold.split(X_frag[:7292], y[:7292])):
        trn_aug1 = trn + 7292
        # val_aug1 = val + 7292

        trn_aug2 = trn + 7292 * 2
        # val_aug2 = val + 7292 * 2

        trn_aug3 = trn + 7292 * 3
        # val_aug3 = val + 7292 * 3

        trn_aug4 = trn + 7292 * 4

        trn = np.concatenate((trn,
                              trn_aug1,
                              trn_aug2,
                              trn_aug3,
                              trn_aug4,
                              ))
        # val = np.concatenate((val, val_aug1, val_aug2, val_aug3))


        y_train = to_categorical(np.repeat(y[trn], npersamp), num_classes=NUM_CLASSES)
        y_valid = to_categorical(np.repeat(y[val], npersamp), num_classes=NUM_CLASSES)
        X_train = X_frag[trn].reshape(-1, step, NUM_FEATURES)
        X_valid = X_frag[val].reshape(-1, step, NUM_FEATURES)

        model = lstm(step)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['acc',
                              # keras_acc_combo
                               ],
                      #run_eagerly=True
                      )
        plateau = ReduceLROnPlateau(monitor='val_acc',  # 'val_keras_acc_combo',
                                    verbose=0,
                                    mode='max',
                                    factor=0.5,
                                    patience=5,
                                    min_lr=10 ** -5)
        early_stopping = EarlyStopping(monitor='val_acc',  # 'val_keras_acc_combo',
                                       verbose=0,
                                       mode='max',
                                       patience=10)
        checkpoint = ModelCheckpoint(MODEL_PATH+f'lstm_fold{fold}.h5',
                                     monitor='val_acc',  # 'val_keras_acc_combo',
                                     verbose=0,
                                     mode='max',
                                     save_best_only=True)

        if fold == 0:
            print(model.summary())
            print('\n\n')

        print(f'fold {fold}')
        print(f'X_train shape: {X_train.shape} y_train shape: {y_train.shape}')
        print(f'X_valid shape: {X_valid.shape} y_valid shape: {y_valid.shape}')
        model.fit(X_train, y_train,
                            epochs=200,
                            batch_size=256,
                            verbose=1,
                            shuffle=True,
                            validation_data=(X_valid, y_valid),
                            callbacks=[plateau, early_stopping, checkpoint])

        print('-' * 50)
        print('-' * 50)

def conv1d_train(x, y):
    X = x.reshape((len(x), NUM_SAMPLES, NUM_FEATURES))
    kfold = StratifiedKFold(5, shuffle=True, random_state=SEED)
    seed_everything(SEED)

    for fold, (trn, val) in enumerate(kfold.split(X[:7292], y[:7292])):
        trn_aug1 = trn + 7292
        # val_aug1 = val + 7292

        trn_aug2 = trn + 7292 * 2
        # val_aug2 = val + 7292 * 2

        trn_aug3 = trn + 7292 * 3
        # val_aug3 = val + 7292 * 3

        trn_aug4 = trn + 7292 * 4

        trn = np.concatenate((trn,
                              trn_aug1,
                              trn_aug2,
                              trn_aug3,
                              trn_aug4,
                              ))
        # val = np.concatenate((val, val_aug1, val_aug2, val_aug3))

        y_ = to_categorical(y, num_classes=NUM_CLASSES)
        model = conv1d()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['acc',
                               # keras_acc_combo
                               ],
                      # run_eagerly=True
                      )
        plateau = ReduceLROnPlateau(monitor='val_acc',  # val_keras_acc_combo',
                                    verbose=0,
                                    mode='max',
                                    factor=0.5,
                                    patience=5,
                                    min_lr=10 ** -5)
        early_stopping = EarlyStopping(monitor='val_acc',  # val_keras_acc_combo',
                                       verbose=0,
                                       mode='max',
                                       patience=10)
        checkpoint = ModelCheckpoint(MODEL_PATH + f'conv1d_fold{fold}.h5',
                                     monitor='val_acc',  # val_keras_acc_combo',
                                     verbose=0,
                                     mode='max',
                                     save_best_only=True)

        if fold == 0:
            print(model.summary())

        print(fold)
        model.fit(
            X[trn], y_[trn],
            epochs=200,
            batch_size=128,
            verbose=1,
            shuffle=True,
            validation_data=(X[val], y_[val]),
            callbacks=[plateau, early_stopping, checkpoint]
        )

        print('-' * 50)
        print('-' * 50)

def train():
    print('start train')
    train = pd.read_csv(BASE_PATH + TRAIN_PATH)
    train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
    train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5

    X = resample(train)
    y = train.groupby('fragment_id')['behavior_id'].min()
    X = (X - np.array(MEAN).reshape((1, 1, 8, 1))) / np.array(STD).reshape((1, 1, 8, 1))
    assert X.shape == (7292, 60, 8, 1)

    x1 = jitter(X, 8)
    x2 = jitter(X, 10)
    x3 = jitter(X, 12)
    x4 = jitter(X, 15)

    X = np.concatenate((X, x1, x2, x3, x4))
    y = np.tile(y, 5)

    cnn_train(X, y)
    lstmfcn_train(X, y)
    lstm_train(X, y)
    conv1d_train(X, y)


def inference():
    print('start inference')
    test = pd.read_csv(BASE_PATH+TEST_PATH)
    sub = pd.read_csv(BASE_PATH+'sample_submission.csv')

    test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
    test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

    t = resample(test)
    t = (t - np.array(MEAN).reshape((1,1,8,1))) / np.array(STD).reshape((1,1,8,1))
    assert t.shape == (7500, 60, 8, 1)

    cnn_proba_t = cnn_inference(t)
    lstmfcn_proba_t = lstmfcn_inference(t)
    lstm_proba_t = lstm_inference(t)
    conv1d_proba_t = conv1d_inference(t)

    yt = np.argmax(
        (
            cnn_proba_t * BLEND_COEF[0] +
            lstmfcn_proba_t * BLEND_COEF[1] +
            lstm_proba_t * BLEND_COEF[2] +
            conv1d_proba_t * BLEND_COEF[3]
    ), axis=1)
    sub.behavior_id = yt
    sub.to_csv('submission.csv', index=False)

print(tf.__version__)
train()
inference()
print('-'*50)
print('finished')

