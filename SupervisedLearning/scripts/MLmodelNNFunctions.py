#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:48:20 2021

@author: bmondal
"""

import numpy as np
import sqlite3 as sq
import pandas as pd
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs

import keras_tuner as kt

from sklearn.model_selection import RepeatedKFold

#%%
'''
#https://stackoverflow.com/a/62583369
Epoch is an approach by which we pass the same dataset multiple times to the 
network in order to find optimal weights.

As we are using Gradient descent for optimization and there is a possibility 
of landing at local minima, so in order to overcome that we pass the same 
dataset n times (i.e. n Epochs) to find optimal weights.

The number of Epochs is subjected to application 
and a smaller number of Epochs might lead to underfitting 
while a larger number of iteration could end up introducing overfitting.
'''       
#%% ----------------------- Hyper parameter tuning ----------------------------
# get the model
class MyHyperModel(kt.HyperModel):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
    def build(self, hp):
        MyReLu = tf.keras.layers.ReLU(max_value=1.0)
        MyLeakyRelu = tf.keras.layers.ReLU(max_value=1.0, negative_slope=0.01)
        d = {'ReLU': MyReLu, 'LeakyReLU':MyLeakyRelu}
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.n_inputs, )))
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=d[hp.Choice(f"activation_{i}", ['ReLU','LeakyReLU'])],
                    kernel_regularizer=regularizers.l2(0.0001)
                    )
                )
            if hp.Boolean(f"dropout_{i}"):
                model.add(layers.Dropout(
                    rate=hp.Choice(f'dpr_{i}', values=[1e-1, 2e-1, 3e-1, 4e-1, 5e-1])
                    )
                    )
                
        model.add(layers.Dense(self.n_outputs, activation=MyReLu))
        
        learning_start = hp.Float("lr", min_value=0.001, max_value=0.1, step=0.001)
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            learning_start,
            decay_steps=10,
            decay_rate=0.5,
            staircase=False)
        
        model.compile(
            optimizer=keras.optimizers.Adam(lr_schedule),
            loss='mean_squared_error',
            metrics=[
                'mean_squared_logarithmic_error',
                'mean_absolute_error']
        )
        return model
    
    def fit(self, hp, model, x, y, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.experimental.preprocessing.Normalization()(x)
        return model.fit(
            x,
            y,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )


def OptimalHyperParameters(X_train,X_test,Y_train,Y_test):
    n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
    hypermodel = MyHyperModel(n_inputs, n_outputs)
    tuner = kt.Hyperband(hypermodel=hypermodel,
                         objective='val_loss',
                         max_epochs=200,
                         #max_trials=3,
                         factor=3,
                         overwrite=True,
                         executions_per_trial=1,
                         directory='my_dir',
                         project_name='intro_to_kt')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner.search(X_train, Y_train, epochs=100, validation_split=0.2, 
                 verbose=0, callbacks=[stop_early])
    
    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(tuner.results_summary())
    #'and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}'
    
    
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    #model = tuner.hypermodel.build(best_hps)
    model = hypermodel.build(best_hps)
    history = model.fit(X_train, Y_train, epochs=100, verbose=0, validation_split=0.2)

    val_acc_per_epoch = history.history['val_loss']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel_final = hypermodel.build(best_hps)

    # Retrain the model
    history2 = hypermodel_final.fit(X_train, Y_train, epochs=best_epoch, 
                                    verbose=0, validation_split=0.2)
    
    return hypermodel_final, history2, hypermodel_final.metrics_names

# Default hyper-parameters
def CreateHyperParamers(N_TRAIN, nepochs=200,nbatch=2,decayrate=10):
    # decayrate == Decay the learning rate to 1/2 of the base 
    # rate at 10 epochs, 1/3 at 20 epochs and so on.
    BATCH_SIZE = N_TRAIN//nbatch
    STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
    DECAY_STEP = STEPS_PER_EPOCH * (nepochs//decayrate)
    optimizer=tf.optimizers.Adam(learning_rate=0.001)
    return (nepochs, nbatch, BATCH_SIZE, DECAY_STEP, optimizer)

#%%---------------------------- Model functions -------------------------------
def CheckVersions():
    #conda install -c conda-forge tensorflow
    import tensorflow as tf
    print(tf.__version__)
    print('tensorflow: 2.6.2; Your version is: {}'.format(tf.__version__))
    return 

def GenerateData(df):
    X, Y = df.iloc[:, :8], df.iloc[:,-2:]
    return X, Y

def get_dataset_test(dbname):
    conn = sq.connect(dbname)
    df = pd.read_sql_query('SELECT * FROM EgSODATA WHERE GALLIUM==100', conn)
    df = df.dropna()
    """
    D==Direct bandgap at the Gamma point
    d==Direct bandgap at some other k-point
    I==Indirect bandgap with VBM at the Gamma point
    i==Indirect bandgap with VBM at some other k-point
    """
    #df['NATURE'] = df['NATURE'].map({1: 'D', 2: 'd', 3: 'I', 4: 'i'})
    #df = pd.get_dummies(df, columns=['NATURE'], prefix='', prefix_sep='')

    X, Y = GenerateData(df)
    return X, Y

def TrainTestSplit_p1(dff, frac=0.8, random_state=0):
    dff_train = dff.sample(frac=frac, random_state=random_state)
    dff_test = dff.drop(dff_train.index)
    return dff_train, dff_test

def TrainTestSplit(df, xfeatures, yfeatures, YfeaturesScale=1, frac=0.8, random_state=0):
    X, Y, EgNature = df[xfeatures], df[yfeatures]/YfeaturesScale, df['NATURE']
    X_train, X_test = TrainTestSplit_p1(X, frac=frac, random_state=random_state)
    Y_train, Y_test = TrainTestSplit_p1(Y, frac=frac, random_state=random_state)
    EN_train, EN_test = TrainTestSplit_p1(EgNature, frac=frac, random_state=random_state)
    return X_train, X_test, Y_train, Y_test, EN_train, EN_test

# get the dataset
def get_dataset(dbname):
    conn = sq.connect(dbname)
    df = pd.read_sql_query('SELECT * FROM COMPUTATIONALDATA', conn)
    df = df.dropna()
    """
    D==Direct bandgap at the Gamma point
    d==Direct bandgap at some other k-point
    I==Indirect bandgap with VBM at the Gamma point
    i==Indirect bandgap with VBM at some other k-point
    """
    df['NATURE'] = df['NATURE'].map({1: 'D', 2: 'd', 3: 'I', 4: 'i'})
    df = pd.get_dummies(df, columns=['NATURE'], prefix='', prefix_sep='')
    X, y = df.iloc[:, :8], df.iloc[:, 11:]
    return X, y

def FeatureNormalization(X):
    # Feature normalization: require tf.__version__>=2.6
    normalizer = layers.experimental.preprocessing.Normalization(axis=-1)
    #normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X))
    return normalizer

def get_optimizer(DECAY_STEP):
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=DECAY_STEP,
        decay_rate=1,
        staircase=False)
    return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
  return [
    #tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80),
    #tf.keras.callbacks.TensorBoard(logdir/name),
  ]

# get the model
def get_model(normalizer, n_inputs, n_outputs, dropoutrate=0.1):
    # https://github.com/tensorflow/tensorflow/issues/45786
    # https://stackoverflow.com/a/56869141
    # https://towardsdatascience.com/7-popular-activation-functions-you-should-
    # know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6
    activationfn = tf.keras.layers.ReLU(max_value=1.0)
    activationfnt2 = tf.keras.layers.ReLU(max_value=1.0, negative_slope=0.01)
    model = keras.Sequential(
            [
                #keras.Input(shape=(n_inputs,)),
                normalizer,
                layers.Dense(32, activation=activationfn, kernel_regularizer=regularizers.l2(0.0001), name="layer1"),
                layers.Dropout(dropoutrate),
                layers.Dense(32, activation=activationfnt2, kernel_regularizer=regularizers.l2(0.0001), name="layer2"),
                layers.Dropout(dropoutrate),
                layers.Dense(32, activation=activationfnt2, kernel_regularizer=regularizers.l2(0.0001), name="layer3"),
                layers.Dropout(dropoutrate),
                layers.Dense(32, activation=activationfn, kernel_regularizer=regularizers.l2(0.0001), name="layer4"),
                #layers.Dropout(dropoutrate),
                layers.Dense(n_outputs, activation=activationfn, name="output")
            ],
            name='my_sequential'
            )
    return model

def compile_and_fit(normalizer, X, Y, DECAY_STEP, name=None, optimizer=None, \
                    lossfn='mean_squared_error', batchsize=96, max_epochs=1000):
    
    n_inputs, n_outputs = X.shape[1], Y.shape[1]
    model = get_model(normalizer, n_inputs, n_outputs)
    
    if optimizer is None:
        optimizer = get_optimizer(DECAY_STEP)
        
    print(model.summary())
    
    model.compile(optimizer=optimizer,
                  loss=lossfn, 
                   metrics=[
                       'mean_squared_error',
                       'mean_absolute_percentage_error',
                       'mean_squared_logarithmic_error',
                       'mean_absolute_error']
                  )
    
    history = model.fit(X, 
                        Y,
                        validation_split=0.2,
                        verbose=0, 
                        epochs=max_epochs,
                        batch_size = batchsize,
                        callbacks=get_callbacks(name),
                        #steps_per_epoch = STEPS_PER_EPOCH
                        )
    print("\nMetrices:")
    print(model.metrics_names)
    return model, history, model.metrics_names
 
# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, 
            y_train, 
            verbose=0, 
            epochs=20,
            validation_split = 0.2)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		# store result
		print('>%.3f' % mae)
		results.append(mae)
	return results
