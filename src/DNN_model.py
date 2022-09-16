import sys
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Dropout
from tensorflow.keras import Input, Model
from tensorflow.python.ops import nn


def get_model_DF(X, f_loss, f_metrics, f_activation_1, f_activation_2, f_activation_3, LR) :
    """
      Structure of DNN model.

      Parameters:
         
         X (1D array): Channel gain.
         f_loss: Loss function.
         f_metrics: List of metrics.
         f_activation1: First activation function for the output (alpha).
         f_activation2: Second activation function for the output (P_R).
         f_activation2: Third activation function for the output (P_S).
         LR : Learning rate.
    
      Returns:
      
        DNN model.
    """
    opt = tf.keras.optimizers.Adam(learning_rate = LR)
    inputs = Input(shape=(X.shape[1]))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output1 = Dense(1, activation=f_activation_1)(x)
    output2 = Dense(1, activation=f_activation_2)(x)
    output3 = Dense(1, activation=f_activation_3)(x)
    merged = tf.keras.layers.Concatenate()([output1, output2, output3])
    model = Model(inputs=inputs, outputs=[merged])
    model.compile(loss=f_loss, optimizer=opt, metrics=f_metrics)
    model.summary()
    return model

def layers_size(X, f_loss, f_metrics, f_activation_1, f_activation_2, f_activation_3, LR, n_hidden_layers):
    """
      DNN with specific number of hidden layers.

      Parameters:
         
         X (1D array): Channel gain.
         f_loss: Loss function.
         f_metrics: List of metrics.
         f_activation1: First activation function for the output (alpha).
         f_activation2: Second activation function for the output (P_R).
         f_activation2: Third activation function for the output (P_S).
         LR : Learning rate.
         n_hidden_layers: Number of hidden layers required.

    """
    opt = tf.keras.optimizers.Adam(learning_rate = LR)
    
    if n_hidden_layers == 4 : 
        inputs = Input(shape=(X.shape[1]))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        output1 = Dense(1, activation=f_activation_1)(x)
        output2 = Dense(1, activation=f_activation_2)(x)
        output3 = Dense(1, activation=f_activation_3)(x)
        merged = tf.keras.layers.Concatenate()([output1, output2, output3])
        model = Model(inputs=inputs, outputs=[merged])
        model.compile(loss=f_loss, optimizer=opt, metrics=f_metrics)
        model.summary()
        
    elif n_hidden_layers == 3 : 
        
        inputs = Input(shape=(X.shape[1]))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        output1 = Dense(1, activation=f_activation_1)(x)
        output2 = Dense(1, activation=f_activation_2)(x)
        output3 = Dense(1, activation=f_activation_3)(x)
        merged = tf.keras.layers.Concatenate()([output1, output2, output3])
        model = Model(inputs=inputs, outputs=[merged])
        model.compile(loss=f_loss, optimizer=opt, metrics=f_metrics)
        model.summary()
    elif n_hidden_layers == 2 : 
        inputs = Input(shape=(X.shape[1]))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        output1 = Dense(1, activation=f_activation_1)(x)
        output2 = Dense(1, activation=f_activation_2)(x)
        output3 = Dense(1, activation=f_activation_3)(x)
        merged = tf.keras.layers.Concatenate()([output1, output2, output3])
        model = Model(inputs=inputs, outputs=[merged])
        model.compile(loss=f_loss, optimizer=opt, metrics=f_metrics)
        model.summary()
    elif n_hidden_layers == 1 : 
     
        inputs = Input(shape=(X.shape[1]))
        x = Dense(128, activation='relu')(inputs)
        output1 = Dense(1, activation=f_activation_1)(x)
        output2 = Dense(1, activation=f_activation_2)(x)
        output3 = Dense(1, activation=f_activation_3)(x)# Access
        merged = tf.keras.layers.Concatenate()([output1, output2, output3])
        model = Model(inputs=inputs, outputs=[merged])
        model.compile(loss=f_loss, optimizer=opt, metrics=f_metrics)
        model.summary()
    return model

def neurons_size(X, f_loss, f_metrics, f_activation_1, f_activation_2, f_activation_3, LR, M):
    """
      DNN with specific number of neurons.

      Parameters:
         
         X (1D array): Channel gain.
         f_loss: Loss function.
         f_metrics: List of metrics.
         f_activation1: First activation function for the output (alpha).
         f_activation2: Second activation function for the output (P_R).
         f_activation2: Third activation function for the output (P_S).
         LR : Learning rate.
         n_hidden_layers: Number of hidden layers required.

    """
    opt = tf.keras.optimizers.Adam(learning_rate = LR)

    inputs = Input(shape=(X.shape[1]))
    x = Dense(M, activation='relu')(inputs)
    x = Dense(2*M, activation='relu')(x)
    x = Dense(2*M, activation='relu')(x)
    x = Dense(2*M, activation='relu')(x)
    output1 = Dense(1, activation=f_activation_1)(x)
    output2 = Dense(1, activation=f_activation_2)(x)
    output3 = Dense(1, activation=f_activation_3)(x)
    merged = tf.keras.layers.Concatenate()([output1, output2, output3])
    model = Model(inputs=inputs, outputs=[merged])
    model.compile(loss=f_loss, optimizer=opt, metrics=f_metrics)
    model.summary()
    return model



