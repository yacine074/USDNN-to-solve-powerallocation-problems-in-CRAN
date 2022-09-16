"""
  this file contains the different useful functions.

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn
from multiprocessing import Pool

# FDFR, FDF2 ==> F_R, F_S


def A_(g_PP, P_P=10.0, tau=0.25):
    """
      Compute the term (A) in the QoS constraint.
      Parameters:
         g_PP: 1D Array containing channel gain between primary transmitter and primary receiver.
      Returns:
         A (float): result of A.
    """  
    return ((g_PP*P_P)/((1+(g_PP*P_P))**(1-tau)-1))-1
    
def C(x):
    """
      Shannon capacity function.
      Parameters:
         x: Signal-to-noise-ratio.
      Returns:
         capacity (float): shanon capacity.
    """   
    capacity = (1/2*np.log2(1+x))
    return capacity


def F_R(alpha, P_S, g_SR, g_PR, P_P=10.0):
    """
      Compute first secondary SNR for DF.

      Parameters:
         alpha: 1D Array containing alpha values.
         P_S: 1D Array containing power of secondary network values.
         g_SR: 1D Array channel gain between secondary transmitter and relay.
         g_PR: 1D Array channel gain between primary transmitter and relay.
         P_P: Primary power.

      Returns:
         X (float): First SNR for Decode-and-Forward.
    """
    X = (g_SR*(1-alpha**2)*P_S**2)/(g_PR*P_P+1)
    return X

#def FDF2(alpha, P_S, P_R, g_SS, g_RS, g_PS, P_P=10.0):

def F_S(alpha, P_R, P_S, g_SS, g_RS, g_PS, P_P=10.0):
    """
      This function calculate the different parameters using Gekko.

      Parameters:
         alpha: 1D Array containing alpha values.
         P_S: 1D Array containing Power of secondary network values.
         Pr: 1D Array containing Power of relay.
         g_SS: 1D Array channel gain between secondary transmitter and secondary receiver.
         g_RS: 1D Array channel gain between relay and secondary receiver.
         g_PS: 1D Array channel gain between relay and secondary receiver.
         P_P: Primary power.

      Returns:
         X (float): Second SNR for Decode-and-Forward.
    """
    X = ((g_SS*P_S**2+g_RS*P_R**2)+2*(np.sqrt(g_RS*g_SS)*alpha*P_S*P_R))/(g_PS*P_P+1) 
    return X

def BF_A_squeeze(x):
    return exhaustive_search(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])

def benchmark_generator(H): 

    with Pool() as p:
        BF_res =  p.map(BF_A_squeeze, H)

    return np.squeeze(np.asarray(BF_res, dtype="float64"))

def data_filter(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS):
    s = 10**(-40/10)/10
    mask = np.all(np.stack([g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS], axis=1)>=s, axis=1)
    
    return g_RP[mask], g_PP[mask], g_SR[mask], g_PR[mask], g_SS[mask], g_RS[mask], g_SP[mask], g_PS[mask]

def history_extraction(W, key, root_dir="DNN"):
    """
        Parameters:
          W (Lambda): dictionary that contains the values of lambda as key (str) and values (int).
          key : str used to extract the history (loss or val_loss, outage, val_outage...)
        Returns:
          training history of loss or secondary_rate or primary rate degradation
    """

    
    temp, data = [], []
    
    for ld_k in W.keys():
  
        history = np.load(root_dir+'/lambda = '+ld_k+'/history/'+ld_k+'.npy',allow_pickle='TRUE').item()

        temp.append(history[key])

        data.append(temp)
        temp  = []
    return data


#------------ tensorflow functions for DNN ------------# 

def custom_sigmoid(x):
    """
    Modified sigmoid function used for handling predicted powers.

    Parameters:
      x: tensor.
    Returns:
      Output of sigmoid function range between 0 and sqrt(10)
    """
    output = tf.multiply(tf.sqrt(tf.constant(10,dtype=tf.float32)),nn.sigmoid(x))
    # Cache the logits to use for crossentropy loss.
    output._keras_logits = x  # pylint: disable=protected-access
    return output


def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=tf.float32))
    return numerator / denominator




