import sys
import os
import tensorflow as tf
from tensorflow.python.ops import nn
from src.utils import *
from src.loss_function import *


def opportunistic_rate_DF(W=10**(0.5), tau=0.25, P_P = 10.0):
    def opportunistic_rate(G, y_out):
        """
          Metrics used on DL model for throughput calculation.
          This function will get those parameters as input
          G: Channel gain tensor.
          y_out: Predicted parameter.

          Parameters:
            Lambda : Penalty for QoS
            Tau : degradation factor for the primary network
          Returns:
            opportunistic rate mean 
        """
        g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S = get_loss_data(G, y_out)
        
        Lambda = tf.constant(W, dtype=tf.float32)  # ==> lambda 
        
        Tau = tf.constant(tau, dtype=tf.float32) # ==> Tau 

        P_P = tf.multiply(tf.ones(tf.shape(P_R), dtype=tf.dtypes.float32),10)

        R_P, R_P_max, Qos, R_S_opt, SNR = compute_loss(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S, Lambda, Tau, P_P)

        return R_S_opt
    return opportunistic_rate

def outage_DF(W=10**(0.5), tau=0.25, P_P = 10.0): 
    def outage(G, y_out): 
        #Primary_ARD_Percentage_DF
        """
          metrics used on DL model for testing Primary achievable rate degradation percentage .

          Parameters:
            G: Channel gain tensor.
            y_out: Predicted parameter.
          Returns:
            percentage of the empirical outage as the proportion of
samples in the dataset (or channel settings) for which the target
primary QoS constraint is not met 
        """

        g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S = get_loss_data(G, y_out)
        
        Lambda = tf.constant(W, dtype=tf.float32)  # ==> lambda 
        
        Tau = tf.constant(tau, dtype=tf.float32) # ==> Tau 

        P_P = tf.multiply(tf.ones(tf.shape(P_R), dtype=tf.dtypes.float32),10)

        R_P, R_P_max, Qos, R_S_opt, SNR = compute_loss(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S, Lambda, Tau, P_P)

        # 1 - ratio(Rp, Rp_)
        PR_D = tf.subtract(tf.constant(1,dtype=tf.float32), tf.divide(R_P, R_P_max))

        #ARD > tau  
        mask_PR_D = tf.greater(PR_D, Tau)# boolean tensor 
        
        return tf.multiply(tf.cast(mask_PR_D, tf.float32),tf.constant(100.0, dtype=tf.float32)) # return mask_PR_D
    return outage

def delta_DF(W=10**(0.5), tau=0.25, P_P = 10.0):
    def Delta(G, y_out):
 
        """
          Metrics used on DL model for Delta.

          Parameters:
            G: Channel gain tensor.
            y_out: Predicted parameter.
          Returns:
            Degradation of the primary achievable rate
    caused by the opportunistic interference
        """

        g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S = get_loss_data(G, y_out)
        
        Lambda = tf.constant(W, dtype=tf.float32)  # ==> lambda 
        
        Tau = tf.constant(tau, dtype=tf.float32) # ==> Tau 

        P_P = tf.multiply(tf.ones(tf.shape(P_R), dtype=tf.dtypes.float32),10)

        R_P, R_P_max, Qos, R_S_opt, SNR = compute_loss(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S, Lambda, Tau, P_P)


        # 1 - ratio(Rp, Rp_)
        PR_D = tf.subtract(tf.constant(1,dtype=tf.float32), tf.divide(R_P, R_P_max))

        return tf.multiply(PR_D,tf.constant(100.0, dtype=tf.float32))
    return Delta


def quality_of_service_violation_DF(W = 10**(0.5), tau=0.25, P_P=10.0, QoS_thresh = -5): 
    def V_Qos(G, y_out): 
        """
          metrics used on DL model for testing QoS viloation .
          Parameters:
            G: Channel gain tensor.
            y_out: Predicted parameter.
          Returns:
            Number of violated QoS 
        """

        g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S = get_loss_data(G, y_out)
       
        Lambda = tf.constant(W, dtype=tf.float32)  # ==> lambda 
        
        Tau = tf.constant(tau, dtype=tf.float32) # ==> Tau 

        P_P = tf.multiply(tf.ones(tf.shape(P_R), dtype=tf.dtypes.float32),10)
        
        R_P, R_P_max, Qos, R_S_opt, SNR = compute_loss(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S, Lambda, Tau, P_P)

        ########### QoS ################

        # function A' ==> A'(Gpp) : (Gpp*Pp)/((1+(Gpp*Pp))**(1-tau)-1)-1 ==> (Gpp*Pp)/(R1) 

        A_ = tf.subtract(tf.divide(tf.multiply(g_PP,P_P),R_P),tf.constant(1, dtype=tf.float32))

        #Qos = (Gsp*Ps**2+Grp*Pr**2+2*np.sqrt(Gsp*Grp)*Alpha*Ps*Pr)-A_/A_

        Qos = tf.add(tf.add(tf.multiply(g_SP,tf.pow(P_S,2)),tf.multiply(g_RP,tf.pow(P_R,2))), tf.multiply(tf.constant(2,dtype=tf.float32),(tf.multiply(tf.multiply(tf.sqrt(tf.multiply(g_SP,g_RP)),P_S),tf.multiply(alpha,P_R)))))
        Qos = tf.subtract(Qos, A_)
        #n_Qos = tf.divide(Qos, A_) # Normalization

        #Qos > 10**-5  
        mask_pr = tf.greater(Qos,tf.math.pow(tf.constant(10, dtype=tf.float32), QoS_thresh))# boolean array 

        return mask_pr
    return V_Qos



def delta_out_DF(W=10**(0.5), tau=0.25, P_P = 10.0):
    def delta_out(G, y_out):
 
        """
          Metrics used on DL model for Delta.

          Parameters:
            G: Channel gain tensor.
            y_out: Predicted parameter.
          Returns:
            Degradation of the primary achievable rate
    caused by the opportunistic interference
        """

        g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S = get_loss_data(G, y_out)
        
        Lambda = tf.constant(W, dtype=tf.float32)  # ==> lambda 
        
        Tau = tf.constant(tau, dtype=tf.float32) # ==> Tau 

        P_P = tf.multiply(tf.ones(tf.shape(P_R), dtype=tf.dtypes.float32),10)

        R_P, R_P_max, Qos, R_S_opt, SNR = compute_loss(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S, Lambda, Tau, P_P)


        # 1 - ratio(Rp, Rp_)
        PR_D = tf.subtract(tf.constant(1,dtype=tf.float32), tf.divide(R_P, R_P_max))
        
        d_out = tf.boolean_mask(PR_D, PR_D > Tau)
        d_out = tf.reduce_mean(d_out)
        
        return tf.multiply(d_out,tf.constant(100.0, dtype=tf.float32))
    return delta_out






