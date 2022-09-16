import tensorflow as tf
from src.utils import *

def compute_loss(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S, W, Tau, P_P):
         

        # SNR1 : (Gsr*(1-alpha**2)*Ps**2)/(Gpr*Pp+1)
        
        SNR1 = tf.multiply(g_SR,(tf.multiply(tf.subtract(tf.constant(1,dtype=tf.float32), tf.pow(alpha, 2)), tf.pow(P_S, 2))))
        SNR1 = tf.divide(SNR1, tf.add(tf.multiply(g_PR, P_P), tf.constant(1,dtype=tf.float32)))

        # SNR2 : ((Gss*Ps**2+Grs*Pr**2)+2*(np.sqrt(Grs*Gss)*Alpha*Ps*Pr)) ==> L1+L2/Gps*Pp+1
        L1 = tf.add(tf.multiply(g_SS,tf.pow(P_S,2)),tf.multiply(g_RS,tf.pow(P_R,2)))
        L2 = tf.multiply(tf.constant(2,dtype=tf.float32),tf.multiply(tf.multiply(tf.sqrt(tf.multiply(g_RS,g_SS)),P_S),tf.multiply(alpha,P_R)))

        SNR2 = tf.add(L1,L2)
        SNR2= tf.divide(SNR2, tf.add(tf.multiply(g_PS, P_P),tf.constant(1,dtype=tf.float32)))

        SNR_opt = tf.minimum(SNR1, SNR2)
        ########### R_P ################
        SNR_P_max = tf.multiply(g_PP, P_P)

        K1 =  tf.add(tf.multiply(g_RP, tf.pow(P_R, 2)),tf.multiply(g_SP,tf.pow(P_S, 2)))

        K2 = tf.multiply(tf.constant(2,dtype=tf.float32),tf.multiply(tf.multiply(tf.sqrt(tf.multiply(g_SP, g_RP)),alpha),tf.multiply(P_S, P_R)))

        H2 = tf.add(K1, tf.add(K2 ,tf.constant(1,dtype=tf.float32)))

        SNR_P = tf.divide(SNR_P_max, H2)

        R_P =  tf.multiply(tf.constant(0.5, dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_P)))

        R_P_max = tf.multiply(tf.constant(0.5, dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_P_max)))

        ########### QoS ################

        # ((Gpp*Pp)/((1+(Gpp*Pp))**(1-tau)-1))-1 ==> (Gpp*Pp)/(R1) 
        A1 = tf.add(tf.constant(1, dtype=tf.float32),tf.multiply(g_PP,P_P))
        A1 = tf.pow(A1, tf.math.subtract(tf.constant(1, dtype=tf.float32),Tau))
        A1 = tf.math.subtract(A1,tf.constant(1, dtype=tf.float32))
        
        A_ = tf.subtract(tf.divide(tf.multiply(g_PP,P_P),A1),tf.constant(1, dtype=tf.float32))
   
        #Qos = tf.multiply(W,tf.keras.activations.relu(tf.subtract(R1,tf.multiply(tf.subtract(tf.constant(1,dtype=tf.float32),Tau),R1_max)))) 
        
        Qos = tf.add(tf.add(tf.multiply(g_SP,tf.pow(P_S,2)),tf.multiply(g_RP,tf.pow(P_R,2))),tf.multiply(tf.constant(2,dtype=tf.float32),tf.multiply(tf.sqrt(tf.multiply(g_SP,g_RP)),tf.multiply(P_S,tf.multiply(alpha,P_R)))))
    
        Qos = tf.subtract(Qos, A_)

        n_Qos = tf.multiply(W,tf.keras.activations.relu(Qos)) 
        
        R_S_opt =  tf.multiply(tf.constant(0.5, dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_opt)))
        
        return R_P, R_P_max, n_Qos, R_S_opt, SNR_opt, #Qos
    
def get_loss_data(G, y_hat):
    
    # index retrieval
    
    g_RP_indx, g_PP_indx, g_SR_indx, g_PR_indx, g_SS_indx, g_RS_indx, g_SP_indx, g_PS_indx  = [0], [1], [2], [3], [4], [5], [6], [7]
    alpha_indx, P_R_indx, P_S_indx  = [0], [1], [2]

    # tensors retrieval
    g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S = tf.gather(G, g_RP_indx, axis=1), tf.gather(G, g_PP_indx, axis=1), tf.gather(G, g_SR_indx, axis=1), tf.gather(G, g_PR_indx, axis=1), tf.gather(G, g_SS_indx, axis=1), tf.gather(G, g_RS_indx, axis=1), tf.gather(G, g_SP_indx, axis=1), tf.gather(G, g_PS_indx, axis=1), tf.gather(y_hat, alpha_indx, axis=1), tf.gather(y_hat, P_R_indx, axis=1), tf.gather(y_hat, P_S_indx, axis=1)

    return g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S
    
def loss_DF(W=10**(0.5), tau=0.25, P_P = 10.0):
    
    def DF_loss(G, y_hat):
            
        g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S = get_loss_data(G, y_hat)
        
        Lambda = tf.constant(W, dtype=tf.float32)  # ==> lambda 
        
        Tau = tf.constant(tau, dtype=tf.float32) # ==> Tau 

        P_P = tf.multiply(tf.ones(tf.shape(P_R), dtype=tf.dtypes.float32),10)

        R_P, R_P_max, Qos, R_S_opt, SNR = compute_loss(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S, Lambda, Tau, P_P)
        
        res = tf.reduce_mean(-R_S_opt+Qos) 
        
        return res
    return DF_loss


