import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from src.utils import *
from src.loss_function import *
from src.DNN_model import *
from src.DNN_metrics import *

### ### ### ### ### ### ### ### Debit metrics ### ### ### ### ### ### ### ### 

def secondary_rate(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, alpha, P_R, P_S, P_P=10.0) : 
    """
    Function for opportunstic users rate calculation.

    Parameters:
      Grp: 1D Array containing Alpha values.
      Gpp: 1D Array containing gain between primary transmitter and primary receiver.
      Gsr: 1D Array containing gain between secondary transmitter and relay.
      Gpr: 1D Array containing gain between primary transmitter and relay.
      Gss: 1D Array containing gain between secondary transmitter and secondary receiver.
      Grs: 1D Array containing gain between relay and secondary receiver.
      Gsp: 1D Array containing gain between secondary transmitter and primary receiver.
      Gps: 1D Array containing gain between secondary transmitter and primary receiver.
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.

    Returns:
       opportunstic users rate
    """

    R_S = np.zeros(g_RP.shape)

    R_S = np.minimum(C(F_R(alpha, P_S, g_SR, g_PR)),C(F_S(alpha, P_R, P_S, g_SS, g_RS, g_PS, P_P)))
    
    return R_S


def avreage_gap(X, Y):
    """avreage gap between the predicted secondary rate and the obtained rate via bruteforce"""

    return np.mean(X) - np.mean(Y)

def relative_avreage_gap(X, Y):
    """relative avreage gap between the predicted secondary rate and the obtained rate via bruteforce"""

    return (np.mean(X) - np.mean(Y))/(np.mean(Y))


def primary_rate_degradation(g_RP, g_PP, g_SP, alpha, P_R, P_S, P_P =10.0):
    """
      Parameters:
         g_RP: 1D Array containing channel coefficient between relay and primary receiver.
         g_PP: 1D Array containing channel coefficient between primary transmitter and primary receiver.
         g_SP: 1D Array containing gain between secondary transmitter and primary receiver.
         alpha: 1D Array containing alpha values.
         P_R: 1D Array containing power of relay.
         P_S: 1D Array containing power of secondary network.
      
      Returns:
         primary rate degradation .

    """
    R_P = C((g_PP*P_P)/(g_RP*P_R**2+g_SP*P_S**2+2*(np.sqrt(g_SP*g_RP)*P_S*P_R*alpha)+1))
    R_P_max = C(g_PP*P_P)

    #res = 1-(R_P/R_P_max)
    res = np.nan_to_num((R_P_max-R_P)/R_P_max)
    return res

def primary_stats(g_RP, g_PP, g_SP, alpha, P_R, P_S, P_P = 10.0, tau = 0.25):
    
    """ Mean of Delta, Max of Delta, mean of outage, outage """

    res = primary_rate_degradation(g_RP, g_PP, g_SP, alpha, P_R, P_S)

    mean_res = np.nanmean(res) # Mean of Delta 
    max_res = np.max(res) # Max of Delta
    #res =  # Outage
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            mean_outage = np.nanmean(res[res>tau])
        except RuntimeWarning:
            mean_outage = 0

    outage = np.nanmean(res>tau)

    return mean_res*100, max_res*100, mean_outage*100, outage*100

def secondary_users_stats(X, Y, W, LR, P_P=10.0, tau = 0.25, root_dir="DNN"):
   
    rate_gap, rate_gap_temp = [], []
    
    for ld_k in W.keys():
        
        for lr_k in LR.keys():
            
            model = tf.keras.models.load_model(root_dir+'/lambda = '+ld_k+'/weights/'+ld_k+'.h5', custom_objects={'DF_loss':loss_DF(W,tau),'opportunistic_rate':opportunistic_rate_DF(W, tau, P_P), 'outage':outage_DF(W, tau, P_P),'Delta': delta_DF(W, tau, P_P), 'delta_out':delta_out_DF(W, tau, P_P), 'V_Qos':quality_of_service_violation_DF(tau=0.25),"custom_sigmoid":custom_sigmoid})
            
            predictions = model.predict(X)

            secondary_rate_hat = secondary_rate(X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], X[:,5], X[:,6], X[:,7], predictions[:,0], predictions[:,1], predictions[:,2])
            
            secondary_rate_true = Y[:,0]
            
            rate_gap_temp.append(relative_avreage_gap(secondary_rate_hat, secondary_rate_true))
        
        rate_gap.append(rate_gap_temp)
        
        rate_gap_temp = []
 
    return np.asarray(rate_gap)*100


def primary_users_stats(X, Y, W, LR, tau = 0.25, P_P=10.0, root_dir="DNN"):
    
    P_stats, temp_stats = [], []
    
    for ld_k in W.keys():
        
        for lr_k in LR.keys():         

            model = tf.keras.models.load_model(root_dir+'/lambda = '+ld_k+'/weights/'+ld_k+'.h5', custom_objects={'DF_loss':loss_DF(W,tau),'opportunistic_rate':opportunistic_rate_DF(W, tau, P_P), 'outage':outage_DF(W, tau, P_P),'Delta': delta_DF(W, tau, P_P), 'delta_out':delta_out_DF(W, tau, P_P), 'V_Qos':quality_of_service_violation_DF(tau=0.25),"custom_sigmoid":custom_sigmoid})
      
            predictions = model.predict(X)

            mean_delta, max_delta, delta_out, outage  = primary_stats(X[:,0], X[:,1], X[:,6], predictions[:,0], predictions[:,1], predictions[:,2])
            
            temp_stats.append(np.stack([[mean_delta, max_delta, delta_out, outage]],axis=1))
            
        P_stats.append(temp_stats)
        
        temp_stats = []
    
    return np.asarray(P_stats)

def stats(X, y_GT, y_hat):

    rate_hat = secondary_rate(X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], X[:,5], X[:,6], X[:,7], y_hat[:,0], y_hat[:,1], y_hat[:,2])

    rate_gap = relative_avreage_gap(rate_hat, y_GT[:,0]) 
    
    #v_tau = tau_violation_percentage(X[:,0], X[:,1], X[:,6], y_hat[:,0], y_hat[:,1], y_hat[:,2])
    
    P_stats = primary_stats(X[:,0], X[:,1], X[:,6], y_hat[:,0], y_hat[:,1], y_hat[:,2])
    
    return np.asarray([rate_gap]), np.asarray([P_stats])[:,0], np.asarray([P_stats])[:,1], np.asarray([P_stats])[:,2], np.asarray([P_stats])[:,3]






#-------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------- plot -----------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#

def train_evaluation(data, val_data, ylab, x_lim, y_lim, Lambda, filename):
    """
      Parameters:
         data: 1D array contains learning data history.
         data: 1D array contains learning data validation history.
         ylab: y label.
         x_lim: set the x limits of the current axes.
         y_lim : set the y-limits of the current axes.
         Lambda : dictionary that contains the values of lambda as key (str) and values (int)  
         filename : path to store the generated figure.
      Returns:
         Plot for achievable rate, Loss, primary rate degradation and QoS violation evolution.
    """

    #sns.set(style='white')
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.grid()

    m_train = ['D','o']
    m_val = ['o','P']
    ls_train = ['solid','dotted']
    m_color = ['plum', 'wheat'] 
    fig = plt.figure(1)

    for i in range (0, len(Lambda)) :
        plt.plot(data[i][0][:], label = r'Training set ($\lambda ='+list(Lambda.keys())[i].replace('_','^{')+'}$)',ls = ls_train[i], lw = 1
      , markerfacecolor = m_color[i], dash_capstyle = 'round', color = 'black', marker = m_train[i], markersize = 9, markevery = 50)
        plt.plot(val_data[i][0][:], label = 'Validation set ($\lambda = '+list(Lambda.keys())[i].replace('_','^{')+'}$)', ls='-.', lw = 0.4, marker = m_val[i], markerfacecolor = m_color[i], dash_capstyle = 'round', color = 'black', markersize = 9, markevery = 50)

        plt.xlabel("Epochs", fontsize= 20)
        plt.grid()
        plt.ylabel(ylab, fontsize= 20)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        lgd = plt.legend(loc='best', fontsize= 16)#title="Learning rate"

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    plt.show()
    plt.ion()
    fig.savefig(filename+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

def test_evaluation(rate_gap, mean_delta, max_delta, delta_out, outage):
    
    # First part : Average and maximum primary rate degradation and average degradation when in outage (∆out) as functions of λ over the test set.
    
    references = np.round(np.array([10**-1,10**-0.75,10**-0.5,10**-0.25,10**0,10**0.25,10**0.5,10**0.75,10**1,10**1.25,10**1.5,10**1.75,10**2]), decimals=4)
       
    plt.rcParams["figure.figsize"] = (10,5)
    #sns.set(style='white')
    fig = plt.figure(1)

    plt.grid()
    xs = np.linspace(1, 21, 100)

    plt.hlines(y=25, xmin=0, xmax=len(xs), colors='black', linestyles='--', lw=2, label=r'$\tau = 25\%$')

    plt.plot(references, mean_delta, label = 'Average', marker='8',markersize=9)

    plt.plot(references, max_delta, label= 'Max', marker='^',markersize=9)

    plt.plot(references, delta_out, label= r'$\Delta_{out}$', marker='s',markersize=9)

    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Hyperparameter $\lambda$',fontsize= 20)
    plt.ylabel('Primary network degradation (%)',fontsize= 20)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    #plt.xticks(references,['$10^{-1}$', '$10^{-0.75}$','$10^{-0.5}$', '$10^{-0.25}$','$10^{0}$', '$10^{0.25}$','$10^{0.5}$', '$10^{0.75}$','$10^{1}$','$10^{1.25}$','$10^{1.5}$','$10^{1.75}$','$10^{2}$'])
    plt.legend(loc = 'best', fontsize= 16)
    fig.savefig('Primary_network_degradation_stats.pdf', bbox_inches='tight')
    plt.show()

    plt.close()
    
    # Second part : Plot G and Outage
    
    plt.rcParams["figure.figsize"] = (10,5)
    #sns.set(style='white')
    fig = plt.figure(1)
    plt.grid()
    
    xs = np.linspace(1, 21, 100)

    plt.plot(references, rate_gap, label='Relative average gap', marker='^',markersize=9)
    plt.plot(references, outage, label = 'Outage', marker='H',markersize=9)

    plt.ylabel('Percentage', fontsize= 20)
    plt.xscale('log')
    plt.xlabel(r'Hyperparameter $\lambda$',fontsize= 20)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 

    #plt.xticks(A,['$10^{-1}$', '$10^{-0.75}$','$10^{-0.5}$', '$10^{-0.25}$','$10^{0}$', '$10^{0.25}$','$10^{0.5}$', '$10^{0.75}$','$10^{1}$','$10^{1.25}$','$10^{1.5}$','$10^{1.75}$','$10^{2}$'])
    plt.legend(loc = 'best', fontsize= 16)
    fig.savefig('G_and_Outage.pdf', bbox_inches='tight')
    plt.show()




def histogram(g_RP, g_PP, g_SP, alpha, P_R, P_S, alpha_2, P_R_2, P_S_2):
    """
    Parameters:
      g_RP: channel coefficient between relay and primary receiver.
      g_PP: channel coefficient between primary transmitter and primary receiver. 
      g_SP: channel coefficient between secondary transmitter and primary receiver.
      alpha: Array containing alpha values.
      P_R: Array containing Power of relay.
      P_S: Array containing Power of secondary network.
    Returns:
      histogram for primary rate degradation 
    """
    #sns.set(style='white')
    plt.rcParams["figure.figsize"] = (10,5)

    res = primary_rate_degradation(g_RP, g_PP, g_SP , alpha, P_R, P_S)*100
    
    res2 = primary_rate_degradation(g_RP, g_PP, g_SP , alpha_2, P_R_2, P_S_2)*100

    #fig, ax = plt.subplots(1) # Creates figure fig and add an axes, ax.
    xs = np.linspace(1, 21, 10**5)

    plt.vlines(x=25, ymin=0, ymax=len(xs), colors='black', linestyles='--', lw=2, label=r'$\tau = 25\%$')


    plt.hist(res, 100, histtype='step', ls=':', lw = 2 , color='red',label='$\lambda = 10^{0.5}$')
    plt.hist(res2, 100, histtype='step', ls='-',  lw = 2, label='$\lambda = 10^{2}$')

    fig = plt.figure(1)
    #plt.xlim((-1,40))
    plt.grid()
    plt.yscale('log')
    plt.xlabel('Primary acheivable rate degradation ($\%$)', fontsize= 20)
    plt.ylabel('Samples', fontsize= 20)
    plt.legend(loc='best')
    #plt.annotate(r"$\lambda$ = "+'$'+Lambda_value.replace('_','^{')+'}$', xy=(0.05,0.9),xycoords='axes fraction',
    #           fontsize=14)

    lgd = plt.legend(
        fancybox=True, shadow=True, fontsize= 16, bbox_to_anchor=(0.6,0.5))#title="Learning rate"



    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 

    fig.savefig('histogram''.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.ion()
    plt.pause(1)
    plt.close()  
    


