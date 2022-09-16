from src.utils import *

import numpy as np
from scipy.stats import rice, nakagami
import scipy


pos_min = 0 # min position in cell
pos_max = 10 # max position in cell


def distance(samples_nbr = int(2E6), pos_min = 0, pos_max = 10):
    U_P = np.random.uniform(pos_min, pos_max, (samples_nbr, 2))
    D_P = np.random.uniform(pos_min, pos_max, (samples_nbr, 2))
    U_S = np.random.uniform(pos_min, pos_max, (samples_nbr, 2))
    D_S = np.random.uniform(pos_min, pos_max, (samples_nbr, 2))
    R  = 5 * np.ones((samples_nbr, 2))

    # Calculate distance

    # Distance Primary network
    d_PP = np.linalg.norm(U_P - D_P, axis=1)
    d_PR = np.linalg.norm(U_P - R  , axis=1)
    d_RP = np.linalg.norm(R   - D_P, axis=1)
    d_PS = np.linalg.norm(U_P - D_S, axis=1)

    # Distance Secondary network
    d_SS = np.linalg.norm(U_S - D_S, axis=1)
    d_SR = np.linalg.norm(U_S - R  , axis=1)
    d_RS = np.linalg.norm(R   - D_S, axis=1)
    d_SP = np.linalg.norm(U_S - D_P, axis=1)
    
    return d_PP, d_PR, d_RP, d_PS, d_SS, d_SR, d_RS, d_SP

def channel_gain_with_gaussian_fading(d, mu=0.0, sigma=7, alpha=3):  # channel gain model
    """
      Channel gain model from [1].
      [1] : Ding, Z., Yang, Z., Fan, P., & Poor, H. V. (2014). On the performance of non-orthogonal multiple access in 5G systems with randomly deployed users. IEEE signal processing letters, 21(12), 1501-1505.
      Args:
         d: distance between source and destination
         mu : gaussian fading mean
         sigma: gaussian fading sigma
         alpha: is the path loss factor
      Returns:
        channel coefficient
    """    

    s = np.random.normal(mu, sigma, d.shape[0])
    h = s/np.sqrt(1.0 + np.power(d, alpha))
    #g = h**2
    return h


def channel_model_2(d): 
    """
      Channel gain model from reference [2].
      [2] : Savard, Anne, and E. Veronica Belmega. "Optimal power allocation in a relay-aided cognitive network." Proceedings of the 12th EAI International Conference on Performance Evaluation Methodologies and Tools. 2019.

      Args:
         d: distance between source and destination
      Returns:
        channel gain
    """      
    h = 1/d**(3/2) # g=(1/d**(3/2))**2
    return h

def uniform_gain(v_min = 10**-3, v_max = 10**3, samples_nbr = int(2E6)):
    """
    reference : https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    Drawn samples from the uniform distribution represents channel gain.
      Args:
         None
      Returns:
        channel gain
    """   
    # Exponential selection
    h_SS =  np.random.uniform(v_min, v_max, samples_nbr)
    h_SR =  np.random.uniform(v_min, v_max, samples_nbr)
    h_RS =  np.random.uniform(v_min, v_max, samples_nbr)
    h_PP =  np.random.uniform(v_min, v_max, samples_nbr)
    h_PR =  np.random.uniform(v_min, v_max, samples_nbr)
    h_RP =  np.random.uniform(v_min, v_max, samples_nbr)
    h_SP =  np.random.uniform(v_min, v_max, samples_nbr)
    h_PS =  np.random.uniform(v_min, v_max, samples_nbr)


    return Gpp, Gpr, Grp, Gss, Gsr, Grs, Gsp, Gps # Acces Added

def rician_fading(samples_nbr = int(2E6), b = 0.775):
    '''
    reference [4] : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rice.html
    '''
    return rice.rvs(b, size=samples_nbr)

def nakagami_fading(samples_nbr = int(2E6), nu = 4.97):
    '''
    reference [5] : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nakagami.html
    '''
    return nakagami.rvs(nu, size=samples_nbr)


def channel_type():
    ans=True
    while ans:
        print("""
        1. Channel with gaussian fading [1]
        2. Channel with Anne model [2]
        3. Channel with Uniform distribution[3] 
        4. Channel with Rician fading [4]
        5. Channel with Nakagami fading [5]
        6.Exit/Quit
        """)
        #4. AWGN 
        #5. Channel gain with Rician fading 
        #6. Channel gain with Nakagami fading
        #7. Noisy channel gain with gaussian fading
        ans=input("Select channel type\n")
        if ans=="1":
            d_PP, d_PR, d_RP, d_PS, d_SS, d_SR, d_RS, d_SP = distance()
            h_PP = channel_gain_with_gaussian_fading(d_PP)
            h_PS = channel_gain_with_gaussian_fading(d_PS)
            h_PR = channel_gain_with_gaussian_fading(d_PR)
            h_SP = channel_gain_with_gaussian_fading(d_SP)
            h_SS = channel_gain_with_gaussian_fading(d_SS)
            h_SR = channel_gain_with_gaussian_fading(d_SR)
            h_RP = channel_gain_with_gaussian_fading(d_RP)
            h_RS = channel_gain_with_gaussian_fading(d_RS)
    
            print("Channel created")
            return h_PP, h_PR, h_RP, h_SS, h_SR, h_RS, h_SP, h_PS
            ans = None

        elif ans =='2':
            d_PP, d_PR, d_RP, d_PS, d_SS, d_SR, d_RS, d_SP = distance()
            h_PP, h_PR, h_RP, h_SS, h_SR, g_RS, g_SP, g_PS = channel_model_2(d_PP), channel_model_2(d_PR), channel_model_2(d_RP), channel_model_2(d_SS), channel_model_2(d_SR), channel_model_2(d_RS), channel_model_2(d_SP), channel_model_2(d_PS) 
            print("Channel created")
            return h_PP, h_PR, h_RP, h_SS, h_SR, h_RS, h_SP, h_PS
            ans = None
            
        elif ans=="3":
            h_PP, h_PR, h_RP, h_SS, h_SR, h_RS, h_SP, h_PS  = uniform_gain() # or g
            print("Channel created")
            return h_PP, h_PR, h_RP, h_SS, h_SR, h_RS, h_SP, h_PS
            ans = None
        elif ans =='4':
            h_PP, h_PR, h_RP, h_SS, h_SR, h_RS, h_SP, h_PS = rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading() 
            print("Channel created")
            return h_PP, h_PR, h_RP, h_SS, h_SR, h_RS, h_SP, h_PS # or g
            ans = None
        elif ans =='5':
            h_PP, h_PR, h_RP, h_SS, h_SR, h_RS, h_SP, h_PS = nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading()
            print("Channel created")
            return h_PP, h_PR, h_RP, h_SS, h_SR, h_RS, h_SP, h_PS # or g
            ans = None
        elif ans =='6':
            ans = None
        else:
            print("Not Valid Choice Try again")
            
