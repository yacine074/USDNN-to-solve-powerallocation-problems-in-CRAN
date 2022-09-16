import sys
import os
import numpy as np
from src.utils import *

#from gekko import GEKKO

def exhaustive_search(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, P_R_max = 10.0, P_S_max = 10.0, P_P = 10.0):
    ''' Bruteforce with QoS constraint'''
    alpha = np.linspace(0, np.sqrt(1.0), 100)
    P_R = np.linspace(0, np.sqrt(P_R_max), 100)
    P_S = np.linspace(0, np.sqrt(P_S_max), 100)


    A,B,C = np.meshgrid(alpha, P_R, P_S)
   
    # if QoS constraint respected
    mask = (((g_SP*C**2)+(g_RP*B**2))+2*(np.sqrt(g_SP*g_RP)*A*C*B)) <= A_(g_PP)

    
    A = A[mask]
    B = B[mask]
    C = C[mask]
    
    SNR1 = F_R(A, C, g_SR, g_PR) # F_R and F_S
    SNR2 = F_S(A, B, C, g_SS, g_RS, g_PS, P_P)
    SNR = np.minimum(SNR1,SNR2)

    ind = np.argmax(SNR)

    SNR_opt, alpha_opt, P_R_opt, P_S_opt = SNR[ind], A[ind], B[ind], C[ind]
    
    c = lambda t: (1/2*np.log2(1+t))
    c_func = np.vectorize(c)
    
    return np.array([[g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, c_func(SNR_opt), alpha_opt**2, P_R_opt**2, P_S_opt**2]])

def BF_A_W_qos(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, P_P=10.0, P_R_max=10.0, P_S_max=10.0):
    
    ''' Bruteforce without QoS constraint'''

    alpha = np.linspace(0, np.sqrt(1.0), 100)
    P_R = np.linspace(0, np.sqrt(P_R_max), 100)
    P_S = np.linspace(0, np.sqrt(P_S_max), 100)

    A,B,C = np.meshgrid(alpha, P_R, P_S)
  
    
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    
    SNR1 = F_R(A, C, g_SR, g_PR) 
    SNR2 = F_S(A, B, C, g_SS, g_RS, g_PS, P_P)
    SNR = np.minimum(SNR1, SNR2)

    ind = np.argmax(SNR)

    SNR_opt, alpha_opt, P_R_opt, P_S_opt = SNR[ind], A[ind], B[ind], C[ind]
    
    c = lambda t: (1/2*np.log2(1+t))
    c_func = np.vectorize(c)
    
    return np.array([[g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, c_func(SNR_opt), alpha_opt**2, P_R_opt**2, P_S_opt**2]])

def SA_MDB_DF(g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, P_P=10.0, P_R_max=10.0, P_S_max=10.0):
    ''' Analyical solution without QoS constraint'''
    # Creation of a numpy array of the same dimension as GRP containing Ps_max
    P_S = np.sqrt(P_S_max)*np.ones(g_RP.shape)
    P_R = np.sqrt(P_R_max)*np.ones(g_RP.shape)

    # SNR1 et SNR2

    F1 = F_R(0, P_S, g_SR, g_PR)
    F2 = F_S(0, P_R, P_S, g_SS, g_RS, g_PS, P_P)
    # Create an empty array to store the optimal_alpha and optimal SNR values
    alpha_opt = np.zeros(g_RP.shape)
    SNR_opt = np.zeros(g_RP.shape)

    mask = F1 <= F2

    # if F1 <= F2 == False, F2 <= F1==True, bool table
    nmask = np.logical_not(mask)

    SNR_opt[mask] = FDFR(0, P_S[mask], g_SR[mask], g_PR[mask]) 
    a = (g_SR[nmask]*P_S[nmask]**2*(g_PS[nmask]*P_P+1))
    b = (2*np.sqrt(g_RS[nmask]*g_SS[nmask])*P_S[nmask]*P_R[nmask]*(g_PR[nmask]*P_P+1))

    delta = ((4*g_RS[nmask]*g_SS[nmask]*P_S[nmask]**2*P_R[nmask]**2*(g_PR[nmask]*P_P+1)**2)+\
    (4*g_SR[nmask]**2*P_S[nmask]**4*(g_PS[nmask]*P_P+1)**2)-\
    (4*g_SR[nmask]*P_S[nmask]**2*(g_PS[nmask]*P_P+1)*(g_SS[nmask]*P_S[nmask]**2+g_RS[nmask]*P_R[nmask]**2)*(g_PR[nmask]*P_P+1)))
    alpha_opt[nmask] = (-b+np.sqrt(delta))/(2*a)

    #F1 is equal to F2
    #SNR_opt[nmask] = FDFR_BF(alphaOpt[nmask], Ps[nmask], Gsr[nmask])
    SNR_opt[nmask] = FDF2(alpha_opt[nmask], P_S[nmask], P_R[nmask], g_SS[nmask], g_RS[nmask], g_PS[nmask], P_P)

    res_analytique = np.stack((g_RP, g_PP, g_SR, g_PR, g_SS, g_RS, g_SP, g_PS, C(SNR_opt), alpha_opt**2, P_R**2, P_S**2), axis=1)

    return  res_analytique    


'''    
def Gekko_A(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Pp = 10.0) :

  m = GEKKO(remote=True)
  # Bounds for the constraints and initialization of the variables Alpha, Pr, Ps
  Alpha = m.Var(0.5,lb=0,ub=Alpha_tilde)
  Ps    = m.Var(1.5,lb=0,ub=Psmax_tilde)
  Pr    = m.Var(0,lb=0,ub=Prmax_tilde)
 
 # QoS constraint
 
  m.Equation(Gsp*Ps**2+Grp*Pr**2+2*(np.sqrt(Gsp*Grp)*Alpha*Ps*Pr) <= A_(Gpp)) 
  
  # Calling the two SNR 

  Func_FDFR = FDFR(Alpha, Ps, Gsr, Gpr)
  Func_FDF2 = FDF2(Alpha, Pr, Ps, Gss, Grs, Gps, Pp)#FDF2(Alpha, Ps, Pr, Gss, Grs, Gps, Pp)
  Z = m.Var()
  # max(min(SNR1,SNR2))
  m.Maximize(Z)
  m.Equation(Z<=Func_FDFR)
  m.Equation(Z<=Func_FDF2)
  m.solve(disp=False)    # solve
  c = lambda t: (1/2*np.log2(1+t))
  c_func = np.vectorize(c)
  res_Gekko = np.array([[Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps,c_func(Z.value[0]),alpha.value[0]**2,Pr.value[0]**2,Ps.value[0]**2]])

  return  res_Gekko
'''    


