from library import *
from parameters import *
from functions import *

# Drawn samples from the uniform distribution which represents the position of users
US, DS, UP, DP, R = np.zeros((Nbr,2)), np.zeros((Nbr,2)), np.zeros((Nbr,2)), np.zeros((Nbr,2)),  np.zeros((Nbr,2))

US[:,0], US[:,1] = np.random.uniform(p_min,p_max,Nbr), np.random.uniform(p_min,p_max,Nbr)
DS[:,0], DS[:,1] = np.random.uniform(p_min,p_max,Nbr), np.random.uniform(p_min,p_max,Nbr)
UP[:,0], UP[:,1] = np.random.uniform(p_min,p_max,Nbr), np.random.uniform(p_min,p_max,Nbr)
DP[:,0], DP[:,1] = np.random.uniform(p_min,p_max,Nbr), np.random.uniform(p_min,p_max,Nbr)
R[:,0], R[:,1] = 5, 5

# Calculate distance

# Distance Primary network
Dpp_E = calculateDistance(UP[:,0],UP[:,1],DP[:,0],DP[:,1])
Dpr_E = calculateDistance(UP[:,0],UP[:,1],R[:,0],R[:,1])
Drp_E = calculateDistance(R[:,0],R[:,1],DP[:,0],DP[:,1])
Dps_E = calculateDistance(UP[:,0],UP[:,1],DS[:,0],DS[:,1])

# Distance Secondary network

Dss_E = calculateDistance(US[:,0],US[:,1],DS[:,0],DS[:,1])
Dsr_E = calculateDistance(US[:,0],US[:,1],R[:,0],R[:,1])
Drs_E = calculateDistance(R[:,0],R[:,1],DS[:,0],DS[:,1])
Dsp_E = calculateDistance(US[:,0],US[:,1],DP[:,0],DP[:,1])

def gain_generator(d):
  """
      Channel gain model based on Rayleigh fading from reference [1].
      [1] : Ding, Zhiguo, et al. "On the performance of non-orthogonal multiple access in 5G systems with randomly deployed users." IEEE signal processing letters 21.12 (2014): 1501-1505.
      Args:
         d: Distance between each point.
      Returns:
        Channel gain between each users and relay.
  """    
  s = np.random.normal(mu, sigma, Nbr)
  h = s/np.sqrt(1+(d)**pathloss_factor) 
  h = h**2 # h/N_var division by the same noise variance N_var
  return h

#Noise_var = 10
def noisy_gain_generator(d, N_var):
  s = np.random.normal(mu, sigma, Nbr)
  h = s/np.sqrt(1+(d)**pathloss_factor) 
  
  h = h**2 # h/N_var division by the same noise variance N_var
  h = np.sqrt(h) + np.random.normal(mu, N_var, Nbr)  
  h = h**2  
  return h

def gain_generator_2(d): # anne model
  """
      Channel gain model from reference [2].
      [2] Savard, Anne, and E. Veronica Belmega. "Optimal power allocation in a relay-aided cognitive network." Proceedings of the 12th EAI International Conference on Performance Evaluation Methodologies and Tools. 2019.
      
      Args:
         d: Distance between each nodes.
      Returns:
        Channel gain between each users and relay.
  """      
  return (1/d**(3/2))**2

def uniform_gain():
  """
      Drawn samples from the uniform distribution represents channel gain.
      Args:
         None
      Returns:
        Gpp, Gpr, Grp, Gss, Gsr, Grs (represents different channel gain)
  """   
  # Exponential selection
  Gss =  np.random.uniform(v_min,v_max,Nbr)
  Gsr =  np.random.uniform(v_min,v_max,Nbr)
  Grs =  np.random.uniform(v_min,v_max,Nbr)
  Gpp =  np.random.uniform(v_min,v_max,Nbr)
  Gpr =  np.random.uniform(v_min,v_max,Nbr)
  Grp =  np.random.uniform(v_min,v_max,Nbr)
  Gsp =  np.random.uniform(v_min,v_max,Nbr)
  Gps =  np.random.uniform(v_min,v_max,Nbr)


  return Gpp, Gpr, Grp, Gss, Gsr, Grs, Gsp, Gps # Acces Added

def rician_fading():
    return rice.rvs(b, size=200000)

def nakagami_fading():
    return nakagami.rvs(nu, size=200000)

def gain_generator_3(d):
    return np.random.normal(mu,sigma)*np.sqrt(d**-pathloss_factor)

def add_noise(x, N_var):
    x = np.sqrt(x) + np.random.normal(mu, N_var, x.shape[0]) 
    x = x**2
    return x

def channel_type():
    ans=True
    while ans:
        print("""
        1. Our model channel gain [need ref]
        2. Uniform channel gain
        3. Anne channel gain [need ref]
        4. AWGN
        5. Rician fading 
        6. Nakagami fading
        7. Our model channel gain (Noise added)
        8.Exit/Quit
        """)
        ans=input("Select channel gain\n")
        if ans=="1":
          GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS = gain_generator(Dpp_E), gain_generator(Dpr_E), gain_generator(Drp_E), gain_generator(Dss_E), gain_generator(Dsr_E), gain_generator(Drs_E), gain_generator(Dsp_E), gain_generator(Dps_E) # Acces Added
          print("Channel gain created")
          return GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS
          ans = None
        elif ans=="2":
          GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS  = uniform_gain()
          print("Channel gain created")
          return GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS
          ans = None
        elif ans =='3':
          GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS = gain_generator_2(Dpp_E), gain_generator_2(Dpr_E), gain_generator_2(Drp_E), gain_generator_2(Dss_E), gain_generator_2(Dsr_E), gain_generator_2(Drs_E), gain_generator_2(Dsp_E), gain_generator_2(Dps_E) 
          print("Channel gain created")
          return GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS
          ans = None
        elif ans =='4':
            GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS = gain_generator_3(Dpp_E), gain_generator_3(Dpr_E), gain_generator_3(Drp_E), gain_generator_3(Dss_E), gain_generator_3(Dsr_E), gain_generator_3(Drs_E), gain_generator_3(Dsp_E), gain_generator_3(Dps_E) 
            print("Channel gain created")
            return GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS
            ans = None
        elif ans =='5':
            GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS = rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading() 
            print("Channel gain created")
            return GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS
            ans = None
        elif ans =='6':
            GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS = nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading()
            print("Channel gain created")
            return GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS
            ans = None
        elif ans =='7':
            Noise_variance = float(input('Enter noise variance \n'))
            GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS = noisy_gain_generator(Dpp_E, Noise_variance), noisy_gain_generator(Dpr_E, Noise_variance), noisy_gain_generator(Drp_E, Noise_variance), noisy_gain_generator(Dss_E, Noise_variance), noisy_gain_generator(Dsr_E, Noise_variance), noisy_gain_generator(Drs_E, Noise_variance), noisy_gain_generator(Dsp_E, Noise_variance), noisy_gain_generator(Dps_E, Noise_variance) 
            print("Channel gain created")
            return GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS
            ans = None
        elif ans =='8':
          ans = None
        else:
           print("Not Valid Choice Try again")