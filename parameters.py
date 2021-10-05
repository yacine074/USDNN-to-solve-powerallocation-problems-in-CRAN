from packages import *

#initialization of parameters
Pp = 10.0 # primary network power
Pr_max, Ps_max = 10.0, 10.0 # relay power, secondary network power

Alpha_tilde = math.sqrt(1)
Psmax_tilde = math.sqrt(Ps_max)
Prmax_tilde = math.sqrt(Pr_max)

pathloss_factor = 3

tau = 0.25

Nbr = 2000000# Number of configuration

p_min = 0 # min position in cell
p_max = 10 # max position in cell

v_min = 10**-3 # min channel value gain for uniform distribution
v_max = 10**3  # max channel value gain for uniform distribution 

mu, sigma = 0, 7 # mean and standard deviation for data generation

QoS_thresh = -5 # QoS_threshold

#N_var = 5 # Noise variance (Recently added)


b = 0.775 # rice
nu = 4.97 # nakagami

Noise_variance = 0.1

root_dir ='DNN'
