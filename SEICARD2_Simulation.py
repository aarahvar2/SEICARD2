# ------------------------------------------------------------------------------------------------------------------#
__author__ = "Amir Reza Alizd-Rahvar"
__version__ = "1.0.0"
__email__ = "alizad@ipm.ir"
__status__ = "Final"

######################################################################
# If you use Visual Studio, use CTRL+F5 for running of the code
######################################################################

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)
######################### Strain 1 parameters #################################
beta1 = 0.2  #R0^1 = 2
p1_dic = {'incubation_time_inv':1/5,
           'beta_A':beta1,'beta_I':beta1,'beta_C':beta1,
           'g_ar':1/10,'g_ir':1/10,'g_cr':1/10,'g_cd':1/10,
           'A_frac':0.1,'C_frac':0.1,'D_frac':0.05}


######################### Strain 2 parameters #################################

beta2 = 0.27  #Set beta2 to 0.13, 0.2, and 0.27 to have R0^2=1.3, 2, and 2.7, respectively
p2_dic = {'incubation_time_inv':1/5,
           'beta_A':beta2,'beta_I':beta2,'beta_C':beta2,
           'g_ar':1/10,'g_ir':1/10,'g_cr':1/10,'g_cd':1/10,
           'A_frac':0.4,'C_frac':0.1,'D_frac':0.05}



t_advent_arr = np.array(900)
t_advent_arr = np.append(t_advent_arr, np.arange(100,-10,-10))

p1 = convert(p1_dic)
p2 = convert(p2_dic)

R0_1 = p1.A_frac*p1.beta_A/p1.g_ar + (1-p1.A_frac)*((1-p1.D_frac)*p1.C_frac*p1.beta_C/p1.g_cr+(1-p1.C_frac)*p1.beta_I/p1.g_ir)
R0_2 = p2.A_frac*p2.beta_A/p2.g_ar + (1-p2.A_frac)*((1-p2.D_frac)*p2.C_frac*p2.beta_C/p2.g_cr+(1-p2.C_frac)*p2.beta_I/p2.g_ir)

# Total population, N.
N = 5000

# A grid of time points (in days)
t = [0 , 300]


# Initial number of infected and recovered individuals, I0 and R0.
E_1_0, E_2_0 = 1, 1 
A_1_0, A_2_0 = 0, 0
I_1_0, I_2_0 = 0, 0
C_1_0, C_2_0 = 0, 0
D_1_0, D_2_0 = 0, 0
R_1_0, R_2_0 = 0, 0



# The SIR model differential equations.
def ode(t,y, N,p1,p2,t_advent):
    
    #Check if strain 2 has emerged?
    a =  advent_new_Strain(t,y, N,p1,p2,t_advent)

    S,E_1, A_1, I_1, C_1, R_1, D_1, E_2, A_2, I_2, C_2, R_2, D_2 = y
    

    dS_dt  =-(  (p1.beta_A*A_1+p1.beta_I*I_1+p1.beta_C*C_1)+
              a*(p2.beta_A*A_2+p2.beta_I*I_2+p2.beta_C*C_2))*S / N

    ###################################################################################################
    #                                       STRAIN 1
    ###################################################################################################

    dE_1dt =(p1.beta_A*A_1+p1.beta_I*I_1+p1.beta_C*C_1)*S / N- p1.incubation_time_inv * E_1

    dA_1dt = p1.A_frac * p1.incubation_time_inv * E_1 - p1.g_ar * A_1

    dI_1dt = (1-p1.A_frac) * (1-p1.C_frac) * p1.incubation_time_inv * E_1 - p1.g_ir * I_1

    dC_1dt = (1-p1.A_frac) * p1.C_frac * p1.incubation_time_inv * E_1 - \
        ((1-p1.D_frac) * p1.g_cr + p1.D_frac * p1.g_cd) * C_1

    dR_1dt = (p1.g_ar * A_1) + (p1.g_ir * I_1) + ((1-p1.D_frac) * p1.g_cr * C_1)

    dD_1dt = p1.D_frac * p1.g_cd * C_1


    ###################################################################################################
    #                                       STRAIN 2
    ###################################################################################################
    

    if a==0:
        #Strain 2 has not emerged yet (set all the parameters of its ODE to zero)
        dE_2dt = 0
        dA_2dt = 0
        dI_2dt = 0
        dC_2dt = 0
        dR_2dt = 0
        dD_2dt = 0

    else:
        #Strain 2 has emerged
        dE_2dt =(p2.beta_A*A_2+p2.beta_I*I_2+p2.beta_C*C_2)*S / N- p2.incubation_time_inv * E_2

        dA_2dt = p2.A_frac * p2.incubation_time_inv * E_2 - p2.g_ar * A_2

        dI_2dt = (1-p2.A_frac) * (1-p2.C_frac) * p2.incubation_time_inv * E_2 - p2.g_ir * I_2

        dC_2dt = (1-p2.A_frac) * p2.C_frac * p2.incubation_time_inv * E_2 - \
            ((1-p2.D_frac) * p2.g_cr + p2.D_frac * p2.g_cd) * C_2

        dR_2dt = (p2.g_ar * A_2) + (p2.g_ir * I_2) + ((1-p2.D_frac) * p2.g_cr * C_2)

        dD_2dt = p2.D_frac * p2.g_cd * C_2

     
    
    return dS_dt, dE_1dt, dA_1dt, dI_1dt,dC_1dt,dR_1dt,dD_1dt,dE_2dt,dA_2dt,dI_2dt,dC_2dt,dR_2dt,dD_2dt


def advent_new_Strain(t,y, N,p1,p2,t_advent):
    #This function return 0 and 1 before and after the advent of the new strain
    if t<t_advent:
        return 0
    else:
        return 1


def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[0]+5, ax.get_ylim()[1]*scale

def solve_SEICARD(t_advent):
    print(f't_advent = {t_advent}')

    a =  advent_new_Strain(0,0, N,p1,p2,t_advent)
    S_0 = N - E_1_0 - A_1_0 - I_1_0 - C_1_0 - R_1_0 - a*E_2_0 - a*A_2_0 - a*I_2_0 - a*C_2_0 - a*R_2_0


    # Initial conditions vector
    y0 = S_0, E_1_0, A_1_0, I_1_0, C_1_0, R_1_0, D_1_0, E_2_0, A_2_0, I_2_0, C_2_0, R_2_0, D_2_0

    sol = solve_ivp(ode, t, y0, max_step=1, args=(N,p1,p2,t_advent),events=advent_new_Strain)
    S,E_1, A_1, I_1, C_1, R_1, D_1, E_2, A_2, I_2, C_2, R_2, D_2 = sol.y
    return [(R_1[-1]+D_1[-1])/N*100 , (R_2[-1]+D_2[-1])/N*100, (D_1[-1])/N*100 , (D_2[-1])/N*100 ]

inputs = list(t_advent_arr)
num_cores = cpu_count()

if __name__ == '__main__':
    pool = Pool(num_cores)
    result= pool.map(solve_SEICARD, inputs)
    pool.close()
    pool.join()  # block at this line until all processes are done
    print("completed")
    result = np.array(result)

 
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')

    ax = fig.add_subplot(211, axisbelow=True)
    ax.plot(inputs[1:],result[1:,0],label='Strain 1');plt.legend();
    ax.plot(inputs[1:],result[1:,1],label='Strain 2');plt.legend();
    ax.plot(inputs[1:],result[1:,0]+result[1:,1],label='Total');plt.legend();
    #ax.set_xlabel('Advent time of Strain 2 (days)')
    ax.set_xlim(0,inputs[1])
    ax.set_ylabel('Infected (%)')
    ax.set_ylim(0,100)
    ax.set_title(r"$R_0^{(1)}$ = "+str(round(R0_1*10)/10)+ r", $R_0^{(2)}$ = "+str(round(R0_2*10)/10)+ r", $P_1^A$ = "+str(p1.A_frac)+ r", $P_2^A$ = "+str(p2.A_frac))
    
    ax = fig.add_subplot(212, axisbelow=True)
    ax.plot(inputs[1:],result[1:,2],label='Strain 1');plt.legend();
    ax.plot(inputs[1:],result[1:,3],label='Strain 2');plt.legend();
    ax.plot(inputs[1:],result[1:,2]+result[1:,3],label='Total');plt.legend();
    ax.set_xlim(0,inputs[1])
    ax.set_xlabel('T$_E$ (days)')
    ax.set_ylabel('Cumulative mortality(%)')
    ax.set_ylim(0,.5)

    plt.show()

