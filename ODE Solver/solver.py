# This code uses fourth order Runge-Kutta method to solve the coupled ODEs for the genes.

import os
import numpy as np
from Hill import act_hf, inh_hf
import matplotlib.pyplot as plt
import json


prod_indices = np.loadtxt('prod_indices').astype(np.int64)
deg_indices = np.loadtxt('deg_indices').astype(np.int64)
act_param_indices = np.loadtxt('act_param_indices').astype(np.int64)
inh_param_indices = np.loadtxt('inh_param_indices').astype(np.int64)

total_genes = len(prod_indices)

# Loading Parameter Set
params = np.loadtxt('TS_parameters.dat')


def f(r,t,param_set):
    num_genes = np.size(r)
    f_val = np.empty(r.shape)
    for g in range(num_genes):
        production = param_set[prod_indices[g]]
        degradation = param_set[deg_indices[g]]
        activation = act_hf(r,act_param_indices[g,:],param_set)
        inhibition = inh_hf(r,inh_param_indices[g,:],param_set)
        f_val[g] = production*activation*inhibition - degradation*r[g]
    
    return f_val


def RK4(r,a,b,N,param_set):
    h = (b-a)/N
    final_data = np.empty((N,np.size(r)))
    tpoints = np.arange(a,b,h)
    for ind,t in enumerate(tpoints):
        final_data[ind,:] = r
        k1 = h * f(r,t,param_set)
        k2 = h * f(r + 0.5*k1, t + 0.5*h, param_set)
        k3 = h * f(r + 0.5*k2, t + 0.5*h, param_set)
        k4 = h * f(r + k3, t + h, param_set)
        r += (k1 + 2*k2 + 2*k3 + k4)/6
    
    return final_data


def solve(parameter_set_num,deg_name,train_data_name,stability,plot=False):
    # Defaults
    total_time = 15     # in seconds
    dt = 0.01           # in seconds
    N = int((total_time)/dt)  # number of time steps

    # Creating the new folder for storing the training data and degradation values
    new_dir = f'Data for {stability} parameters'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    param_set = params[parameter_set_num-1,2:]

    # Creating the degradation file for GRN Deducer
    deg_path = os.path.join(os.path.dirname(__file__),new_dir,f"{deg_name}_{parameter_set_num}.txt")
    deg = np.empty(deg_indices.shape)
    for d in range(len(deg_indices)):
        deg[d] = param_set[deg_indices[d]]
    np.savetxt(deg_path,deg)

    # Random Initialization of first data point
    # r = np.random.normal(size=(total_genes))
    r = np.random.uniform(size=(total_genes),low=0.0,high=100.0)
    data_points_selected = False
    final_data_generated = False
    while not final_data_generated:
        time_series_data = RK4(r,0.0,total_time,N,param_set)
        
        if plot == True:
            # Plotting
            for g in range(total_genes):
                plt.plot(range(N),time_series_data[:,g],label=chr(65+g))
            plt.legend()
            plt.title(f'{parameter_set_num}-th Parameter')
            plt.show()
        
        if not data_points_selected:
            keep_current_points = input('Are current points good? [Y/n] =')
            print('')
            if keep_current_points != 'Y':
                print(f'Select initial data points for parameter set {parameter_set_num}')
                for g in range(total_genes):
                    r[g] = float(input(f'Select initial value for gene {g+1}:'))
                print('')
                print(f"Current Total Time = {total_time}")
                total_time = int(input("New Total Time (in seconds) ="))
                N = int((total_time)/dt)  # updating the number of time steps
            else:
                final_data_generated = True

    train_path = os.path.join(os.path.dirname(__file__),new_dir,f"{train_data_name}_{parameter_set_num}.txt")
    np.savetxt(train_path,time_series_data)

    
    return None


# Monostable
stability = 'monostable'
solve_for, already_solved = [], []
with open("to_gen_mono.json",'r') as file:
    solve_for = json.load(file)
file.close()

pth = os.path.join(os.path.dirname(__file__),"Data for monostable parameters")
dir_list = os.listdir(pth)
for fl in dir_list:
    n = ""
    if fl[0] == 't':
        for i in fl:
            if i.isnumeric() and int(i) in list(range(10)):
                n += i
        already_solved.append(int(n))


# # Bistable
# stability = 'bistable'
# solve_for, already_solved = [], []
# with open("to_gen_bi.json",'r') as file:
#     solve_for = json.load(file)
# file.close()

# pth = os.path.join(os.path.dirname(__file__),"Data for bistable parameters")
# dir_list = os.listdir(pth)
# for fl in dir_list:
#     n = ""
#     if fl[0] == 't':
#         for i in fl:
#             if i.isnumeric() and int(i) in list(range(10)):
#                 n += i
#         already_solved.append(int(n))


for prm in solve_for:
    if  (prm not in already_solved):
        solve(prm,'degradation','training_data',stability,plot=True)
        already_solved.append(prm)
