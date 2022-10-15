import numpy as np
import os


file_names_mono = os.listdir('Data for monostable parameters')
file_names_bi = os.listdir('Data for bistable parameters')

new_dir_1 = 'Data for monostable parameters Multiple Normalized'
if not os.path.exists(new_dir_1):
    os.makedirs(new_dir_1)
new_dir_2 = 'Data for bistable parameters Multiple Normalized'
if not os.path.exists(new_dir_2):
    os.makedirs(new_dir_2)


for f in file_names_mono:
    if f[0] == 't':
        p = os.path.join(os.path.dirname(__file__),"Data for monostable parameters",f"{f}")
        data = np.loadtxt(p)
        new_dat = np.empty(shape=data.shape)
        num_genes = data.shape[1]
        param = ""
        for i in f:
                if i.isnumeric():
                    param += i
        param_int = int(param)
        for g in range(num_genes):
            new_dat[:,g] = 1/np.max(data[:,g]) * data[:,g] - 1
        p = os.path.join(os.path.dirname(__file__),"Data for monostable parameters Multiple Normalized",f"training_data_mult_norm_{param_int}.txt")
        np.savetxt(p,new_dat)


for f in file_names_bi:
    if f[0] == 't':
        p = os.path.join(os.path.dirname(__file__),"Data for bistable parameters",f"{f}")
        data = np.loadtxt(p)
        new_dat = np.empty(shape=data.shape)
        num_genes = data.shape[1]
        param = ""
        for i in f:
                if i.isnumeric():
                    param += i
        param_int = int(param)
        for g in range(num_genes):
            new_dat[:,g] = 1/np.max(data[:,g]) * data[:,g] - 1
        p = os.path.join(os.path.dirname(__file__),"Data for bistable parameters Multiple Normalized",f"training_data_mult_norm_{param_int}.txt")
        np.savetxt(p,new_dat)