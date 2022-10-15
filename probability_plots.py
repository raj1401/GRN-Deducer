import numpy as np
import matplotlib.pyplot as plt
import accessory_functions as acc_f
import json
import os


stability = "bistable"
pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters","training_data_2.txt")
total_genes = (np.loadtxt(pth)).shape[1]

if stability == "monostable":
    tks = []
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","to_gen_mono.json")
    with open(pth,'r') as f:
        tks = json.load(f)
    f.close()
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","monostable_data.txt")
    monostable_data = np.loadtxt(pth)
    plt.plot(monostable_data[:,1])
    plt.xlabel("Parameter Sets")
    plt.ylabel("A/B")
    plt.title("Ratio of steady state values v/s parameter sets")
    plt.show()

    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters","Prob_Matrix.txt")
    prob_mat = np.loadtxt(pth)

    # Plotting Error
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters","Error_Array.txt")
    err_arr = np.loadtxt(pth)
    plt.plot(err_arr)
    plt.xticks(ticks=range(len(tks)),labels=tks)
    plt.xlabel("Parameter Number")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Square Error on Training Various Data Sets")
    plt.show()

    perm_mat = acc_f.permutation_matrices(total_genes)
    # print(perm_mat)
    for i in range(prob_mat.shape[1]):
        if (np.max(np.abs(prob_mat[:,i])) > 0):
            plt.plot(prob_mat[:,i],label=perm_mat[i])
    plt.xlabel("Parameter Number")
    plt.ylabel("Probability")
    plt.xticks(ticks=range(len(tks)),labels=tks)
    plt.title("Probability of coming from various topologies")
    plt.legend()
    plt.show()
else:
    tks = []
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","to_gen_bi.json")
    with open(pth,'r') as f:
        tks = json.load(f)
    f.close()

    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","bistable_data.txt")
    bistable_data = np.loadtxt(pth)
    plt.plot(bistable_data[:,1])
    plt.xlabel("Parameter Sets")
    plt.ylabel("A/B")
    plt.title("Ratio of steady states going to A high versus B high v/s parameter sets")
    plt.show()

    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters","Prob_matrix.txt")
    prob_mat = np.loadtxt(pth)

    # Plotting Error
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters","Error_Array.txt")
    err_arr = np.loadtxt(pth)
    plt.plot(err_arr)
    plt.xticks(ticks=range(len(tks)),labels=tks)
    plt.xlabel("Parameter Number")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Square Error on Training Various Data Sets")
    plt.show()

    perm_mat = acc_f.permutation_matrices(total_genes)
    #print(perm_mat)
    for i in range(prob_mat.shape[1]):
        if (np.max(np.abs(prob_mat[:,i])) > 0):
            plt.plot(prob_mat[:,i],label=perm_mat[i])
    plt.xlabel("Parameter Number")
    plt.ylabel("Probability")
    plt.xticks(ticks=range(len(tks)),labels=tks)
    plt.title("Probability of coming from various topologies")
    plt.legend()
    plt.show()
