import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import accessory_functions as acc_f
import json
import os


stability = "monostable"
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

    fig = plt.figure(figsize=[25,25])
    gs = gspec.GridSpec(nrows=6,ncols=1)
    gs.update(left=0.062,right=0.97,bottom=0.083,top=0.962,wspace=0.162,hspace=0.998)

    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    ax5 = plt.subplot(gs[4,0])
    ax6 = plt.subplot(gs[5,0])

    # Plotting A/B ratio
    ax1.plot(monostable_data[:,1])
    ax1.set_ylabel("A/B")
    ax1.set_title("Ratio of steady state values v/s parameter sets")

    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters","Prob_Matrix.txt")
    prob_mat = np.loadtxt(pth)

    # Plotting Error
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for monostable parameters","Error_Array.txt")
    err_arr = np.loadtxt(pth)
    ax2.plot(err_arr)
    #ax2.set_xticks(ticks=range(len(tks)),labels=tks)
    ax2.set_ylabel("MS Err")
    ax2.set_title("Mean Square Error on Training Various Data Sets")

    perm_mat = acc_f.permutation_matrices(total_genes)
    
    # For making the cumulative
    cumul_mat = np.zeros(shape=prob_mat.shape)
    for prm in range(prob_mat.shape[0]):
        cumul_mat[prm,np.argmax(prob_mat[prm,:])] = 1
    
    # Plotting probabilities
        
    ax3.plot(prob_mat[:,0],label=perm_mat[0], color="blue")
    #ax3.set_xticks(ticks=range(len(tks)),labels=tks)
    ax3.set_ylabel(f"{perm_mat[0]}")
    ax3.set_title("Probabilities v/s Parameter Sets")

    ax4.plot(prob_mat[:,2],label=perm_mat[2], color="green")
    #ax4.set_xticks(ticks=range(len(tks)),labels=tks)
    ax4.set_ylabel(f"{perm_mat[2]}")

    ax5.plot(prob_mat[:,6],label=perm_mat[6], color="orange")
    #ax5.set_xticks(ticks=range(len(tks)),labels=tks)
    ax5.set_ylabel(f"{perm_mat[6]}")

    ax6.plot(prob_mat[:,8],label=perm_mat[8], color="black")
    #ax6.set_xticks(ticks=range(len(tks)),labels=tks)
    ax6.set_xlabel("Parameter Number")
    ax6.set_ylabel(f"{perm_mat[8]}")
    plt.show()

    fig7,ax7 = plt.subplots()
    # Plotting cumulatives
    bin_size = 50
    width=0.5
    cumul_plt = np.zeros(shape=(cumul_mat.shape[0]//bin_size + 1,cumul_mat.shape[1]))
    for i,x in enumerate(range(0,cumul_mat.shape[0],bin_size)):
        if x+bin_size < cumul_mat.shape[0]:
            for j in range(cumul_mat.shape[1]):
                cumul_plt[i,j] = np.sum(cumul_mat[x:x+bin_size,j])
        else:
            for j in range(cumul_mat.shape[1]):
                cumul_plt[i,j] = np.sum(cumul_mat[x:,j])
    
    x_val = np.arange(cumul_plt.shape[0])
    rects1 = ax7.bar(x_val - width/2, cumul_plt[:,0], width, label=perm_mat[0], color='blue')
    rects2 = ax7.bar(x_val - width/4, cumul_plt[:,2], width, label=perm_mat[2], color="green")
    rects3 = ax7.bar(x_val + width/4, cumul_plt[:,6], width, label=perm_mat[6], color="orange")
    rects4 = ax7.bar(x_val + width/2, cumul_plt[:,8], width, label=perm_mat[8], color="black")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax7.set_xlabel("Parameter Set Bins")
    ax7.set_ylabel('Occurence of highest probability')
    ax7.set_title('Shift in Probabilities with A/B ratio')
    ax7.legend()

    # ax7.bar_label(rects1,padding=3)    
    # ax7.bar_label(rects2,padding=3) 
    # ax7.bar_label(rects3,padding=3) 
    # ax7.bar_label(rects4,padding=3)       

    plt.show()

else:
    tks = []
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","to_gen_bi.json")
    with open(pth,'r') as f:
        tks = json.load(f)
    f.close()

    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","bistable_data.txt")
    bistable_data = np.loadtxt(pth)

    fig = plt.figure(figsize=[25,25])
    gs = gspec.GridSpec(nrows=6,ncols=1)
    gs.update(left=0.062,right=0.97,bottom=0.083,top=0.962,wspace=0.162,hspace=0.998)

    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    ax5 = plt.subplot(gs[4,0])
    ax6 = plt.subplot(gs[5,0])
    
    ax1.plot(bistable_data[:,1])
    ax1.set_ylabel("A/B")
    ax1.set_title("Ratio of steady states going to A high versus B high v/s parameter sets")

    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters","Prob_matrix.txt")
    prob_mat = np.loadtxt(pth)

    # Plotting Error
    pth = os.path.join(os.path.dirname(__file__),"ODE Solver","Data for bistable parameters","Error_Array.txt")
    err_arr = np.loadtxt(pth)
    
    ax2.plot(err_arr)
    #ax2.set_xticks(ticks=range(len(tks)),labels=tks)
    ax2.set_ylabel("MS Err")
    ax2.set_title("Mean Square Error on Training Various Data Sets")

    perm_mat = acc_f.permutation_matrices(total_genes)
    
    # For making the cumulative
    cumul_mat = np.zeros(shape=prob_mat.shape)
    for prm in range(prob_mat.shape[0]):
        cumul_mat[prm,np.argmax(prob_mat[prm,:])] = 1
    
    # Plotting probabilities
        
    ax3.plot(prob_mat[:,0],label=perm_mat[0], color="blue")
    #ax3.set_xticks(ticks=range(len(tks)),labels=tks)
    ax3.set_ylabel(f"{perm_mat[0]}")
    ax3.set_title("Probabilities v/s Parameter Sets")

    ax4.plot(prob_mat[:,2],label=perm_mat[2], color="green")
    #ax4.set_xticks(ticks=range(len(tks)),labels=tks)
    ax4.set_ylabel(f"{perm_mat[2]}")

    ax5.plot(prob_mat[:,6],label=perm_mat[6], color="orange")
    #ax5.set_xticks(ticks=range(len(tks)),labels=tks)
    ax5.set_ylabel(f"{perm_mat[6]}")

    ax6.plot(prob_mat[:,8],label=perm_mat[8], color="black")
    #ax6.set_xticks(ticks=range(len(tks)),labels=tks)
    ax6.set_xlabel("Parameter Number")
    ax6.set_ylabel(f"{perm_mat[8]}")
    plt.show()

    fig7,ax7 = plt.subplots()
    # Plotting cumulatives
    bin_size = 20
    width=0.5
    cumul_plt = np.zeros(shape=(cumul_mat.shape[0]//bin_size + 1,cumul_mat.shape[1]))
    for i,x in enumerate(range(0,cumul_mat.shape[0],bin_size)):
        if x+bin_size < cumul_mat.shape[0]:
            for j in range(cumul_mat.shape[1]):
                cumul_plt[i,j] = np.sum(cumul_mat[x:x+bin_size,j])
        else:
            for j in range(cumul_mat.shape[1]):
                cumul_plt[i,j] = np.sum(cumul_mat[x:,j])
    
    x_val = np.arange(cumul_plt.shape[0])
    rects1 = ax7.bar(x_val - width/2, cumul_plt[:,0], width, label=perm_mat[0], color='blue')
    rects2 = ax7.bar(x_val - width/4, cumul_plt[:,2], width, label=perm_mat[2], color="green")
    rects3 = ax7.bar(x_val + width/4, cumul_plt[:,6], width, label=perm_mat[6], color="orange")
    rects4 = ax7.bar(x_val + width/2, cumul_plt[:,8], width, label=perm_mat[8], color="black")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax7.set_xlabel("Parameter Set Bins")
    ax7.set_ylabel('Occurence of highest probability')
    ax7.set_title('Shift in Probabilities with A/B ratio')
    ax7.legend()

    # ax7.bar_label(rects1,padding=3)    
    # ax7.bar_label(rects2,padding=3) 
    # ax7.bar_label(rects3,padding=3) 
    # ax7.bar_label(rects4,padding=3)       

    plt.show()