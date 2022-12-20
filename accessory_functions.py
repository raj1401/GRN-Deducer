import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Defaults
dt = 0.01
# Euler Integration Step
def g_n1(f,g_n,gamma):
    return g_n + dt * (f - gamma*g_n)

def generate_data(initial_val,genes_to_train,gamma,time_steps,f_gen,x_gen,NN_models):
    x_gen[0,:] = initial_val
    for t in range(time_steps):
        #print(f"Generating Data. Progress = {round(t/time_steps*100)} %", end="\r")
        input_val = np.array([x_gen[t,:]])
        for gene in range(genes_to_train):
            NN_model = NN_models[gene]
            f_gen[t,gene] = NN_model(tf.convert_to_tensor(input_val, dtype=tf.float64))
            if t != time_steps - 1:
                x_gen[t+1,gene] = g_n1(f_gen[t,gene],x_gen[t,gene],gamma[gene])
    
    # Clearing the model to avoid clutter
    tf.keras.backend.clear_session()
    return x_gen, f_gen


# Function that converts vector y_i to vector f_i using diff. eqn. in Shen et. al. pg.2
def f(y,gma,dt):
    length = np.size(y,0)
    f_vec = np.empty(length)
    for i in range(length-1):
        f_vec[i] = (y[i+1,0] - y[i,0])/dt + gma*y[i,0]
    f_vec[-1] = f_vec[-2]
    return np.transpose(f_vec)


def mean_squared_error(x_train,x_gen):
    m = x_train.shape[0]
    return 1/(2*m) * np.sum(np.square(x_train-x_gen))



def permutation_matrices(num_genes):
    num_edges = num_genes * (num_genes - 1)
    val_vector = -1 * np.ones(num_edges)
    matrices_list = []
    end_loop = False
    while not end_loop:
        matrix = np.zeros((num_genes,num_genes))
        k = 0
        # Constructing the matrix from val_vector
        for i in range(num_genes):
            for j in range(num_genes):
                if i != j:
                    matrix[i,j] = val_vector[k]
                    k += 1
        matrices_list.append(matrix)
        idx = -1
        val_vector[idx] += 1
        while val_vector[idx] > 1:
            val_vector[idx] = -1
            idx -= 1
            if idx < -num_edges:
                end_loop = True
                break
            val_vector[idx] += 1   
    
    return matrices_list


# Gives the derivative at any x_i value
def der_ij(x_i,hidden_layer_param,output_layer_param,hidden_bias,act_fn):
    z = x_i * hidden_layer_param + hidden_bias
    if (act_fn == 'tanh'):
        derivative = 1 - np.square(np.tanh(z))
        mul1 = np.multiply(output_layer_param,hidden_layer_param)
        result = (np.multiply(mul1,derivative))
        return np.sum(result)
    elif (act_fn == 'sigmoid'):
        denom = np.square(np.exp(z/2)+np.exp(-z/2))
        mul1 = np.multiply(output_layer_param,hidden_layer_param)
        mul2 = np.divide(z,denom)
        result = (np.multiply(mul1,mul2))
        return np.sum(result)
    else:
        print("Invalid Activation Function")
        return 0

def deduce_interactions(x_train,NN_models,genes_to_train,act_fn,plot=False):
    interaction_matrix = np.empty((genes_to_train,genes_to_train))
    plt_num = 0
    for j in range(genes_to_train):
        NN_model = NN_models[j]
        output_theta = NN_model.layers[1].get_weights()[0]
        output_theta = np.asarray(output_theta[:,0],dtype=np.float64)
        hidden_bias = np.asarray(NN_model.layers[0].get_weights()[1],dtype=np.float64)
        max_val = np.max(x_train[:,j])
        min_val = 0   # Any gene can have a minimum expression of zero
        for i in range(genes_to_train):
            plt_num += 1
            hidden_theta = NN_model.layers[0].get_weights()[0]
            hidden_theta = np.asarray(hidden_theta[i,:],dtype=np.float64)
            #interaction_matrix[i,j] = der_ij_at_zero(hidden_theta,output_theta,hidden_bias)
            #interaction_matrix[i,j] = der_ij((max_val+min_val)/2,hidden_theta,output_theta,hidden_bias)
            derf_val = np.empty(1000)
            x_val = np.empty(1000)
            if plot == True:
                plt.subplot(genes_to_train,genes_to_train,plt_num)
            for ind,x in enumerate(np.linspace(min_val,max_val,1000,dtype=np.float64)):
                derf_val[ind] = der_ij(x,hidden_theta,output_theta,hidden_bias,act_fn)
                x_val[ind] = x
            interaction_matrix[i,j] = np.mean(derf_val)
            if plot == True:
                plt.plot(x_val,derf_val)
    
    if plot == True:
        plt.show()
    
    return interaction_matrix