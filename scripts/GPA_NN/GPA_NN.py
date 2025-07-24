#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow as tf
import numpy as np
import re
import sys
import copy

main_dir = copy.deepcopy(os.getcwd())
if main_dir == "/content":
    main_dir = main_dir + "/GPA-source_code"
if "GPA_NN" not in main_dir:
    os.chdir(main_dir + "/scripts/GPA_NN")
sys.path.append('../../')

# input parameters --------------------------------------
from scripts.util.input_args import input_params
p, _ = input_params()

import yaml
from yaml.loader import SafeLoader
yaml_file = "../../configs/{dataset}-{generative_model}.yaml".format(dataset=p.dataset, generative_model=p.generative_model)


with open(yaml_file, 'r') as f:
    param = yaml.load(f, Loader=SafeLoader)
    #print(param)
    
updated_param = vars(p)
for param_key, param_val in updated_param.items():
    if type(param_val) == type(None):
        continue
    param[param_key] = param_val
    
if param['alpha']:
    par = [param['alpha']]
    param['exptype'] = '%s=%05.2f-%s' % (param['f'], param['alpha'], param['Gamma'])
else: 
    par = []
    param['exptype'] = '%s-%s' % (param['f'], param['Gamma'])
if param['L'] == None:
    param['expname'] = '%s_%s' % (param['exptype'], 'inf')
else:
    param['expname'] = '%s_%.4f' % (param['exptype'], param['L'])
if param['reverse'] == True:
    param['expname'] += '_reverse'
    
    
# Mixture GPA settings ------------------------------------------------
if 'label' in param.keys():
    if len(param['label']) > 2:
        from scripts.util.generate_data import reduce_data
        
        if 'iter_transient_target' not in param.keys():
            transient_ts = np.array(param['label'][1:-1])
            param['iter_transient_target'] = list( np.floor(transient_ts/param['label'][-1]*param['epochs']))
            
        if 'weight_decay_mixture_target' not in param.keys():
            param['weight_decay_mixture_target'] = 1.0
            
            

# Data generation ----------------------------------------
from scripts.util.generate_data import generate_data
param, X_, Y_, X_label, Y_label = generate_data(param)


Q = tf.constant(X_*p.rescale_factor, dtype=tf.float32) # constant
P = tf.Variable(Y_*p.rescale_factor, dtype=tf.float32) # variable
    
if param['N_conditions'] >1:
    Q_label = tf.constant(X_label, dtype=tf.float32)
    P_label = tf.constant(Y_label, dtype=tf.float32)
    
else:
    Q_label, P_label = None, None
    

data_par = {'P_label': P_label, 'Q_label': Q_label, 'mb_size_P': param['mb_size_P'], 'mb_size_Q': param['mb_size_Q'], 'N_samples_P': param['N_samples_P'], 'N_samples_Q': param['N_samples_Q'], }

try:
    data_par.update({'Q_time_label': param['X_time_label'], 'weight_decay': param['weight_decay_mixture_target']})
except:
    pass

print("Data prepared.")
print(param)

# Discriminator learning  -----------------------------------------
# Discriminator construction using Neural Network
from lib.construct_NN import check_nn_topology, initialize_NN, model

N_fnn_layers, N_cnn_layers, param['activation_ftn'] = check_nn_topology(param['NN_model'], param['N_fnn_layers'], param['N_cnn_layers'], param['N_dim'], param['activation_ftn'])

NN_par = {'NN_model':param['NN_model'], 'activation_ftn':param['activation_ftn'], 'N_dim': param['N_dim'], \
'N_cnn_layers':N_cnn_layers, 'N_fnn_layers':N_fnn_layers, 'N_conditions': param['N_conditions'], \
'constraint': param['constraint'], 'L': param['L'], 'eps': param['eps'], }

W, b = initialize_NN(NN_par)
phi = model(NN_par)  # discriminator

# scalar optimal value optimization for f-divergence
nu = tf.Variable(0.0, dtype=tf.float32)

parameters = {'W':W, 'b':b, 'nu':nu} # Learnable parameters for the discriminator phi

# calculate f_Lip divergence for calculating error between current particles and intermediate target
if param['termination_error_metric'] == 'f_Lip':
    W_error, b_error = initialize_NN(NN_par)
    phi_error = model(NN_par)  # discriminator

    # scalar optimal value optimization for f-divergence
    nu_error = tf.Variable(0.0, dtype=tf.float32)

    parameters_error = {'W':W_error, 'b':b_error, 'nu':nu_error} # Learnable parameters for the discriminator phi


# Train setting
from lib.train_NN import train_disc
lr_phi = tf.Variable(param['lr_phi'], trainable=False) # lr for training a discriminator function

# (Discriminator) Loss ----------------------------------------------
loss_par = {'f': param['f'], 'formulation': param['formulation'], 'par': par, 'reverse': param['reverse'], 'lamda': param['lamda']}

# Transporting particles --------------------------------------------
# ODE solver setting
from lib.transport_particles import calc_vectorfield, solve_ode
dPs = []
if param['ode_solver'] in ['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5']:
    aux_params = []
else:
    aux_params = {'parameters': parameters, 'phi': phi, 'Q': Q, 'lr_phi': lr_phi,'epochs_phi': param['epochs_phi'], 'loss_par': loss_par, 'NN_par': NN_par, 'data_par': data_par, 'optimizer': param['optimizer']}

# Applying mobility to particles
if param['mobility'] == 'bounded':
    from lib.construct_NN import bounded_relu  # mobility that bounding particles (For image data)
        
# Train setting
lr_P_init = param['lr_P'] # Assume that deltat = deltat(t)
if param['ode_solver'] == "DOPRI5": # deltat = deltat(x,t)
    lr_P_init = [param['lr_P']]*param['N_samples_P']
    # Low dimensional example=> rank 2, Image example=> rank 4
    for i in range(1, tf.rank(P)):
        lr_P_init = np.expand_dims(lr_P_init, axis=i)
lr_P = tf.Variable(lr_P_init, trainable=False)
lr_Ps = []




# Save & plot settings -----------------------------------------------
# Metrics to calculate
from scripts.util.evaluate_metric import calc_ke, calc_grad_phi
if np.prod(param['N_dim']) <= 12:
    from scripts.util.evaluate_metric import calc_sinkhorn
if np.prod(param['N_dim']) > 100:
    from scripts.util.evaluate_metric import calc_fid
trajectories = []
vectorfields = []
divergences = []
KE_Ps = []
FIDs = []

# saving/plotting parameters
if param['save_iter'] >= param['epochs']:
    param['save_iter'] = 1

if param['plot_result'] == True:
    from scripts.util.plot_result import plot_result

if not os.path.exists(main_dir + '/assets/' + param['dataset']):
    os.makedirs(main_dir + '/assets/' + param['dataset'])

param['expname'] = param['expname']+'_%04d_%04d_%02d_%s' % (param['N_samples_Q'], param['N_samples_P'], param['random_seed'], param['exp_no'])
filename = main_dir + '/assets/' + param['dataset']+'/%s.pickle' % param['expname']

if param['plot_intermediate_result'] == True:
    if 'gaussian' in param['dataset'] and 'Extension' not in param['dataset']:
         r_param = param['sigma_Q']
    elif 'student_t' in param['dataset']:
        r_param = param['nu']
    elif param['dataset'] == 'Extension_of_gaussian':
        r_param = param['a']
    else:
        r_param = None
    
    
# Train ---------------------------------------------------------------
import matplotlib.pyplot as plt
import time 
t0 = time.time()
                                    
if 'label' in param.keys():
    if len(param['label']) > 2:
        t_idx = 0
        t_idxs = sorted(set(param['X_time_label']))
        intermediate_target = tf.constant(X_[param['X_time_label']==t_idxs[t_idx]], dtype=tf.float32)
        if param['termination_error_metric'] == 'f_Lip':
            data_par_error = {'P_label': P_label, 'Q_label': Q_label, 'mb_size_P': param['mb_size_P'], 'mb_size_Q': intermediate_target.shape[0], 'N_samples_P': param['N_samples_P'], 'N_samples_Q': intermediate_target.shape[0], }

for it in range(1, param['epochs']+1): # Loop for updating particles P
    parameters, current_loss, dW_norm = train_disc(parameters, phi, P, Q, lr_phi, param['epochs_phi'], loss_par, NN_par, data_par, param['optimizer'], print_vals=True)
    dPs.append( calc_vectorfield(phi, P, parameters, NN_par, loss_par, data_par) )
    
    if param['ode_solver'] == "DOPRI5": # deltat adust
        P, dPs, dP, lr_P = solve_ode(P, lr_P, dPs, param['ode_solver'], aux_params) # update P
    else:
        P, dPs, dP = solve_ode(P, lr_P, dPs, param['ode_solver'], aux_params) # update P

    if param['mobility'] == 'bounded':
        P.assign(bounded_relu(P))
     
    lr_Ps.append(lr_P.numpy())
        
    # save results
    divergences.append(current_loss)
    KE_P = calc_ke(dP, param['N_samples_P'])
    KE_Ps.append(KE_P)
    
    
    if len(param['label']) > 2:  # mixture GPA termination condition based on error
        if param['termination_error_metric'] == 'KE_P':
            error = KE_P
        elif param['termination_error_metric'] == 'f_Lip':
            parameters_error, error, _ = train_disc(parameters_error, phi_error, P, intermediate_target, lr_phi, param['epochs_phi'], loss_par, NN_par, data_par_error, param['optimizer'], print_vals=True)
            
        if (error < param['termination_error'] and t_idx < len(t_idxs)-1) or it in param['iter_transient_target']:
            print(f"Replace the target; error={error}, error threshold={param['termination_error']}")
            Q, data_par = reduce_data(Q, data_par, rescale_factor=p.rescale_factor)
            t_idx += 1
            intermediate_target =  tf.constant(X_[param['X_time_label']==t_idxs[t_idx]], dtype=tf.float32)
            if param['termination_error_metric'] == 'f_Lip':
                data_par_error = {'P_label': P_label, 'Q_label': Q_label, 'mb_size_P': param['mb_size_P'], 'mb_size_Q': intermediate_target.shape[0], 'N_samples_P': param['N_samples_P'], 'N_samples_Q': intermediate_target.shape[0], }
    
    elif param['termination_error_metric'] == 'KE' and KE_P < param['termination_error'] and it >= 10:    # Naive GPA termination condition based on error threshold
        print(f"current loss={KE_P}, error threshold for termination={param['termination_error']}")
        break
        
    elif param['termination_error_metric'] == 'f_Lip' and current_loss < param['termination_error']:    # Naive GPA termination condition based on error threshold
            print(f"current loss={current_loss}, error threshold for termination={param['termination_error']}")
            break

    
    
    
    if param['epochs']<=100 or it%param['save_iter'] == 0:
        trajectories.append(P.numpy() / p.rescale_factor)
        if np.prod(param['N_dim']) < 500:
            vectorfields.append(-dP.numpy())
        elif np.prod(param['N_dim']) >= 784:  # image data
            FIDs.append( calc_fid(pred=P.numpy(), real=Q.numpy()) )
    
    
    if it % (param['epochs']/10) == 0:
        display_msg = 'iter %6d: loss = %.10f, norm of dW = %.2f, kinetic energy of P = %.10f, average learning rate for P = %.6f' % (it, current_loss, dW_norm, KE_P, tf.math.reduce_mean(lr_P).numpy())
        if len(FIDs) > 0 :
            display_msg = display_msg + ', FID = %.3f' % FIDs[-1]   
        print(display_msg)
        
        if param['plot_intermediate_result'] == True:
            data = {'trajectories': trajectories, 'divergences': divergences, 'KE_Ps': KE_Ps, 'FIDs':FIDs, 'X_':X_, 'Y_':Y_, 'X_label':X_label, 'Y_label':Y_label, 'dt': lr_Ps, 'dataset': param['dataset'], 'r_param': r_param, 'vectorfields': vectorfields, 'save_iter':param['save_iter']}
            if param['N_dim'] ==2:
                data.update({'phi': phi, 'W':W, 'b':b, 'NN_par':NN_par})
            plot_result(filename, intermediate=True, epochs = it, iter_nos = None, data = data, show=False, proj_axes = p.plot_axes)
        
        
        if np.prod(param['N_dim']) == 1:
            zz = phi(xx,None, W,b,NN_par).numpy()
            zz = np.reshape(zz, -1)
            phis.append(zz)
            
        

total_time = time.time() - t0
print(f'total time {total_time:.3f}s')
        


# Save result ------------------------------------------------------
import pickle
if param['N_dim'] == 1:
    X_ = np.concatenate((X_, np.zeros(shape=X_.shape)), axis=1)
    Y_ = np.concatenate((Y_, np.zeros(shape=Y_.shape)), axis=1)
    
    trajectories = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in trajectories]
    vectorfields = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in vectorfields]
  
if param['L'] == None:
    param['L'] = 'inf'
param.update({'X_': X_, 'Y_': Y_, 'lr_Ps':lr_Ps,})
result = {'trajectories': trajectories, 'vectorfields': vectorfields, 'divergences': divergences, 'KE_Ps': KE_Ps, 'FIDs': FIDs, }

try:
    np.savetxt(main_dir + "/data/"+param['dataset']+'/output_dataset_dim_%d.csv' % param['N_dim'], trajectories[-1], delimiter=",")
except:
    print("Warning: Output csv file is not saved.")
        
# Save trained data
with open(filename,"wb") as fw:
    pickle.dump([param, result] , fw)
print("Results saved at:", filename)

# Plot final result ------------------------------------
if param['plot_result'] == True:
    plot_result(filename, intermediate=False, show=False, proj_axes = param['plot_axes'])

os.chdir(main_dir)
