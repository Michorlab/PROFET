#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import argparse

import numpy as np
import tensorflow as tf

# parameters for running train_time_dep_vectorfields.py
# basic usage
# python3 train_time_dep_vectorfields.py --dataset Transport_Genes --ts 0 2000 4000 --files [names of GPA results]
parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', type=str, help='dataset name', default='Transport_genes',
)
parser.add_argument(
    '--ts', type=float, nargs='+', help='list of physical time points',
)
parser.add_argument(
    '--files', type=str, nargs='+', help='list of GPA output files. Make sure that files are TIME ORDERED',
)
parser.add_argument(
    '--hidden_units', type=int, nargs='+', help='number of hidden units to train the vectorfield NN', default=[64,64,64,64]
)
parser.add_argument(
    '--mini_batch_size_t', type=int, help='number of different times to be sampled', default=10,
)
parser.add_argument(
    '--mini_batch_size_x', type=int, help='number of different times to be sampled', default=500,
)
parser.add_argument(
    '--threshold_type', type=str, choices = ['f_Lip', 'KE'], help='error threshold to decide termination time for GPA',
)
parser.add_argument(
    '--threshold_val', type=float, help='error threshold value to decide termination time for GPA',
)
parser.add_argument(
    '--exp_memo', type=str, help='example memo', default='KL_3timepoints',
)
parser.add_argument(
    '--lr_NN', type=float, help='learning rate for neural network', default = 0.00001
)
parser.add_argument(
    '--iterations', type=int, help='number of iterations for neural network update', default=20000,
)



p, _ = parser.parse_known_args()
p = vars(p)


current_filename = __file__
assets_dir = current_filename.split('scripts')[0]
assets_dir = assets_dir + 'assets/'

vectorfields = []
trajectories = []
gpa_Ts = [0.0]
# read GPA results
for file in p['files']:
    filename = assets_dir + p['dataset'] + '/' + file
    
    with open(filename, "rb") as fr:
        param, result = pickle.load(fr)
        
    if p['threshold_type'] == 'f_Lip':
        error_val = result['divergences']
    elif p['threshold_type'] == 'KE':
        error_val = result['KE_Ps']
        
    idx = len(result['vectorfields'])
    for i, val in enumerate(error_val):
        if val <= p['threshold_val'] and i > 9:
            idx = int(i/param['save_iter']) + 1
            break
            
    vectorfields.append(result['vectorfields'][:idx])
    trajectories.append(result['trajectories'][:idx])
    gpa_Ts.append(gpa_Ts[-1] + param['save_iter'] * len(result['vectorfields'][:idx]) * param['lr_P'])

# --------
# time conversion
if len(p['files']) > 1:  # multiple GPA results to be combined
    max_ratio = 0.0
    ratios = []
    for i in range(len(p['ts'])-1):
        r = (gpa_Ts[i+1] - gpa_Ts[i])/(p['ts'][i+1] - p['ts'][i])
        ratios.append(r)
        max_ratio = np.maximum(max_ratio, r)
        
    ts = [0.0]
    for i in range(len(p['ts'])-1):
        ts.append(max_ratio*p['ts'][i+1])
        ratios[i] /= max_ratio
        
    print(f"GPA time = {gpa_Ts}, Physical time = {p['ts']}")
    print(f"Ratios between GPA times = {ratios}, Converted time = {ts}")
else:
    ratios = [1.0]
    ts = gpa_Ts
    
    print(f"GPA time = {gpa_Ts}, Physical time = {p['ts']}, Converted time = {ts}")
    
# --------
# Neural network setup
space_dim = vectorfields[0][0].shape[1]

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev)
    
def initialize_W(layers = p['hidden_units']):
    NN_W, NN_b=[], []
    
    for i, l_i in enumerate(layers):
        if i == 0:
            W = tf.Variable(xavier_init(size=[space_dim + 1, l_i]),  dtype=tf.float32)
        else:
            W = tf.Variable(xavier_init(size=[layers[i-1], l_i]),  dtype=tf.float32)
        b = tf.Variable(tf.zeros([1,l_i]), dtype=tf.float32)

        NN_b.append(b)
        NN_W.append(W)
    
    W = tf.Variable(xavier_init(size=[layers[-1], space_dim]),  dtype=tf.float32)
    b = tf.Variable(tf.zeros([1,space_dim]), dtype=tf.float32)

    NN_b.append(b)
    NN_W.append(W)
    
    return NN_W, NN_b
    
def v(x, t, W, b):   # neural newtork for time-dependent vectorfield
    num_layers = len(W)
    # regularity of activation function corresponds to
    # the smoothness of time-dependent vectorfield
    activation_ftn = tf.nn.tanh
    
    h = tf.concat([x, t*tf.ones([x.shape[0], 1], dtype=tf.float32)], axis=1)
    for l in range(0,num_layers-1):
        h = activation_ftn(tf.add(tf.matmul(h, W[l]), b[l]))
    out=tf.add(tf.matmul(h, W[-1]), b[-1])

    return out

W, b = initialize_W()
lr_NN = p['lr_NN'] #0.00001
iterations = p['iterations']

# loss function
def MSE(true, predict):
    return tf.math.reduce_sum(tf.math.square(true - predict))/predict.shape[0]
    

# optimizer
def adam_update(grad, iter, m, v, beta1=0.9, beta2=0.999, eps=1e-8):
    grad = grad.numpy()
    if iter == 0:
        m, v = np.zeros_like(grad), np.zeros_like(grad)
        
    # reweight m, v
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad**2
    
    m_hat = m/(1-beta1**(iter+1))
    v_hat = v/(1-beta2**(iter+1))
    
    grad_hat = m_hat/(np.sqrt(v_hat)+eps)
    
    return grad_hat, m, v
    
def adam(lr_NN, iter, W, dW, m_W, v_W, b, db, m_b, v_b, descent=True, calc_dW_norm=False):
    dW_norm = 0
    if descent == False:
        lr = - lr_NN
    else:
        lr = lr_NN
    
    for l in range(len(W)):
        if calc_dW_norm == True:
            dW_norm = max(dW_norm, tf.norm(dW[l]))
        dW_hat, m_W[l], v_W[l] = adam_update(dW[l], iter, m_W[l], v_W[l])
        W[l].assign(W[l] - lr * dW_hat)
        
        db_hat, m_b[l], v_b[l] = adam_update(db[l], iter, m_b[l], v_b[l])
        b[l].assign(b[l] - lr * db_hat)
            
    return dW_norm, m_W, v_W, m_b, v_b

# --------------
# Prepare training data
def mini_batch(vf_ts, x_ts, x_T, ts = ts, ratios = ratios, n_t = p['mini_batch_size_t'], n_x = p['mini_batch_size_x']):
    # vf_ts / x_ts: nested list of velocity / position evaluations at different time points
    #               primary level) different GPA runs in time order
    #               secondary level) velocities / positions at different time steps for each GPA run
    # ts: list of physical time points for v
    # c: real constant that is a multiplier of velocity
    # n_t: number of different time t to be sampled
    # n_x = all samples (= number of samples alloted to the same time t)
    true_vs, true_xs, true_ts= [], [], []

    for i, (vf_t, x_t) in enumerate(zip(vf_ts, x_ts)):
        if i == len(vf_ts)-1:
            n_t_1 = n_t - i * int(n_t/len(vf_ts))
        else:
            n_t_1 = int(n_t/len(vf_ts))
            
        dt =  (ts[i+1] - ts[i])/ len(vf_t)  # time step size for adapted to time horizon of v(x,t)
        t_ids = np.random.choice(len(vf_t), n_t_1, replace=False) # sample n_t_1 time points
        
        if vf_t[0].shape[0] > n_x:
            x_ids = np.random.choice(vf_t[0].shape[0], n_x, replace=False) # sample n_x spatial points
        else:
            x_ids = np.arange(0, vf_t[0].shape[0])
        
        t = dt * t_ids + ts[i]  #  sampled time for v(x,t)
        # return their corresponding velocity, x(t), t values
        for j, t_id in enumerate(t_ids):
            true_vs.append(ratios[i] * vf_t[t_id][x_ids])
            true_xs.append(x_t[t_id][x_ids])
            true_ts.append(np.ones([true_xs[-1].shape[0],1]) * t[j])
            
        # always use the real data
        true_vs.append(ratios[i] * vf_t[0][x_ids])                  # source/intermediate target
        true_xs.append(x_t[0][x_ids])                               # source/intermediate target
        true_ts.append(np.ones([true_xs[-1].shape[0],1]) * ts[i])   # source/intermediate target
        
    # always use the real data
    true_xs.append(x_T[x_ids])                                      # stationary target
    true_vs.append(np.zeros_like(true_xs[-1]))                      # stationary target
    true_ts.append(np.ones([true_xs[-1].shape[0],1]) * ts[-1])      # stationary target
        
    return np.vstack(true_vs), np.vstack(true_xs), np.vstack(true_ts)

# --------------
# Train
for it in range(1,iterations+1):
    true_v, x, t = mini_batch(vectorfields, trajectories, param['X_'])
    true_v = tf.constant(true_v, dtype=tf.float32)
    x = tf.constant(x, dtype=tf.float32)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch([W, b])
        mse_loss = MSE(true_v, v(x, t, W, b))
        loss = mse_loss
        
    dW, db = tape.gradient(loss, [W,b])

    if it == 1:
        m_W, v_W, m_b, v_b = [0]*len(W), [0]*len(W), [0]*len(W), [0]*len(W)
    dW_norm, m_W, v_W, m_b, v_b = adam(lr_NN, it, W, dW, m_W, v_W, b, db, m_b, v_b) # update vectorfield v([x,t]; W, b)
    
    mse_loss_eval = mse_loss.numpy()
    if it % int(iterations/10) == 0:
        print(f"Iter {it}: MSE = {mse_loss_eval}")
        
W = [w.numpy() for w in W]
b = [b_.numpy() for b_ in b]

# converted t
p['numerical_ts'] = ts
    
save_filename = assets_dir + p['dataset'] + '/' + p['exp_memo'] + '.pickle'
with open(save_filename,"wb") as fw:
    pickle.dump([W, b, p], fw)
print("Results saved at:", save_filename)
