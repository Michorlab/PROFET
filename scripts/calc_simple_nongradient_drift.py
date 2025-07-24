#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# argv[1]: dataset name in ['Transport_genes', 'Test_transport_genes']
# argv[2]: size of PCA-reduced dimension in case of dataset name = 'Transport_genes'
# ----------------------------------------
import os
import numpy as np
import sys
import pandas as pd

current_filename = __file__
data_dir = current_filename.split('scripts')[0]
data_dir = data_dir + 'data/'

days_mat = {}
if sys.argv[1] == 'Transport_genes':
    N_dim = int(sys.argv[2])
    
    # load gene expression data
    filename = "emt_pca_projected.csv"
    full_mat = np.genfromtxt(data_dir + filename, delimiter=',')
    reduced_mat = full_mat[:,:N_dim]
    
    # load day info
    df_cls = pd.read_table(data_dir + 'Cells_Days.txt', sep="\t")
    
    days = sorted(set(df_cls['day']))
    for day in  set(df_cls['day']):
        days_mat[day] = reduced_mat[df_cls['day']==day]
    
elif sys.argv[1] == 'Test_transport_genes':
    if len(sys.argv) > 2:
        N_dim = int(sys.argv[2])
    else:
        N_dim = 26
    
    days = [0,1,2,3,4]
    
    for day in days:
        df = pd.read_table(data_dir + 'traj_dataset/log_traj_%d.log' % day, sep="\t", header=None)
        days_mat[day] = np.array(df)[:,:-1]

N_choices = 500
idxs = []
avg_directions = []
for day_i in days[:-1]:
    for day_f in days[0:]:
        if day_i >= day_f:
            continue
        idx1 = np.random.randint(0, days_mat[day_i].shape[0], size=N_choices)
        idx2 = np.random.randint(0, days_mat[day_f].shape[0], size=N_choices)

        directions = days_mat[day_f][idx2,:] - days_mat[day_i][idx1,:]
        avg_directions.append(np.mean(directions, axis=0))
        idxs.append([day_i, day_f])
    

avg_directions = np.vstack(avg_directions)
idxs = np.vstack(idxs)

# Save & plot settings -----------------------------------------------

filename = data_dir + sys.argv[1]+'_simple_avg_directions_dim%d.csv' % N_dim


avg_directions = pd.DataFrame(avg_directions)
idxs = pd.DataFrame(idxs)
df = pd.concat([idxs, avg_directions], axis=1)
df.to_csv(filename, sep='\t')
