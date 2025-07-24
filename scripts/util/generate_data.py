import numpy as np
import tensorflow as tf
import sys
import pandas as pd

current_filename = __file__
data_dir = current_filename.split('scripts')[0]
data_dir = data_dir + 'data/'

def reduce_data(Q, data_par, rescale_factor=1.0):
    X_new = Q.numpy()
    ts = sorted(set(data_par['Q_time_label']))
    X_new = X_new[data_par['Q_time_label'] != ts[0]]
    if type(data_par['Q_label']) != type(None):
        data_par['Q_label'] = data_par['Q_label'][data_par['Q_time_label'] != ts[0]]
    data_par['Q_time_label'] = data_par['Q_time_label'][data_par['Q_time_label'] != ts[0]]
    data_par['N_samples_Q'] = X_new.shape[0]
    data_par['mb_size_Q'] = X_new.shape[0]
    
    Q = tf.constant(X_new/rescale_factor, dtype=tf.float32) # constant
    #print(data_par['Q_time_label'])
    print('Target data were reduced.')
    
    return Q, data_par

def generate_one_hot_encoding(
    N: int,
    N_classes: int,
    random_seed: int,
    data=[]
):
    if data == []:
        np.random.seed(random_seed)
        data = np.random.randint(N_classes, size=N)
        
    data = np.ndarray.flatten(data)
    X_label = np.zeros((data.size, data.max()+1))
    X_label[np.arange(data.size),data] = 1
    
    return X_label
    
def time_labeled_data(
    X_ts,#: list
    ts
):
    X_time_label = []
    for t, X_t in zip(ts, X_ts):
        X_time_label.append(t*np.ones(X_t.shape[0]))
    X_ = np.concatenate(X_ts) # target
    X_time_label = np.concatenate(X_time_label) # target labeled by time
    
    return X_, X_time_label
    
def import_gene_expression_data_short(
    N_dim: int,
    label, #:list(int),
    reduction #: str,
):
    from numpy import genfromtxt
    
    #print(data_dir)
    
    if reduction == 'EMT_PCA':
        filename = "emt_pca_projected.csv"
    elif reduction == 'PCA':
        filename = "pca_projected.csv"
    elif reduction == 'EMT':
        filename = "Gene_Expression_Matrix-EMT_Hallmark_genes.txt"
    else:
        print("Dataset for the reduction method is not available")
    
    if "csv" in filename:
        full_mat = genfromtxt(data_dir + filename, delimiter=',')
    elif "txt" in filename:
        full_mat = np.loadtxt(data_dir + filename, delimiter='\t', skiprows=1, usecols=range(1,176,1))
   
    reduced_mat = full_mat[:,:N_dim]
    
    # Discriminate source and target
    df_cls = pd.read_table(data_dir + 'Cells_Days.txt', sep="\t")
    X = reduced_mat[df_cls['day']==label]
    X_label = []
    
    return X, X_label
    
        
def import_gene_expression_data(
    N_dim: int,
    label, #:list(int),
    reduction, #: str,
    N_conditions: int,
    random_seed: int,
    target_label_iter
):
    from numpy import genfromtxt
    
    #print(data_dir)
    
    if reduction == 'EMT_PCA':
        filename = "emt_pca_projected.csv"
    elif reduction == 'PCA':
        filename = "pca_projected.csv"
    elif reduction == 'EMT':
        filename = "Gene_Expression_Matrix-EMT_Hallmark_genes.txt"
    else:
        print("Dataset for the reduction method is not available")
    
    if "csv" in filename:
        full_mat = genfromtxt(data_dir + filename, delimiter=',')
    elif "txt" in filename:
        full_mat = np.loadtxt(data_dir + filename, delimiter='\t', skiprows=1, usecols=range(1,176,1))
   
    reduced_mat = full_mat[:,:N_dim]
    
    # Discriminate source and target
    df_cls = pd.read_table(data_dir + 'Cells_Days.txt', sep="\t")
    Y_ = reduced_mat[df_cls['day']==label[0]] # source
    
    print(label)
    X_ts = []
    
    if len(target_label_iter) == 1: # Original GPA or Regularized GPA
        X_ts.append(reduced_mat[df_cls['day']==label[-1]])
    else:  # regularized GPA
        for t in np.sort(label[1:]):
            X_ts.append(reduced_mat[df_cls['day']==t])
    X_, X_time_label = time_labeled_data(X_ts, target_label_iter) # target, time label on individual samples
        
    if N_conditions == 1:
        X_label, Y_label = [], []
    else:
        # load label data
        X_data = genfromtxt(data_dir + 'target_label.csv', delimiter=',', dtype=np.int32)
        Y_data = genfromtxt(data_dir + 'source_label.csv', delimiter=',', dtype=np.int32)
        
        #print(sum(X_data), len(X_data)-sum(X_data), sum(Y_data), len(Y_data)-sum(Y_data))
        
        X_label = generate_one_hot_encoding(N=len(X_data), N_classes=N_conditions, random_seed = random_seed, data=X_data)
        Y_label = generate_one_hot_encoding(N=len(Y_data), N_classes=N_conditions, random_seed = random_seed, data=Y_data)
    
    return X_, X_label, Y_, Y_label, X_time_label
    
def import_test_gene_expression_data_short(
    label #:list(int),
):
    df = pd.read_table(data_dir + 'traj_dataset/log_traj_%d.log' % label, sep="\t", header=None)
    X = np.array(df)[:,:-1]
    X_label = []
    return X, X_label
    
def import_test_gene_expression_data(
    label, #:list(int),
    N_conditions: int,
    random_seed: int,
    target_label_iter
):
    df = pd.read_table(data_dir + 'traj_dataset/log_traj_%d.log' % label[0], sep="\t", header=None)
    Y_ = np.array(df)[:,:-1]
    
    
    print(label)
    X_ts = []
    if len(target_label_iter) == 1: # Original GPA or Regularized GPA
        df = pd.read_table(data_dir + 'traj_dataset/log_traj_%d.log' % label[-1], sep="\t", header=None)
        X_ts.append(np.array(df)[:,:-1])
    else:  # Mixture GPA
        for t in np.sort(label[1:]):
            df = pd.read_table(data_dir + 'traj_dataset/log_traj_%d.log' % t, sep="\t", header=None)
            X_ts.append(np.array(df)[:,:-1])
            
    X_, X_time_label = time_labeled_data(X_ts, target_label_iter) # target, time label on individual samples
        
    #df = pd.read_table(data_dir + 'traj_dataset/log_traj_%d.log' % label[1], sep="\t", header=None)
    #X_ = np.array(df)[:,:-1]
    
    if N_conditions == 1:
        X_label, Y_label = [], []
    else:
        # load label data
        X_data = genfromtxt(data_dir + 'target_label.csv', delimiter=',', dtype=np.int32)
        Y_data = genfromtxt(data_dir + 'source_label.csv', delimiter=',', dtype=np.int32)
        
        #print(sum(X_data), len(X_data)-sum(X_data), sum(Y_data), len(Y_data)-sum(Y_data))
        
        X_label = generate_one_hot_encoding(N=len(X_data), N_classes=N_conditions, random_seed = random_seed, data=X_data)
        Y_label = generate_one_hot_encoding(N=len(Y_data), N_classes=N_conditions, random_seed = random_seed, data=Y_data)
    
    return X_, X_label, Y_, Y_label, X_time_label



def generate_data(param):
    # Input
    # param: parameters dictionary
    # Outputs
    # X_, Y_, [X_label, Y_label]
    # param
    if param['N_dim'] == None:
        param['N_dim'] = 2
        
    sys.path.append('../')
    
    if param['dataset'] == 'Transport_genes':
        labels = "_".join(str(l) for l in param['label'])
        param['expname'] = "%s-%stimes-%s_dim%d" % (param['expname'], labels, param['reduction'], param['N_dim'])
        
        if 'weight_decay_mixture_target' in param.keys(): #Mixture GPA
            target_label_iter = param['iter_transient_target'] + [param['epochs']]
        else:
            target_label_iter = [param['epochs']]
        
        X_, X_label, Y_, Y_label, X_time_label = import_gene_expression_data(N_dim=param['N_dim'], label=param['label'], reduction=param['reduction'], N_conditions=param['N_conditions'], random_seed=param['random_seed'], target_label_iter=target_label_iter)
        if 'weight_decay_mixture_target' in param.keys():
            param['X_time_label'] = X_time_label
    
    elif param['dataset'] == 'Test_transport_genes':
        labels = "_".join(str(l) for l in param['label'])
        param['expname'] = "%s-%stimes" % (param['expname'], labels)
        
        if 'weight_decay_mixture_target' in param.keys():  #Mixture GPA
            target_label_iter = param['iter_transient_target'] + [param['epochs']]
        else:
            target_label_iter = [param['epochs']]
        
        X_, X_label, Y_, Y_label, X_time_label = import_test_gene_expression_data(label=param['label'], N_conditions=param['N_conditions'], random_seed=param['random_seed'], target_label_iter=target_label_iter)
        param['N_dim'] = X_.shape[1]
        
        if 'weight_decay_mixture_target' in param.keys():  #Mixture GPA
            param['X_time_label'] = X_time_label
        
    elif param['dataset'] == 'Toy1':
        if "true_target" in param['exp_no']:
            X_mean, X_std = 4.0, 1.0
            X_ = np.random.normal(loc = X_mean, scale = X_std, size=(param['N_samples_Q'], param['N_dim']))
            
        else:
            import pickle
            
            if 'label' in param.keys():
                labels = "_".join(str(l) for l in param['label'])
                param['expname'] = "%s-%stimes" % (param['expname'], labels)
                
                load_filename = data_dir + "ou_result_%s_multiple_times.pickle" % str(param['exp_no'])
                with open(load_filename, "rb") as fr:
                    X_, param['X_time_label'] = time_labeled_data(pickle.load(fr), param['label']) # target, time label on individual samples
            else:
                load_filename = data_dir + "ou_result_%s.pickle" % str(param['exp_no'])
                with open(load_filename, "rb") as fr:
                    X_ = pickle.load(fr)[0]
                
        Y_means, Y_stds = [0.0, 4.0], [0.5, 1.0]
        idx_0 = np.random.random((param['N_samples_P'],1))>2/3
        Y_ = idx_0 * np.random.normal(loc = Y_means[0], scale = Y_stds[0], size=( param['N_samples_P'], param['N_dim'])) + \
        (1-idx_0) * np.random.normal(loc = Y_means[1], scale = Y_stds[1], size=( param['N_samples_P'], param['N_dim']))
        
    param['N_samples_Q'], param['N_samples_P'] = len(X_), len(Y_)
    if param['mb_size_P'] > param['N_samples_P']:
        param['mb_size_P'] = param['N_samples_P']
    if param['mb_size_Q'] > param['N_samples_Q']:
        param['mb_size_Q'] = param['N_samples_Q']
    param['mb_size_Q'], param['mb_size_P'] = param['N_samples_Q'], param['N_samples_P']
    
    print(param['mb_size_Q'], param['mb_size_P'])
        
    if 'X_label' not in locals():
        X_label = None
    if 'Y_label' not in locals():
        Y_label = None
    return param, X_, Y_, X_label, Y_label
