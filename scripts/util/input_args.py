import argparse
    
def input_params():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument(
        '-N_Q', '--N_samples_Q', type=int, help='total number of target samples', # default=200,
    )
    parser.add_argument(
        '-N_P', '--N_samples_P', type=int, help='total number of prior samples', #default=200,
    )
    parser.add_argument(
        '-N_dim', type=int, help='dimension of input data',
    )
    parser.add_argument(
        '--random_seed', type=int, help='random seed for data generator', #default=0,
    )
    
    # Dataset property
    parser.add_argument(
        '--dataset', type=str, default='Transport_genes',
    )
    parser.add_argument(
        '--label', type=int, nargs='+', help='list of the indices of source and target days',
    )
    parser.add_argument(
        '--reduction', type=str, help='dimension reduction method',
    )
    parser.add_argument(
        '-rescale_factor', type=float, default = 1.0, help='rescale data X by X*rescale_factor',
    )
    
    # Mixture GPA settings
    parser.add_argument(
        '-iter_transient_target', type=int, nargs='+',
        help='approximate iterations to reach transient target states (label[:-1]) for Mixture GPA. If not specified, it will be determined proportional to label[:-1] values within the total iterations (epochs).'
    )
    parser.add_argument(
        '-weight_decay_mixture_target', type=float,
        help='the constant c for the weight decay which is exponential in time: \sum_i exp(-c*t_i) X_t_i where t_i=iter_transient_target, X_t_i=target samples at time t_i')

    
    
    # (f, Gamma)-divergence
    parser.add_argument(
        '--f', type=str, choices=['KL', 'alpha', 'reverse_KL', 'reverse_alpha'], #default='KL',
    )
    parser.add_argument(
        '-alpha', type=float, help='parameter value for alpha divergence',
    )    
    parser.add_argument(
        '--formulation', type=str, choices=['LT', 'DV'], help='LT or DV in case of f=KL, otherwise, keep LT', #default='LT',
    )
    parser.add_argument(
        '--Gamma', type=str, choices=['Lipshitz'], #default='Lipshitz',
    )
    parser.add_argument(
        '-L', type=float, help='Lipshitz constant: default=inf w/o constraint',
    )
    parser.add_argument(
        '--reverse', type=bool,  help='True -> D(Q|P), False -> D(P|Q)', #default=False,
    )
    parser.add_argument(
        '--constraint', type=str, choices=['hard', 'soft'], #default='hard',
    )
    parser.add_argument(
        '-lamda', type=float,  help='coefficient on soft constraint', #default=100.0,
    )
    parser.add_argument(
        '--generative_model', type=str, default="GPA_NN", # choices=['GPA_NN', 'GPA_NN_w_nongradient_drift'],
    )
    
    
    # Neural Network <phi>
    parser.add_argument(
        '-NN', '--NN_model', type=str,  choices=['fnn', 'cnn', 'cnn-fnn'], #default='fnn',
    )
    parser.add_argument(
        '-N_fnn_layers', type=int, nargs='+', help='list of the number of FNN hidden layer units / the number of CNN feed-forward hidden layer units',
    )
    parser.add_argument(
        '-N_cnn_layers', type=int, nargs='+', help='list of the number of CNN channels',
    )
    parser.add_argument(
        '--activation_ftn', type=str, nargs='+',  choices=['relu', 'mollified_relu_cos3','mollified_relu_poly3','mollified_relu_cos3_shift','softplus', 'leaky_relu','elu', 'bounded_relu', 'bounded_elu'], help='[0]: for the fnn/convolutional layer, [1]: for the cnn feed-forward layer, [2]: for the LAST cnn feed-forward layer', #default=['relu',],
    )
    parser.add_argument(
        '-eps', type=float,  help='Mollifier shape adjusting parameter when using mollified relu3 activations', #default = 0.5,
    )
    parser.add_argument(
        '--N_conditions', type=int, help='number of classes for the conditional setting', #default=1,
    )
    
    
    # Discriminator training parameters
    parser.add_argument(
        '--lr_phi', type=float, help='lr for phi',# default=0.001,
    )
    parser.add_argument(
        '-ep_phi', '--epochs_phi', type=int, help='# updates for phi to find phi*',# default=3,
    )
    parser.add_argument(
        '--optimizer', type=str, choices=['sgd', 'adam',], help='optimizer for NN',# default='adam',
    )
    
    # Particles transportation parameters
    parser.add_argument(
        '-ep', '--epochs', type=int, help='# updates for P', #default=1000,
    )
    parser.add_argument(
        '-termination_error', type=float, help='error threshold to terminate', default=0.,
    )
    parser.add_argument(
        '-termination_error_metric', type=str, help='error type to be used for termination', choices=['KE', 'f_Lip']
    )
    parser.add_argument(
        '--ode_solver', type=str, choices=['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5', 'ABM1', 'Heun', 'ABM2', 'ABM3', 'ABM4', 'ABM5', 'RK4', 'ode45', 'Newton', 'BFGS','Gradient_Ascent'], help='ode solver for particle ode', #default='forward_euler',
    )
    parser.add_argument(
        '-mobility', type=str, help='problem dependent mobility function\nRecommendation: MNIST - bounded',
    )
    parser.add_argument(
        '-lr_P_decay', type=str, choices=['rational', 'step',], help='delta t decay',
    )
    parser.add_argument(
        '--lr_P', type=float, help='lr for P',#default=1.0,
    )
    parser.add_argument(
        '-ratio_nongrad_2_grad', type=float, help='ratio of nongradient vectorfield to gradient vectorfield',# default=1.0,
    )
    
    parser.add_argument(
        '--exp_no', type=str, help='short experiment name under the same data', #default=0,
    )
    parser.add_argument(
        '--mb_size_P', type=int, help='mini batch size for the moving distribution P',# default=200,
    )
    parser.add_argument(
        '--mb_size_Q', type=int, help='mini batch size for the target distribution Q',# default=200,
    )
    
    
    
    # save/display
    parser.add_argument(
        '--save_iter', type=int, help='save results per each save_iter',# default=10,
    )
    parser.add_argument(
        '--plot_result', type=bool, help='True -> plot results',# default=False,
    )
    #parser.add_argument(
    #    '--show_plots', type=bool, help='True -> show plots',# default=False,
    #)
    parser.add_argument(
        '--plot_intermediate_result', type=bool, help='True -> save intermediate plots',# default=False,
    )
    parser.add_argument(
        '-filename', type=str, help='filename for GPA_NN-Generator',# default='adam',
    )
    parser.add_argument(
        '-plot_axes', type=int, nargs='+',  help='For dimension > 2, specify 2 plotting axes', default=[0,1],
    )
    
    return parser.parse_known_args()
