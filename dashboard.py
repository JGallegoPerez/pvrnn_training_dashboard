import plotly.graph_objects as go
import dash
from dash import html
from dash import dcc 
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import utils as utils
import os
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-t', type=str, dest='title', help='The title to give to the dashboard. Between quotes.')
parser.add_argument('-d', type=str, dest='results_dir_path', help='The path of the directory where the result subdirectories are located')
parser.add_argument('-r', type=str, nargs='+', dest='results_list', help='The results subdirectories (each folder represented as a cluster job number). Separated by spaces.')
args = parser.parse_args()

title = args.title
results_dir_path = args.results_dir_path
results_list = args.results_list

def get_ws(results_lst):
    """
    Takes as input a list of the names of all the result directories.
    Returns a list of strings corresponding to each result directory. 
    """
    w_pairs_dict = {}
    for r in results_lst:
        r_config_path = os.path.join(results_dir_path, r, 'config.toml')
        assert os.path.isfile(r_config_path), "Incorrect config file path."
        with open(r_config_path, 'r') as file:
            w_pair = []
            for line in file:
                if line.startswith("w = "):
                    w_pair.append(line[len("w = "):].split('\n')[0])
                if len(w_pair) >= 2:
                    break
        w_str = 'Bottom: ' + w_pair[0] + '; Top: ' + w_pair[1]
        w_pairs_dict[w_str] = r
    print(w_pairs_dict)
    return w_pairs_dict, list(w_pairs_dict.keys())
res_dict, ws_lst = get_ws(results_list) 

def get_seq_len(results_lst):
    """
    Takes the list of names of the result directories.
    Returns the sequences' length (int). 
    """
    r1 = results_lst[0]
    r1_config_path = os.path.join(results_dir_path, r1, 'config.toml')
    assert os.path.isfile(r1_config_path), "Incorrect config file path."
    with open(r1_config_path, 'r') as file:
        for line in file:
            if line.startswith("seq_len = "):
                seq_len = int(line[len("seq_len = "):])
    return seq_len

seq_len = get_seq_len(results_list)

app = dash.Dash(__name__)

# Build dash app layout 
app.layout = html.Div(children=[
    html.H1(title, style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    html.Div([
        html.Div("w values: ", style={'textAlign': 'center', 'font-size': 20}),
        dcc.Dropdown(options=ws_lst, value=ws_lst[0], id='w_values', style={'height': '35px', 'width': '240px', 'font-size': 16, 'margin': '0 auto'}),
    ]),
    html.Br(),
    html.Br(),
    # Segment 1
    html.Div([
        html.Div(dcc.Graph(id='errors-plot')),
        html.Div(dcc.Graph(id='weights-plot'))
    ], style={'display': 'flex'}),
    html.Div([
        html.Div("Timesteps range: ", style={'font-size': 20}),
        dcc.Slider(min=0, max=seq_len, step=int(seq_len/20), value=seq_len, id='range_slider'),
    ]),
    # Segment 2
    html.Div([
        html.Div(dcc.Graph(id='layer1-plot')),
        html.Div(dcc.Graph(id='layer2-plot'))
    ], style={'display': 'flex'})
])

def compute_info(train_res, w_str, timesteps):
    """
    Takes as argument a dictionary of several training results subdirectories (strings) with their w values (string);
    a string object that corresponds to the w values to work on;
    and the number of timesteps to perform computations on. 
    Returns the Plotly graphs. 
    """
    result_subdir = train_res[w_str]

    ## SET PATHS OF VARIABLES OF INTEREST AND EXTRACT FILES ##

    # Directory from a specific result
    train_dir = os.path.join(results_dir_path, result_subdir, 'results/train')

    def var_path_setter(vars, subdir):
        """
        Takes as first argument the variables the variables we want to display, as a list of strings.
        The second argument is the subdirectory where the variables are. 
        Returns specific paths to .npy files, as a list of strings. It also returns a list of the corresponding .npz file paths. 
        It searches inside .npz files, and fetches data from the first training sequence stored in each corresponding .npz file.      
        """
        var_paths = []
        npz_paths = []
        for v in vars:
            var_paths.append(os.path.join(train_dir, subdir, v + '_0.npy'))
            npz_paths.append(os.path.join(train_dir, subdir, v + '.npz'))
        return var_paths, npz_paths

    # Paths of variables of interest, Bottom layer
    vars1 = ['d', 'mup', 'muq', 'sigmap', 'sigmaq']
    subdir1 = 'sequences/epo_200000/layer1'
    var_paths1, npz_paths1 = var_path_setter(vars1, subdir1)
    # Unzip the paths from the corresponding .npz files
    for (y, z) in zip(var_paths1, npz_paths1):
        utils.unzip(z, os.path.basename(y))                        

    # Paths of variables of interest, Top layer
    vars2 = ['d', 'mup', 'muq', 'sigmap', 'sigmaq']
    subdir2 = 'sequences/epo_200000/layer2'
    var_paths2, npz_paths2 = var_path_setter(vars2, subdir2)
    # Unzip the paths from the corresponding .npz files
    for (y, z) in zip(var_paths2, npz_paths2):
        utils.unzip(z, os.path.basename(y))   

    # Variable x (output)
    vars_x = ['x']
    subdir_x = 'sequences/epo_200000/output'
    var_path_x, npz_path_x = var_path_setter(vars_x, subdir_x)
    # Unzip the paths from the corresponding .npz files
    for (y, z) in zip(var_path_x, npz_path_x):
        utils.unzip(z, os.path.basename(y)) 

    # Target (we will use only the first sequence later). (3D array). 
    target_path = '/home/jorge/Code/LibPvrnn/datasets/touchlift_14_3600_14.npy' ### NEEDS TO BE ADDED TO REPO ###

    # Obtain ndarrays for the plots for the two layers 
    var_begin = 0
    var_end = timesteps #Timestep range to visualize. Set up by user (slider). 
    d1_arr = utils.npy_to_array(var_paths1[0])[var_begin:var_end,:4]
    mup1_arr = utils.npy_to_array(var_paths1[1])[var_begin:var_end,:1]
    muq1_arr = utils.npy_to_array(var_paths1[2])[var_begin:var_end,:1]
    sigmap1_arr = utils.npy_to_array(var_paths1[3])[var_begin:var_end,:1]
    sigmaq1_arr = utils.npy_to_array(var_paths1[4])[var_begin:var_end,:1]
    d2_arr = utils.npy_to_array(var_paths2[0])[var_begin:var_end,:4]
    mup2_arr = utils.npy_to_array(var_paths2[1])[var_begin:var_end,:1]
    muq2_arr = utils.npy_to_array(var_paths2[2])[var_begin:var_end,:1]
    sigmap2_arr = utils.npy_to_array(var_paths2[3])[var_begin:var_end,:1]
    sigmaq2_arr = utils.npy_to_array(var_paths2[4])[var_begin:var_end,:1]
    x_arr = utils.npy_to_array(var_path_x[0])[var_begin:var_end,:]
    target_arr = utils.npy_to_array(target_path)[0,var_begin:var_end,:]

    ## PLOTS ##

    # PLOT ERROR PLOTS
    error_subdir = 'learning'
    rec_error_path = os.path.join(train_dir, error_subdir, 'recErr.txt')
    kld1_path = os.path.join(train_dir, error_subdir, 'layer1_kld.txt')
    kld2_path = os.path.join(train_dir, error_subdir, 'layer2_kld.txt')
    rec_error = np.loadtxt(rec_error_path, delimiter=' ')
    kld1 = np.loadtxt(kld1_path, delimiter=' ')
    kld2 = np.loadtxt(kld2_path, delimiter=' ')

    fig_errors = make_subplots(rows=3, cols=1)
    # Reconstruction error
    fig_errors.add_trace(
        go.Scatter(x=list(range(len(rec_error))), y=rec_error, name="rec error", legendgroup='1'),
        row=1, col=1)    
    # KLD1 error
    fig_errors.add_trace(
        go.Scatter(x=list(range(len(kld1))), y=kld1, name="kld1 error", legendgroup='2'),
        row=2, col=1)    
    # KLD2 error
    fig_errors.add_trace(
        go.Scatter(x=list(range(len(kld2))), y=kld2, name="kld2 error", legendgroup='3'),
        row=3, col=1)    
    # fig_errors.show()
    fig_errors.update_layout(height=600, width=1200, title_text="ERROR THROUGHOUT TRAINING (log scale)", 
        yaxis1_title = 'Rec. error',
        yaxis2_title = 'KLD L_layer',
        yaxis3_title = 'KLD H_layer',
        yaxis1=dict(tickformat='.2f'),
        yaxis2=dict(tickformat='.2f'),    
        yaxis3=dict(tickformat='.2f'),     
        showlegend=False)
    fig_errors.update_yaxes(type='log')

    # PLOT WEIGHTS FROM WEIGHT MATRICES OF INTEREST

    # Get full paths of where the parameters are (at epoch=200000)
    # Bottom layer:
    l1_path = os.path.join(train_dir, 'parameters/epo_200000/layer1/W.npz')
    # Top layer: 
    l2_path = os.path.join(train_dir, 'parameters/epo_200000/layer2/W.npz')

    # Unzip the relevant weight matrices
    data_l1 = np.load(l1_path)
    data_l2 = np.load(l2_path)

    fig_weights = go.Figure()
    # Add the first boxplot trace
    trace1 = go.Box(y=data_l1['Wzh.npy'].flatten(), name='L_Wzh', boxpoints='all')
    fig_weights.add_trace(trace1)
    # Add the second boxplot trace
    trace2 = go.Box(y=data_l1['Wdh.npy'].flatten(), name='L_Whd', boxpoints='all')
    fig_weights.add_trace(trace2)
    # Add the third boxplot trace
    trace3 = go.Box(y=data_l1['Whdh.npy'].flatten(), name='Wdd', boxpoints='all')
    fig_weights.add_trace(trace3)
    # Add the fourth boxplot trace
    trace4 = go.Box(y=data_l2['Wzh.npy'].flatten(), name='H_Wzh', boxpoints='all')
    fig_weights.add_trace(trace4)
    # Add the fifth boxplot trace
    trace5 = go.Box(y=data_l2['Wdh.npy'].flatten(), name='H_Whd', boxpoints='all')
    fig_weights.add_trace(trace5)
    # fig_weights.show()
    fig_weights.update_layout(height=600, width=1200,
        title='VALUES OF WEIGHTS',
        xaxis=dict(title='Weight matrix'),
        yaxis=dict(title='Value'),    
        yaxis1=dict(tickformat='.2f'),                          
        showlegend=False)

    # PLOT LAYER 1
    def arr2d_trace_iter(fig, arr2d, num_lines, trace_row, trace_col, color=None, legend_group='1', name="name"):
        if color is None:
            for i in range(num_lines):
                fig.add_trace(
                    go.Scatter(x=list(range(len(arr2d))), y=arr2d[:, i], legendgroup=legend_group, name=name),
                    row=trace_row, col=trace_col)
        else:
            for i in range(num_lines):            
                fig.add_trace(
                    go.Scatter(x=list(range(len(arr2d))), y=arr2d[:, i], line=dict(color=color), legendgroup=legend_group, name=name),
                    row=trace_row, col=trace_col)
                
    fig_layer1 = make_subplots(rows=4, cols=1)
    # Layer 1: mup, muq
    arr2d_trace_iter(fig_layer1, mup1_arr, 1, 1, 1, color=None, legend_group='1', name='mup')
    arr2d_trace_iter(fig_layer1, muq1_arr, 1, 1, 1, color=None, legend_group='1', name='muq')
    # Layer 1: sigmap, sigmaq
    arr2d_trace_iter(fig_layer1, sigmap1_arr, 1, 2, 1, color=None, legend_group='2', name='sigmap')
    arr2d_trace_iter(fig_layer1, sigmaq1_arr, 1, 2, 1, color=None, legend_group='2', name='sigmaq')
    # Layer 1: d
    arr2d_trace_iter(fig_layer1, d1_arr, 4, 3, 1, color=None, legend_group='3', name='d')
    # Layer 1: x, target
    arr2d_trace_iter(fig_layer1, x_arr, 3, 4, 1, color=None, legend_group='4', name='output')
    arr2d_trace_iter(fig_layer1, target_arr, 3, 4, 1, color="grey", legend_group='4', name='target')

    fig_layer1.update_layout(height=600, width=1200, title_text="BOTTOM LAYER", 
        yaxis1_title = 'myu',
        yaxis2_title = 'sigma',
        yaxis3_title = 'd ',   
        yaxis4_title = 'output, target',     
        yaxis1=dict(tickformat='.2f'),
        yaxis2=dict(tickformat='.2f'),    
        yaxis3=dict(tickformat='.2f'),    
        yaxis4=dict(tickformat='.2f'),                      
        legend_tracegroupgap = 65)
    fig_layer1.update_traces(texttemplate='%{value:.2f}')

    # PLOT LAYER 2
    fig_layer2 = make_subplots(rows=4, cols=1)
    # Layer 1: mup, muq
    arr2d_trace_iter(fig_layer2, mup2_arr, 1, 1, 1, color=None, legend_group='1', name='mup')
    arr2d_trace_iter(fig_layer2, muq2_arr, 1, 1, 1, color=None, legend_group='1', name='muq')
    # Layer 1: sigmap, sigmaq
    arr2d_trace_iter(fig_layer2, sigmap2_arr, 1, 2, 1, color=None, legend_group='2', name='sigmap')
    arr2d_trace_iter(fig_layer2, sigmaq2_arr, 1, 2, 1, color=None, legend_group='2', name='sigmaq')
    # Layer 1: d
    arr2d_trace_iter(fig_layer2, d2_arr, 4, 3, 1, color=None, legend_group='3', name='d')
    # Layer 1: x, target
    arr2d_trace_iter(fig_layer2, x_arr, 3, 4, 1, color=None, legend_group='4', name='output')
    arr2d_trace_iter(fig_layer2, target_arr, 3, 4, 1, color="grey", legend_group='4', name='target')
    fig_layer2.update_layout(height=600, width=1200, title_text="TOP LAYER", 
        yaxis1_title = 'myu',
        yaxis2_title = 'sigma',
        yaxis3_title = 'd ',   
        yaxis4_title = 'output, target',     
        yaxis1=dict(tickformat='.2f'),
        yaxis2=dict(tickformat='.2f'),    
        yaxis3=dict(tickformat='.2f'),    
        yaxis4=dict(tickformat='.2f'),                      
        legend_tracegroupgap = 65)

    return fig_errors, fig_weights, fig_layer1, fig_layer2


# Callback decorator
@app.callback( [
               Output(component_id='errors-plot', component_property='figure'),
               Output(component_id='weights-plot', component_property='figure'),
               Output(component_id='layer1-plot', component_property='figure'),
               Output(component_id='layer2-plot', component_property='figure')
               ],
               Input('w_values', 'value'),
               Input('range_slider', 'value'))
# Computation to callback function and return graph
def get_graph(w_values, range_slider):
    fig_errors, fig_weights, fig_layer1, fig_layer2 = compute_info(res_dict, w_values, range_slider)
    return [fig_errors, fig_weights, fig_layer1, fig_layer2]

# Run the app
if __name__ == '__main__':
    app.run_server()




















