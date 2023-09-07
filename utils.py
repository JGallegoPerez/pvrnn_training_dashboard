import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os
import csv

def txt_to_array(path):
    with open(path) as f:
        lines = [line for line in f]
    with open(path) as f:
        lines = [float(line.rstrip()) for line in f]
    arr = np.array(lines).reshape(len(lines), 1)
    return arr

def npy_to_array(path):
    arr = np.load(path)
    return arr

def file_to_array(path):
    file_format = (path.split('/')[-1]).split('.')[1]
    if file_format == 'npy':
        arr = np.load(path)
    elif file_format == 'txt':
        with open(path) as f:
            lines = [line for line in f]
        with open(path) as f:
            lines = [float(line.rstrip()) for line in f]
        arr = np.array(lines).reshape(len(lines), 1)
    return arr

def unzip(path_fetch, var_name):
    path_save = os.path.dirname(path_fetch)  # Saves the unzipped file in the parent directory.
    with ZipFile(path_fetch, 'r') as zObject:
        zObject.extract(var_name, path=path_save)
    zObject.close()

# From the 2D array of (only) one variable, it selects a few columns, naming the headers.
# Returns the corresponding CSV file.
def simple_csv_maker(arr2d, file_name, var_name, indexes):  # Note: indexes are coming as a list of ints.
    fields = []
    for index in indexes:
        fields.append(var_name + '_' + str(index))
    new_arr2d = arr2d[:, indexes]
    # Create CSV file
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(new_arr2d)

def csv_maker(dict, file_name):
    fields = []
    indexes_first_arr2d = list(dict.values())[0]
    final_arr2d = file_to_array(list(dict)[0])[:, indexes_first_arr2d]
    for i in range(len(dict)):
        path = list(dict)[i]
        var_name = (path.split('/')[-1]).split('.')[0]
        indexes = list(dict.values())[i]
        for j in indexes:
            fields.append(var_name + '_' + str(j))
    for i in range(1, len(dict)):
        indexes_next_arr2d = list(dict.values())[i]
        next_arr2d = file_to_array(list(dict)[i])[:, indexes_next_arr2d]
        final_arr2d = np.hstack((final_arr2d, next_arr2d))
    # Create CSV file
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(final_arr2d)

#arr2d_lst is a list of 2D arrays with the variables to plot.
#lines_dist is the distance between primitives (or chunks).
def prim_plotter(arr2d_lst, lines_dist, *titles):
    fig, subs = plt.subplots(len(arr2d_lst), sharex=True)
    for i in range(len(arr2d_lst)):
        #Inside one 2D array
        for j in range(arr2d_lst[i].shape[1]):
            x_array = np.arange(arr2d_lst[i].shape[0])
            subs[i].plot(x_array, (arr2d_lst[i])[:,j])
        x_bars = list(np.arange(0, len(x_array), lines_dist))
        subs[i].vlines(x=x_bars, ymin=np.amin(arr2d_lst[i]), ymax=np.amax(arr2d_lst[i])+(0.1*np.amax(arr2d_lst[i])),
                       colors='black', label='vline_multiple - partial height')
        subs[i].set_title(titles[i])
    plt.show()





