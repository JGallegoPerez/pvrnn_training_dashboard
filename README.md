# pvrnn_training_dashboard
Dashboard using Dash and Plotly to visualize PVRNN training results.

This repository showcases a Dashboard that I run locally to help me inspect the training results of a PVRNN network. The files that I use are very specific to the implementation that trained the network and 
how it saves the files (the implementation corresponds to PVRNN's master branch). 

For those unfamiliar with PVRNN, see Reza's paper:

https://groups.oist.jp/sites/default/files/imce/u103429/Reza-NC-final%20accepted%206-25-2019.pdf

The dashboard allows me to visualize all the variables of interest after training a dataset under different *w* combinations, as well as to inspect the most relevant weight connections. 
It features a dropdown menu to toggle between the *w* combinations, and a range slider to shorten or extend the timesteps range. 

See one example below:

![dash1](https://github.com/JGallegoPerez/pvrnn_training_dashboard/assets/89183135/67ff5d31-bb5c-4880-a3ce-ad85645f6646)

One more example, with different *w* settings and a shorter timestep range:

![dash2](https://github.com/JGallegoPerez/pvrnn_training_dashboard/assets/89183135/466150b4-b32a-4c5d-9ab9-c083dbc9d6a4)

## How to use

Clone the repository, or save all the files in the same directory. 

Install the dependencies from the *requirements.txt* file.

We execute the program from the command line. The following command options must be entered:
- [-t] -> The title to give to the dashboard. Between quotes.
- [-d] -> The path of the directory where the results subdirectories are located.
- [-r] -> The results subdirectories (each folder represented as a cluster job number). Separated by spaces.

Example (applicable with the files included in this repository):

*python dashboard.py -t "PVRNN training results with different W values" -d "/home/jorge/Desktop/Code/pvrnn_training_dashboard/training_results" -r 0001_28513260 0002_28513261 0003_28513262 0004_28513263 0005_28513264 0006_28513265 0007_28513266*


***Complying with policy, the data in this repository were not leading to any scientific publication and are employed merely for demonstration purposes.*** 



