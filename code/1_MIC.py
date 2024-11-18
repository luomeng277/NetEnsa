import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from minepy import MINE
from scipy.stats import pearsonr
from sklearn.preprocessing import scale


def MIC_matirx(dataframe, mine):
    data = np.array(dataframe)
    n = len(data[0, :])
    result = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            mine.compute_score(data[:, i], data[:, j])
            result[i, j] = mine.mic()
            result[j, i] = mine.mic()
    RT = pd.DataFrame(result)
    return RT


def calculate_mic_for_files(folder_path,output_folder):

    file_list = os.listdir(folder_path)
    # output_folder = os.listdir(output_folder)
    for file_name in file_list:
        if os.path.isfile(os.path.join(folder_path, file_name)):
    
            file_path = os.path.join(folder_path, file_name)
            x_data = pd.read_table(file_path, index_col=None, header=0, sep="\t")


            index_values = x_data.iloc[:, 0]
            index_values = np.array(index_values)
            index_values = index_values[:, np.newaxis]

            data_values = x_data.iloc[:, 1:].values
            data = data_values.T


            n = data.shape[1]
            result = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    mine = MINE(alpha=0.6, c=15)
                    mine.compute_score(data[:, i], data[:, j])
                    mic_value = mine.mic()
                    result[i, j] = mic_value
                    result[j, i] = mic_value
                result[i][i] = 1


            sym_matrix = np.hstack((index_values, result))
            # sym_matrix = np.concatenate((index_values, result), axis=0)
            temp = (np.zeros((1,), dtype=int))[:, np.newaxis]
            temp1 = np.concatenate((temp, index_values),axis = 0).T
            net_dir = np.concatenate((temp1, sym_matrix),axis = 0)

        net_dir = pd.DataFrame(net_dir)


        file_path = os.path.join(output_folder, file_name )
        net_dir.to_csv(file_path, sep='\t', index=False,header= None)
        # net_dir.to_csv(output_folder + file_name+'.txt', sep='\t', index=True)



if __name__ == '__main__':
    path = '../data/'
    case_dir = path + "/case_dir/"  # case_dir
    net_case_dir = path + "/net_case_dir_mic"
    if not os.path.exists(net_case_dir):
        os.makedirs(net_case_dir)
    calculate_mic_for_files(case_dir, net_case_dir)

    control_dir = path +"/control_dir"  # control_dir
    net_control_dir = path +"/net_control_dir_mic"
    if not os.path.exists(net_control_dir):
        os.makedirs(net_control_dir)
    calculate_mic_for_files(control_dir,net_control_dir)






