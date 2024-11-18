import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import distance
from scipy import cluster
from scipy.cluster.hierarchy import linkage,cut_tree
from dynamicTreeCut import cutreeHybrid
from PyWGCNA import WGCNA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter




def read_files(folder_path, encoding='gbk'):
    os.chdir(folder_path)
    file_list = os.listdir()
    samples = []
    sample_list = []
    for file_name in file_list:
        if os.path.isfile(file_name):
            try:
                x_data = pd.read_table(file_name, index_col=None, header=0, sep="\t", encoding=encoding)
                samples.append(len(x_data.columns) - 1)
                sample_list.append(x_data)
            except UnicodeDecodeError as e:
                print(f"Error decoding file {file_name}: {e}")

    return samples, sample_list



def setdiff(x, y):
    u = list(x)
    v = list(y)
    return [item for item in u if item not in v]



def getnetworks(case_dir, control_dir, net_case_dir, net_control_dir):
    print("importing datasets...")

    union_species1 = set()
    for case_data in case_dir:
        union_species1 |= set(case_data.iloc[:, 0])

    union_species2 = set()
    for control_data in control_dir:
        union_species2 |= set(control_data.iloc[:, 0])

    union_species = union_species1.union(union_species2)
    union_matrix = np.zeros((len(union_species), len(union_species)))
    np.fill_diagonal(union_matrix, 1)
    union_matrix = pd.DataFrame(union_matrix, index=union_species, columns=union_species)

    # case sample
    case_all_sample = pd.DataFrame()
    for k in range(len(case_dir)):
        case1 = pd.melt(case_dir[k], id_vars='species')
        case_all_sample = case_all_sample.append(case1)
    case_com_sample = case_all_sample.pivot_table(index="species", columns="variable", values="value", aggfunc=np.mean,
                                                  fill_value=0)

    # control sample
    control_all_sample = pd.DataFrame()
    for k in range(len(control_dir)):
        control1 = pd.melt(control_dir[k], id_vars='species')
        control_all_sample = control_all_sample.append(control1)
    control_com_sample = control_all_sample.pivot_table(index="species", columns="variable", values="value", aggfunc=np.mean,
                                                        fill_value=0)

    case_tax = np.setdiff1d(case_com_sample.index, control_com_sample.index)
    if len(case_tax) != 0:
        a = np.zeros((len(case_tax), control_com_sample.shape[1]),dtype='object')
        b = case_tax.astype('object')
        a = np.insert(a, 0, b, axis=1)
        a = pd.DataFrame(a)
        a = a.set_index(0)
        a = a.rename(columns=dict(zip(a.columns, control_com_sample.columns)))
        case_com_sample_new = pd.concat([control_com_sample, a], axis=0)
    else:
        case_com_sample_new = case_com_sample

    control_tax = np.setdiff1d(control_com_sample.index, case_com_sample.index)

    if len(control_tax) != 0:
        a = np.zeros((len(control_tax), case_com_sample.shape[1]), dtype='object')
        b = control_tax.astype('object')
        a = np.insert(a, 0, b, axis=1)
        a = pd.DataFrame(a)
        a = a.set_index(0)
        a = a.rename(columns=dict(zip(a.columns, case_com_sample.columns)))
        control_com_sample_new = pd.concat([case_com_sample, a], axis=0)
    else:
        control_com_sample_new = control_com_sample


    print("constructing networks...")

    case_union_data_list = []
    for i in range(len(case_dir)):
        fi = net_case_dir[i]
        fi.set_index(fi.iloc[:, 0], inplace=True)
        fi.drop(fi.columns[0], axis=1, inplace=True)
        matrix = np.zeros((len(union_species), len(union_species)))
        np.fill_diagonal(matrix, 1)
        matrix = pd.DataFrame(matrix, index=union_species, columns=union_species)
        re_union = matrix
        # diff_species = setdiff(union_species, fi.index)
        for j in range(fi.shape[0]):
            row_name = fi.index[j]
            re_union.loc[row_name, :] = fi.loc[row_name, :].astype(float)
            re_union = re_union.fillna(0)
        re_union = re_union.loc[union_species, union_species]
        case_union_data_list.append(re_union)

    # case_union_data_list = pd.DataFrame(case_union_data_list)

    control_union_data_list = []
    for i in range(len(control_dir)):
        fi = net_control_dir[i]
        fi.set_index(fi.iloc[:, 0], inplace=True)
        fi.drop(fi.columns[0], axis=1, inplace=True)
        matrix = np.zeros((len(union_species), len(union_species)))
        np.fill_diagonal(matrix, 1)
        matrix = pd.DataFrame(matrix, index=union_species, columns=union_species)
        re_union = matrix
        # diff_species = setdiff(union_species, fi.index)
        for j in range(fi.shape[0]):
            row_name = fi.index[j]
            re_union.loc[row_name, :] = fi.loc[row_name, :].astype(float)
            re_union = re_union.fillna(0)
        re_union = re_union.loc[union_species, union_species]
        control_union_data_list.append(re_union)
############################1.5function
    ########################################## 3. combination ################################################

    print("integrating networks...")
    matrix1 = np.zeros((len(union_species), len(union_species)))
    np.fill_diagonal(matrix1, 1)
    matrix1 = pd.DataFrame(matrix1, index=union_species, columns=union_species)
    pool_union1 = matrix1
    for i in range(1, (len(pool_union1) - 1)):
        for j in range((i + 1), len(pool_union1)):
            son = 0
            mom = 0
            for k in range(len(case_dir)):
                m = case_samples[k]
                n = case_union_data_list[k]
                v = (1 - n.iloc[i, j] ** 2) / (m - 1)
                w = 1 / v
                son += w * n.iloc[i, j]
                mom += w
            pool_union1.iloc[i, j] = son / mom
            pool_union1.iloc[j, i] = pool_union1.iloc[i, j]
    case_union = pool_union1
    matrix2 = np.zeros((len(union_species), len(union_species)))
    np.fill_diagonal(matrix2, 1)
    matrix2 = pd.DataFrame(matrix2, index=union_species, columns=union_species)
    pool_union2 = matrix2
    for i in range(1, (len(pool_union2) - 1)):
        for j in range((i + 1), len(pool_union2)):
            son = 0
            mom = 0
            for k in range(len(control_dir)):
                m = control_samples[k]
                n = control_union_data_list[k]
                v = (1 - n.iloc[i, j] ** 2) / (m - 1)
                w = 1 / v
                son += w * n.iloc[i, j]
                mom += w
            pool_union2.iloc[i, j] = son / mom
            pool_union2.iloc[j, i] = pool_union2.iloc[i, j]
    control_union = pool_union2
    return control_union,case_union, control_com_sample_new,case_com_sample_new


def pow_adjacency(mat, pow):
    mat_soft = mat.values.copy()
    for i in range(mat_soft.shape[0]):
        for j in range(mat_soft.shape[1]):
            mat_soft[i, j] = mat_soft[i, j] ** pow
    return pd.DataFrame(mat_soft, index=mat.index, columns=mat.columns)



if __name__ == '__main__':
    outpath = '../../result/'
    case_samples, case_dir = read_files("../data/case_dir")
    control_samples, control_dir = read_files("../control_dir")
    net_case_samples, net_case_dir = read_files("../net_case_dir_mic")
    net_control_samples, net_control_dir = read_files("../net_control_dir_mic")
    case_union, control_union, case_com_sample_new, \
    control_com_sample_new = getnetworks(case_dir, control_dir,net_case_dir, net_control_dir)
    case_union.to_csv(outpath + 'case_union_m.csv', index=True)
    control_union.to_csv(outpath + 'control_union_m.csv', index=True)
    case_com_sample_new.to_csv(outpath + 'case_sample.csv', index=True)
    control_com_sample_new.to_csv(outpath + 'control_sample.csv', index=True)
    alldata = pd.concat([case_com_sample_new, control_com_sample_new], axis=1)
    # 保存合并后的 DataFrame 到 CSV 文件
    alldata.to_csv(outpath + 'alldata.csv', index=True)

