"""
Description: This script generates result table on EXP1
"""

import numpy as np
import pandas as pd
from utils.networkx_operations import *
from numpy import unravel_index

def load_EXP1_npz(filename):
    npzfile = np.load(filename)
    accuracy_array = npzfile["accuracy_array"]
    precision_array = npzfile["precision_array"]
    recall_array = npzfile["recall_array"]
    f1_array = npzfile["f1_array"]
    TP_array = npzfile["TP_array"]
    MCC_array = npzfile["MCC_array"]
    cost_array = npzfile["cost_array"]
    elapsed_time_array = npzfile["elapsed_time_array"]
    npzfile.close()
    return accuracy_array, precision_array, recall_array, f1_array, TP_array, MCC_array, cost_array, elapsed_time_array
    
def gen_table(i, idx_beta):
    df_MCA = pd.DataFrame(data=EXP1_MCA[i][:,idx_beta,:], index=gamma_list, columns=alpha_list)
    df_T_i1 = pd.DataFrame(data=EXP1_T_i1[i][:,idx_beta,:], index=gamma_list, columns=alpha_list)
    df_T_i2 = pd.DataFrame(data=EXP1_T_i2[i][:,idx_beta,:], index=gamma_list, columns=alpha_list)
    df_LP = pd.DataFrame(data=EXP1_LP[i][:,idx_beta,:], index=gamma_list, columns=alpha_list)
    return df_MCA, df_T_i1, df_T_i2, df_LP

def gen_result_table(result_idx_alpha0, result_idx_alpha_geq1, result_metric_alpha0, result_metric_alpha_geq1, df_metric_MCA, df_metric_T_i1, df_metric_T_i2, df_metric_LP, idx_beta):
    result_metric_alpha0[0, idx_beta] = np.max(df_metric_MCA.values[:,0])
    result_metric_alpha_geq1[0, idx_beta] = np.max(df_metric_MCA.values[:,1:])
    result_metric_alpha0[1, idx_beta] = np.max(df_metric_T_i1.values[:,0])
    result_metric_alpha_geq1[1, idx_beta] = np.max(df_metric_T_i1.values[:,1:])
    result_metric_alpha0[2, idx_beta] = np.max(df_metric_T_i2.values[:,0])
    result_metric_alpha_geq1[2, idx_beta] = np.max(df_metric_T_i2.values[:,1:])
    result_metric_alpha0[3, idx_beta] = np.max(df_metric_LP.values[:,0])
    result_metric_alpha_geq1[3, idx_beta] = np.max(df_metric_LP.values[:,1:])

    # save the index. NOTE: np.argmax on 2d array returns an index of flattened array.
    result_idx_alpha0[0, idx_beta] = np.argmax(df_metric_MCA.values[:,0])
    result_idx_alpha_geq1[0, idx_beta] = np.argmax(df_metric_MCA.values[:,1:])
    result_idx_alpha0[1, idx_beta] = np.argmax(df_metric_T_i1.values[:,0])
    result_idx_alpha_geq1[1, idx_beta] = np.argmax(df_metric_T_i1.values[:,1:])
    result_idx_alpha0[2, idx_beta] = np.argmax(df_metric_T_i2.values[:,0])
    result_idx_alpha_geq1[2, idx_beta] = np.argmax(df_metric_T_i2.values[:,1:])
    result_idx_alpha0[3, idx_beta] = np.argmax(df_metric_LP.values[:,0])
    result_idx_alpha_geq1[3, idx_beta] = np.argmax(df_metric_LP.values[:,1:])

def gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_metric_alpha0, result_metric_alpha_geq1, df_metric_MCA, df_metric_T_i1, df_metric_T_i2, df_metric_LP, idx_beta):
    # idx is 1 dimension here.
    idx = result_idx_alpha0[0, idx_beta]
    result_metric_alpha0[0, idx_beta] = df_metric_MCA.values[idx, 0]
    idx = result_idx_alpha0[1, idx_beta]
    result_metric_alpha0[1, idx_beta] = df_metric_T_i1.values[idx, 0]
    idx = result_idx_alpha0[2, idx_beta]
    result_metric_alpha0[2, idx_beta] = df_metric_T_i2.values[idx, 0]
    idx = result_idx_alpha0[3, idx_beta]
    result_metric_alpha0[3, idx_beta] = df_metric_LP.values[idx, 0]

    # idx is 2 dimension here.
    idx = unravel_index(result_idx_alpha_geq1[0, idx_beta], df_metric_MCA.values[:,1:].shape)
    result_metric_alpha_geq1[0, idx_beta] = df_metric_MCA.values[:,1:][idx]
    idx = unravel_index(result_idx_alpha_geq1[1, idx_beta], df_metric_T_i1.values[:,1:].shape)
    result_metric_alpha_geq1[1, idx_beta] = df_metric_T_i1.values[:,1:][idx]
    idx = unravel_index(result_idx_alpha_geq1[2, idx_beta], df_metric_T_i2.values[:,1:].shape)
    result_metric_alpha_geq1[2, idx_beta] = df_metric_T_i2.values[:,1:][idx]
    idx = unravel_index(result_idx_alpha_geq1[3, idx_beta], df_metric_LP.values[:,1:].shape)
    result_metric_alpha_geq1[3, idx_beta] = df_metric_LP.values[:,1:][idx]

def save_result_table(result_array, index, columns, filename):
    df_result = pd.DataFrame(data=result_array, index=index, columns=columns)
    df_result.to_csv("table/{}".format(filename), index=True)

def save_result_tables():
    save_result_table(result_accuracy_alpha0, result_index, result_columns, "EXP1_accuracy_alpha0_v3.csv")
    save_result_table(result_precision_alpha0, result_index, result_columns, "EXP1_precision_alpha0_v3.csv")
    save_result_table(result_recall_alpha0, result_index, result_columns, "EXP1_recall_alpha0_v3.csv")
    save_result_table(result_f1_alpha0, result_index, result_columns, "EXP1_f1_alpha0_v3.csv")
    save_result_table(result_TP_alpha0, result_index, result_columns, "EXP1_TP_alpha0_v3.csv")
    save_result_table(result_MCC_alpha0, result_index, result_columns, "EXP1_MCC_alpha0_v3.csv")
    save_result_table(result_cost_alpha0, result_index, result_columns, "EXP1_cost_alpha0_v3.csv")
    save_result_table(result_elapsed_time_alpha0, result_index, result_columns, "EXP1_elapsed_time_alpha0_v3.csv")

    save_result_table(result_accuracy_alpha_geq1, result_index, result_columns, "EXP1_accuracy_alpha_geq1_v3.csv")
    save_result_table(result_precision_alpha_geq1, result_index, result_columns, "EXP1_precision_alpha_geq1_v3.csv")
    save_result_table(result_recall_alpha_geq1, result_index, result_columns, "EXP1_recall_alpha_geq1_v3.csv")
    save_result_table(result_f1_alpha_geq1, result_index, result_columns, "EXP1_f1_alpha_geq1_v3.csv")
    save_result_table(result_TP_alpha_geq1, result_index, result_columns, "EXP1_TP_alpha_geq1_v3.csv")
    save_result_table(result_MCC_alpha_geq1, result_index, result_columns, "EXP1_MCC_alpha_geq1_v3.csv")
    save_result_table(result_cost_alpha_geq1, result_index, result_columns, "EXP1_cost_alpha_geq1_v3.csv")
    save_result_table(result_elapsed_time_alpha_geq1, result_index, result_columns, "EXP1_elapsed_time_alpha_geq1_v3.csv")

if __name__ == "__main__":

    gamma_list = [0, 16, 128]
    beta_list = [1, 2, 4]
    # alpha_list = [0.0] + [float(pow(2, x)) for x in range(9)]
    alpha_list = [0] + [pow(2, x) for x in range(9)]

    n_algo = 4
    result_index = ["MCA", "DST.i=1", "DST.i=2", "LP"]
    result_columns = beta_list
    # accuracy
    result_accuracy_alpha0 = np.zeros((n_algo, len(beta_list)))
    result_accuracy_alpha_geq1 = np.zeros((n_algo, len(beta_list)))
    # precision
    result_precision_alpha0 = np.zeros((n_algo, len(beta_list)))
    result_precision_alpha_geq1 = np.zeros((n_algo, len(beta_list)))
    # recall
    result_recall_alpha0 = np.zeros((n_algo, len(beta_list)))
    result_recall_alpha_geq1 = np.zeros((n_algo, len(beta_list)))
    # f1
    result_f1_alpha0 = np.zeros((n_algo, len(beta_list)))
    result_f1_alpha_geq1 = np.zeros((n_algo, len(beta_list)))
    # TP
    result_TP_alpha0 = np.zeros((n_algo, len(beta_list)))
    result_TP_alpha_geq1 = np.zeros((n_algo, len(beta_list)))
    # MCC
    result_MCC_alpha0 = np.zeros((n_algo, len(beta_list)))
    result_MCC_alpha_geq1 = np.zeros((n_algo, len(beta_list)))
    # cost
    result_cost_alpha0 = np.zeros((n_algo, len(beta_list)))
    result_cost_alpha_geq1 = np.zeros((n_algo, len(beta_list)))
    # elapsed_time
    result_elapsed_time_alpha0 = np.zeros((n_algo, len(beta_list)))
    result_elapsed_time_alpha_geq1 = np.zeros((n_algo, len(beta_list)))
    # idx
    result_idx_alpha0 = np.zeros((n_algo, len(beta_list))).astype(int)
    result_idx_alpha_geq1 = np.zeros((n_algo, len(beta_list))).astype(int)

    x_list = [1]
    for x in x_list:
        EXP1_MCA = load_EXP1_npz("npz/EXP1_MCA_x{}_v3.npz".format(int(x)))
        EXP1_T_i1 = load_EXP1_npz("npz/EXP1_T_i1_x{}_v3.npz".format(int(x)))
        EXP1_T_i2 = load_EXP1_npz("npz/EXP1_T_i2_x{}_v3.npz".format(int(x)))
        EXP1_LP = load_EXP1_npz("npz/EXP1_LP_x{}_v3.npz".format(int(x)))
        for idx_beta, beta in enumerate(beta_list):
            # When preparing the result table, get the index of the paramters alpha and gamma that gives the best MCC value.
            # From those, get the rest of the result statistics.
            df_MCC_MCA, df_MCC_T_i1, df_MCC_T_i2, df_MCC_LP = gen_table(5, idx_beta)
            gen_result_table(result_idx_alpha0, result_idx_alpha_geq1, result_MCC_alpha0, result_MCC_alpha_geq1, df_MCC_MCA, df_MCC_T_i1, df_MCC_T_i2, df_MCC_LP, idx_beta)
            # df_f1_MCA, df_f1_T_i1, df_f1_T_i2, df_f1_LP = gen_table(3, idx_beta)
            # gen_result_table(result_idx_alpha0, result_idx_alpha_geq1, result_f1_alpha0, result_f1_alpha_geq1, df_f1_MCA, df_f1_T_i1, df_f1_T_i2, df_f1_LP, idx_beta)

            # accuracy
            df_accuracy_MCA, df_accuracy_T_i1, df_accuracy_T_i2, df_accuracy_LP = gen_table(0, idx_beta)
            gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_accuracy_alpha0, result_accuracy_alpha_geq1, df_accuracy_MCA, df_accuracy_T_i1, df_accuracy_T_i2, df_accuracy_LP, idx_beta)
            # precision
            df_precision_MCA, df_precision_T_i1, df_precision_T_i2, df_precision_LP = gen_table(1, idx_beta)
            gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_precision_alpha0, result_precision_alpha_geq1, df_precision_MCA, df_precision_T_i1, df_precision_T_i2, df_precision_LP, idx_beta)
            # recall
            df_recall_MCA, df_recall_T_i1, df_recall_T_i2, df_recall_LP = gen_table(2, idx_beta)
            gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_recall_alpha0, result_recall_alpha_geq1, df_recall_MCA, df_recall_T_i1, df_recall_T_i2, df_recall_LP, idx_beta)
            # f1
            df_f1_MCA, df_f1_T_i1, df_f1_T_i2, df_f1_LP = gen_table(3, idx_beta)
            gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_f1_alpha0, result_f1_alpha_geq1, df_f1_MCA, df_f1_T_i1, df_f1_T_i2, df_f1_LP, idx_beta)
            # TP
            df_TP_MCA, df_TP_T_i1, df_TP_T_i2, df_TP_LP = gen_table(4, idx_beta)
            gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_TP_alpha0, result_TP_alpha_geq1, df_TP_MCA, df_TP_T_i1, df_TP_T_i2, df_TP_LP, idx_beta)
            # MCC
            # df_MCC_MCA, df_MCC_T_i1, df_MCC_T_i2, df_MCC_LP = gen_table(5, idx_beta)
            # gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_MCC_alpha0, result_MCC_alpha_geq1, df_MCC_MCA, df_MCC_T_i1, df_MCC_T_i2, df_MCC_LP, idx_beta)
            # cost
            df_cost_MCA, df_cost_T_i1, df_cost_T_i2, df_cost_LP = gen_table(6, idx_beta)
            gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_cost_alpha0, result_cost_alpha_geq1, df_cost_MCA, df_cost_T_i1, df_cost_T_i2, df_cost_LP, idx_beta)
            # gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_cost_alpha0, result_cost_alpha_geq1, df_cost_MCA, df_cost_T_i1, df_cost_T_i2, df_cost_LP, idx_beta)
            # elapsed_time
            df_elapsed_time_MCA, df_elapsed_time_T_i1, df_elapsed_time_T_i2, df_elapsed_time_LP = gen_table(7, idx_beta)
            gen_result_table_idx_given(result_idx_alpha0, result_idx_alpha_geq1, result_elapsed_time_alpha0, result_elapsed_time_alpha_geq1, df_elapsed_time_MCA, df_elapsed_time_T_i1, df_elapsed_time_T_i2, df_elapsed_time_LP, idx_beta)
            save_result_tables()

        #######################
        # Generate cost table and elapsed time table 

        cost_4algo_mean = [np.mean(EXP1_MCA[6]), np.mean(EXP1_T_i1[6]), np.mean(EXP1_T_i2[6]), np.mean(EXP1_LP[6])]
        cost_4algo_std = [np.std(EXP1_MCA[6]), np.std(EXP1_T_i1[6]), np.std(EXP1_T_i2[6]), np.std(EXP1_LP[6])]
        elapsed_time_4algo_mean = [np.mean(EXP1_MCA[7]), np.mean(EXP1_T_i1[7]), np.mean(EXP1_T_i2[7]), np.mean(EXP1_LP[7])]
        elapsed_time_4algo_std = [np.std(EXP1_MCA[7]), np.std(EXP1_T_i1[7]), np.std(EXP1_T_i2[7]), np.std(EXP1_LP[7])]

        df_cost_time_mean = pd.DataFrame(data={"Cost":cost_4algo_mean, "Time": elapsed_time_4algo_mean}, index=result_index)
        df_cost_time_std = pd.DataFrame(data={"Cost":cost_4algo_std, "Time": elapsed_time_4algo_std}, index=result_index)
        df_cost_time_summary = df_cost_time_mean.round(3).astype(str) + " ("+ df_cost_time_std.round(2).astype(str) + ")"
        df_cost_time_summary.to_csv("table/EXP1_cost_time_alpha_all.csv", index=True)

        # Generate cost table and elapsed time table 
        # Only for alpha = 0
        cost_4algo_mean_alpha0 = [np.mean(EXP1_MCA[6][:,:,0]), np.mean(EXP1_T_i1[6][:,:,0]), np.mean(EXP1_T_i2[6][:,:,0]), np.mean(EXP1_LP[6][:,:,0])]
        cost_4algo_std_alpha0 = [np.std(EXP1_MCA[6][:,:,0]), np.std(EXP1_T_i1[6][:,:,0]), np.std(EXP1_T_i2[6][:,:,0]), np.std(EXP1_LP[6][:,:,0])]
        elapsed_time_4algo_mean_alpha0 = [np.mean(EXP1_MCA[7][:,:,0]), np.mean(EXP1_T_i1[7][:,:,0]), np.mean(EXP1_T_i2[7][:,:,0]), np.mean(EXP1_LP[7][:,:,0])]
        elapsed_time_4algo_std_alpha0 = [np.std(EXP1_MCA[7][:,:,0]), np.std(EXP1_T_i1[7][:,:,0]), np.std(EXP1_T_i2[7][:,:,0]), np.std(EXP1_LP[7][:,:,0])]

        df_cost_time_mean_alpha0 = pd.DataFrame(data={"Cost":cost_4algo_mean_alpha0, "Time": elapsed_time_4algo_mean_alpha0}, index=result_index)
        df_cost_time_std_alpha0 = pd.DataFrame(data={"Cost":cost_4algo_std_alpha0, "Time": elapsed_time_4algo_std_alpha0}, index=result_index)
        df_cost_time_summary_alpha0 = df_cost_time_mean_alpha0.round(3).astype(str) + " ("+ df_cost_time_std_alpha0.round(2).astype(str) + ")"
        df_cost_time_summary_alpha0.to_csv("table/EXP1_cost_time_alpha0.csv", index=True)

