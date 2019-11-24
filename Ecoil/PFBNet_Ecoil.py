import pandas as pd
from sklearn import preprocessing
import numpy as np
import time
import xgboost as xgb
import operator
import re
import math

def ecoil_main(data, sample_num, k, alpha, feature_num, iter_num):
    time_start = time.time()

    data1 = np.hstack((data[0,:,:],data[1,:,:],data[2,:,:]))
    data2 = np.zeros((3,1484,int((np.shape(data1)[1])/3)))

    for i in range(3):
        data2[i,:,:] = data1[:,i*sample_num:(i+1)*sample_num]

    # Compute the accumulation of previous time points for time series data.
    data_tm  = time_series(data2, k, alpha)

    # Compute the weights of gene regulatory network using XGBoost.
    vv = xgboost_weight(data_tm, feature_num, iter_num)
    vim = np.transpose(vv)
    print('shape of weights after xgboost is : ',np.shape(vv))

    ## Use a statistical technology to refine the inferred GRN.
    statis = statistical_method(vim)
    vim = vim * statis

    # Compute the running time
    time_end = time.time()
    print('totally cost', time_end - time_start)

    return vim

def time_series(data, k, alpha):
    """
    Compute the accumulation of previous time points for time series data.

    Args:

        data: Time-series experimental data(3D), shape (d*m*n). d: number of 2D matrix, m: number of timeseries_data, n: number of time points
        k: previous k time points.
        alpha: decay factor.

    Returns:

        kdata: matrix after computing previous time points(2D), whose shape is ( (d*(n-k))*m )

    """
    data_3d = np.shape(data)[0]
    data_raw = np.shape(data)[1]
    data_column = np.shape(data)[2]-k
    kdata  = np.zeros((data_3d,data_raw,data_column))
    for d in range(data_3d):
        for t in range(data_raw):
            for i in range(data_column):
                dd = 0
                j = 0
                while j <= k:
                    dd += pow(alpha,k-j)*data[d,t,j+i]
                    j += 1
                kdata[d,t,i] = dd

    kdata1 = np.transpose(np.hstack((kdata[0,:,:],kdata[1,:,:],kdata[2,:,:])))
    print('shape of kdata1 is : ', np.shape(kdata1))

    return kdata1


def xgboost_weight(data, feature_num,  iter_num):
    """
    Compute the weights of gene regulatory network using XGBoost.

    Args:

        data: Experimental data.
        subprob_num: Number of subproblems.
        feature_num: Number of regulator genes.
        iter_num: Number of iterations.

    Returns:

        vim: The matrix recording the weights of regulatory network after using XGBoost.

    """

    vim = np.zeros((data.shape[1], feature_num)).tolist()  # vim: weights of Regulatory network
    for i in range(0, data.shape[1]):
        print("----------------------------------------------------------------", i,
              "----------------------------------------------------------------")

        # split train and test data set
        y = data[:, i]
        #print('the value of y is : ', y)
        if i == 0:
            x = data[:, 1:feature_num]
        elif i < feature_num:
            x = np.hstack((data[:, 0:i], data[:, i + 1:feature_num]))
        else:
            x = data[:, 0:feature_num]

        print('shape of x is : ', np.shape(x))

        # Build model
        params = {

            'booster': 'gbtree',
            'max_depth': 4,
            'min_child_weight':4 ,
            'lambda': 0,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'silent': 1,
            'eta': 0.0008
        }

        dtrain = xgb.DMatrix(x, y)
        plst = params.items()
        model = xgb.train(plst, dtrain, iter_num)

        # Compute and sort feature importance
        importance = model.get_fscore()
        #importance = model.get_score(fmap='', importance_type='total_gain')
        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
        print('size of importance is : ', np.shape(importance))

        # Convert the importance list to matrix weights
        for j in range(0, len(importance)):
            num = re.findall(r'\d+', importance[j][0])
            num = np.array(num)
            num = np.core.defchararray.strip(num, '()')
            num = int(num)
            if i >= feature_num - 1:
                fea_num = num
            else:
                if num < i:
                    fea_num = num
                else:
                    fea_num = num + 1
            vim[i][fea_num] = importance[j][1]

    return vim


def statistical_method(x):
    # Use a statistical technology to further refine the inferred GRN.
    n = np.shape(x)[0]
    m = np.shape(x)[1]
    vv = np.var(x, axis=1, ddof=1)
    w = np.zeros((n, m))
    for i in range(n):
        w[i, :] = vv[i]
    return w
