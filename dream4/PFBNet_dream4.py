import numpy as np
import xgboost as xgb
import operator
import re
import math
import time


def main(data_tm, sample_num, k, alpha, iter_num, data_ko):
    """
    Inferring gene regulatory networks (GRNs) from gene expression data using an integrative XGBoost-based method.

    Args:

        data_tm: Time-series experimental data.
        sample_num: Number of time-series experimental's samples.
        k: Previous k time points.
        alpha: Decay factor.
        iter_num: Number of iterations in XGBoost model.
        data_ko: Knockout experimental data.

    Returns:

        vim: A matrix recording the weights of regulatory network.

    """
    time_start = time.time()

    # Compute the accumulation of previous time points for time series data.
    x, y = time_accumu(data_tm, sample_num, k, alpha)

    # Compute the weights of gene regulatory network using XGBoost.
    feature_num = np.shape(data_tm)[1]
    vv = xgboost_weight(x, y, feature_num, iter_num)
    vv = np.transpose(vv)

    # Integrate knockout data
    vim = normalized_zscore(data_ko, 0)
    vim = vv * vim

    # Normalize inferred GRN matrix by row with L2-norm method.
    vim = normalized_l2norm(vim)

    # Use a statistical technology to refine the inferred GRN.
    statis = statistical_method(vim)
    vim = vim * statis

    # Compute the running time
    time_end = time.time()
    print('totally cost', time_end - time_start)

    return vim


def time_accumu(data, sample_num, k, alpha):
    """
    Compute the accumulation of previous time points for time series data.

    Args:

        data: Time-series experimental data.
        sample_num: Number of samples.
        k: previous k time points.
        alpha: decay factor.

    Returns:

        matrixx: Training data after accumulating several time points.
        matrixy: Label of the training data after accumulating several time points.

    """
    m = int(np.shape(data)[0] / sample_num)
    k = k + 1
    data1 = data
    matrixx = np.ones(((m - k + 1) * sample_num, np.shape(data)[1]))
    matrixy = np.ones(((m - k + 1) * sample_num, np.shape(data)[1]))

    for t in range(np.shape(data)[1]):
        data = data1[:, t]
        wwx = []
        wwy = []
        matx = np.ones(((m - k + 1) * sample_num, 1))
        maty = np.ones(((m - k + 1) * sample_num, 1))
        for j in range(sample_num):
            x = []
            w = []
            XX = 0
            for i in range(k):
                x.append(range(k - i - 1 + 21 * j, m - i + 21 * j))
                w.append(pow(alpha, i))
            w = w[0:k - 1]
            for i in range(1, k):
                X = w[i - 1] * data[x[i]]
                XX = XX + X
            Y = data[x[0]]
            wwx.append(XX)
            matx[(m - k + 1) * j:(m - k + 1) * j + (m - k + 1)] = wwx[j].reshape((m - k + 1), 1)
            wwy.append(Y)
            maty[(m - k + 1) * j:(m - k + 1) * j + (m - k + 1)] = wwy[j].reshape((m - k + 1), 1)
        matrixx[:, t] = matx.reshape((m - k + 1) * sample_num, )
        matrixy[:, t] = maty.reshape((m - k + 1) * sample_num, )

    return matrixx, matrixy


def xgboost_weight(x, y, subprob_num, iter_num):
    """
    Compute the weights of gene regulatory network using XGBoost.

    Args:

        x: training data.
        y: Label of the training data.
        subprob_num: Number of subproblems.
        iter_num: Number of iterations.

    Returns:

        vim: The matrix recording the weights of regulatory network after using XGBoost.

    """
    data = x
    d_size = data.shape
    vim = np.zeros((d_size[1], subprob_num)).tolist()  # vim: weights of Regulatory network

    for i in range(0, d_size[1]):
        print("----------------------------------------------------------------", i,
              "----------------------------------------------------------------")
        y1 = y[:, i].reshape(d_size[0], 1)
        if i == 0:
            x = data[:, 1:subprob_num]
        elif i < subprob_num:
            x = np.hstack((data[:, 0:i], data[:, i + 1:subprob_num]))
        else:
            x = data[:, 0:subprob_num]

        # Build model
        params = {
            'booster': 'gbtree',
            'gamma': 0.2,
            'max_depth': 4,
            'min_child_weight': 4,
            'alpha': 0,
            'lambda': 0,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'silent': 1,
            'eta': 0.0008
        }

        dtrain = xgb.DMatrix(x, y1)
        plst = params.items()
        model = xgb.train(plst, dtrain, iter_num)

        # Compute and sort feature importance
        importance = model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)

        # Convert the importance list to matrix weights
        for j in range(0, len(importance)):
            num = re.findall(r'\d+', importance[j][0])
            num = np.array(num)
            num = np.core.defchararray.strip(num, '()')
            num = int(num)
            if i >= subprob_num - 1:
                fea_num = num
            else:
                if num < i:
                    fea_num = num
                else:
                    fea_num = num + 1
            vim[i][fea_num] = importance[j][1]

    return vim


def normalized_zscore(x, ddof=0):
    # Normalize matrix by column with Z-score method.
    # axis: 1(无偏估计)，0(有偏估计)
    n = np.shape(x)[0]
    m = np.shape(x)[1]
    x_std = np.std(x, axis=0, ddof=ddof)
    beta = np.zeros((n, m))
    w = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            beta[i, j] = sum(x[:, j]) / n
            w[i, j] = abs((x[i, j] - beta[i, j]) / x_std[j])
    return w


def normalized_l2norm(x):
    # Normalize matrix by row with L2-norm method.
    n = np.shape(x)[0]
    y = np.power(x, 2)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_sum = sum(y[i, :])
            w[i, j] = y[i, j] / math.sqrt(x_sum)
    return w


def statistical_method(x):
    # Use a statistical technology to further refine the inferred GRN.
    n = np.shape(x)[0]
    vv = np.var(x, axis=1, ddof=1)
    w = np.zeros((n, n))
    for i in range(n):
        w[i, :] = vv[i]
    return w
