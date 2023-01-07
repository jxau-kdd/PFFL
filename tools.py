'''
Description: 
Author: liyihui
email: 3187858832@qq.com
Date: 2023-01-06 22:29:31
LastEditTime: 2023-01-07 21:15:15
LastEditors: liyihui
'''
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csc import csc_matrix
import copy
import numpy as np

def init_Y_confidence(target):
    res = []
    p_target = copy.deepcopy(target)
    row, col = p_target.shape
    for i in range(row):
        sing_target = p_target[i, :]
        count = sing_target.sum()
        init_y = 1 / count
        sing_target[sing_target > 0] = init_y
        if sing_target.shape[0] == 1:
            x = sing_target.tolist()[0]
        else:
            x = sing_target.tolist()
        res.append(x)
    res = np.array(res)
    return res

def get_candidate(target):
    res = []
    p_target = copy.deepcopy(target)
    row, col = p_target.shape
    for i in range(row):
        p_target_single = p_target[i, :]
        if p_target_single.shape[0] == 1:
            sing_target = np.array(p_target_single[0])
            indexs = np.argwhere(sing_target == 1)
            indexs = [x[1] for x in indexs.tolist()]
            res.append(indexs)
        else:
            sing_target = np.array(p_target_single)
            indexs = np.argwhere(sing_target == 1).flatten().tolist()
            res.append(indexs)
    return res

def get_K_Neighbors(features, k):
    pdist_martrix = pdist(features, metric='euclidean')
    
    martrix = squareform(pdist_martrix)
    topK_neighbors = np.argsort(martrix, axis = 0)[1: k + 1].T
    return topK_neighbors

# 处理稀疏矩阵
def process_csc_matrix(labels, partial_target):
    if isinstance(labels, csc_matrix):
        # print('labels 稀疏矩阵')
        target = labels.todense().T
    else:
        target = labels.T

    if isinstance(partial_target, csc_matrix):
        # print('partial_target 稀疏矩阵')
        partial_target = partial_target.todense().T
    else:
        partial_target = partial_target.T
    return target, partial_target
