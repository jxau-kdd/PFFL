import numpy as np
import copy
import random
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform

# 数据标准化
def normalize(features):
    ConditionalLength = features.shape[1]
    for i in range(ConditionalLength):
        MaxValue = max(features[:, i])
        MinValue = min(features[:, i])
        for j in range(features.shape[0]):
            features[j, i] = (features[j, i] - MinValue) / (MaxValue - MinValue)
    return features

# 决策类编号
def get_number_label(ldl):
    sample_size, label_size = ldl.shape
    # print(sample_size, label_size)
    result = []
    for i in range(sample_size):
        tag_labels = np.zeros((label_size))
        ldl_label = ldl[i, :]
        idxs = np.where(ldl_label != 0)
        valid_tag = ldl_label[idxs].tolist()
        if len(valid_tag) == 0:
            pass
        else:
            valid_tag.sort(reverse=True)
            number = 1
            for tag in valid_tag:
                tag_idxs = np.where(ldl_label==tag)[0]
                for j in tag_idxs:
                    if tag_labels[j] == 0:
                        tag_labels[j] = number
                        number += 1
                        break
        result.append(tag_labels.tolist())
    return np.array(result)

# 返回整个实例集合下的邻域阈值
def GetThreshold(features, ParameterOmega):
    Temp = 0
    cond_feature_len = features.shape[1]
    for i in range(cond_feature_len):
        std_fea = np.std(features[:, i])
        T = float(std_fea / ParameterOmega)
        Temp += T
    Result = float((Temp / cond_feature_len) / ParameterOmega)
    return Result

# 获取所有实例下的邻域集合
def GetAllInstanceNeigborhoodList(features, ParameterOmega, indexs = 'ALL'):
    res = []

    if indexs != 'ALL':
        tmp_fearures = features[:, indexs]
    else:
        tmp_fearures = features

    sample_size = tmp_fearures.shape[0]

    Threshold = GetThreshold(tmp_fearures, ParameterOmega)
    Distances = squareform(pdist(features))
    neig_sorts = np.argsort(Distances)
    
    for i in range(sample_size):
        T = []
        # Vector_1 = tmp_fearures[i, :]
        neigs = neig_sorts[i, :]
        for neig in neigs:
            # Vector_2 = tmp_fearures[j, :]
            Distance = Distances[i, neig]
            if Distance <= Threshold:
                T.append(neig)
            else:
                break
        res.append(T)
    return res, Threshold

# 邻域等价类划分
def neig_equival_class(features, Omega):
    
    Neigborhood_f_Matrix, Threshold = GetAllInstanceNeigborhoodList(features, Omega)
    # print(Neigborhood_f_Matrix)
    return Neigborhood_f_Matrix

# 决策类划分
def decision_class(ldl):
    result = []
    tag_labels = get_number_label(ldl)
    label_num = ldl.shape[1]
    for i in range(label_num):
        label = tag_labels[:, i]
        idxs = np.where(label == 1)[0]
        if len(idxs) > 0:
            result.append(idxs.tolist())
    return result

# 计算下近似
def calc_Lower_approximation(equival_classes, dec_classes):
    sample_size = len(equival_classes)
    dec_size = len(dec_classes)
    count = 0
    for i in range(sample_size):
        equ_class = equival_classes[i]

        for j in range(dec_size):
            dec_class = dec_classes[j]
            if set(equ_class) <= set(dec_class):
                count += 1
    return count

# 计算依赖度
def calc_dep(features, dec_classes, omega):
    equival_classes = neig_equival_class(features, omega)
    low_appr = calc_Lower_approximation(equival_classes, dec_classes)
    yilai = low_appr / features.shape[0]

    return yilai

# 数据归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_K_Neighbors(features, k):
    pdist_martrix = pdist(features, metric='euclidean')
    
    martrix = squareform(pdist_martrix)
    topK_neighbors = np.argsort(martrix, axis = 0)[1: k + 1].T
    return topK_neighbors

# 计算same score
def calc_same_score_Neig(features, predict_Y, idx, same_k, Omega):
    select_features = [i for i in range(features.shape[1]) if i != idx]
    # select_features.append(idx)
    X = features[:, select_features]
    sample_size = X.shape[0]
    feature_len = X.shape[1]
    score_count = 0
    k = same_k
    neigs = get_K_Neighbors(X, k)
    # if feature_len < 20:
    #     neigs = get_K_Neighbors(X, k)
    # else:
    #     neigs, Threshold = GetAllInstanceNeigborhoodList(X, Omega)
    for i in range(sample_size):
        samples = neigs[i]
        # print(samples)
        base_target = predict_Y[i]
        count = 0
        for j in range(len(samples)):
            else_target = predict_Y[samples[j]]
            if base_target == else_target:
                count += 1
        score_count += count / k
    return score_count

# 根据标记置信度得到预测标记
def get_predict_Y(Y_confidence):
    res = []
    sample_size, label_size = Y_confidence.shape
    for i in range(sample_size):
        Y_conf = Y_confidence[i, :]
        max_Y = max(Y_conf)
        max_idx = np.where(Y_conf == max_Y)[0].tolist()
        if len(max_idx) == 1:
            res.append(max_idx[0])
        else:
            res.append(max_idx[random.randint(0, len(max_idx) - 1)])
    return res

# 计算特征的分数
def get_feature_scores(features, partial_labels, dec_classes, base, omega, same_k):

    dep_scores = []
    same_scores = []
    
    predict_Y = get_predict_Y(partial_labels)

    for idx in base:
        tmp_feature_idx = copy.deepcopy(base)
        tmp_feature_idx.remove(idx)

        important_score = calc_dep(features[:, tmp_feature_idx], dec_classes, omega)
        dep_scores.append(important_score)

        same_score = calc_same_score_Neig(features, predict_Y, idx, same_k, omega)
        same_scores.append(same_score)
        
        print('rate: {} %'.format(round((idx + 1) / len(base) * 100, 2), ))
    return np.array(dep_scores), np.array(same_scores)

# 得到特征排序
def get_feature_rank(features, partial_labels, omega, same_k=6, dep_weight=0.5):
    
    feature_len = features.shape[1]
    
    base = [i for i in range(feature_len)]

    dec_classes = decision_class(partial_labels)

    dep_scores, same_scores = get_feature_scores(features, partial_labels, dec_classes, base, omega, same_k)

    dep_scores = normalization(dep_scores)
    same_scores = normalization(same_scores)

    scores = (dep_weight * dep_scores) + ((1 - dep_weight) * same_scores)
    ranks = np.argsort(np.array(scores)) + 1

    return ranks

def demo():
    same_k = 5
    omega = 0.35
    dep_weight = 0.5

    dataset_path = 'demo_ld.mat'
    datas = sio.loadmat(dataset_path)
    features = datas['data']
    partial_labels = datas['Yconfidence']
    features = normalize(features)

    ranks = get_feature_rank(features, partial_labels, omega, same_k, dep_weight)
    print(ranks)

if __name__ == "__main__":
    demo()
