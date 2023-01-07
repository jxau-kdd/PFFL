import numpy as np
from sklearn.cluster import KMeans
from scipy.io import loadmat, savemat
from tools import *

class Ball:
    def __init__(self, features, labels, partial_target, sample_indexs) -> None:
        self.features = features
        self.labels = labels
        self.partial_target = partial_target
        self.sample_indexs = sample_indexs
        self.sample_size = features.shape[0]
        # self.avg_cosine = self.calc_acg_cosine()
        pass

    def get_Yconfidence(self, use_base_Y, k = 10):
        '''
            use_base_Y： 是否使用初始化的Yconfidence来进行标记增强
            k：KNN的邻居个数
        '''
        sample_size = self.features.shape[0]
        if k >= sample_size:
            k = sample_size - 1
        top_k_neigs = get_K_Neighbors(self.features, k)
        base_Yconfidence = init_Y_confidence(self.partial_target)
        # print(top_k_neigs.shape, k)
        candidate = get_candidate(self.partial_target)
        
        wr = [k - i for i in range(k)]
        Q = self.partial_target.shape[1]
        result = []
        for i in range(sample_size):
            k_neighbors = top_k_neigs[i, :]
            sumY = np.zeros(Q)

            percandidate = candidate[i]
            sizecandidate = len(percandidate)
            
            for t in range(sizecandidate):
                indexlabel = percandidate[t]
                for j in range(k):
                    indexneighbor = k_neighbors[j]
                    sumY[indexlabel] = sumY[indexlabel] + self.partial_target[indexneighbor, indexlabel] * wr[j]
            if sum(sumY) == 0:
                sumY = base_Yconfidence[i, :]
            else:
                sumY = sumY / sum(sumY)
                if use_base_Y:
                    sumY = 0.5 * base_Yconfidence[i, :] + 0.5 * sumY
            result.append(sumY.tolist())
        return np.array(result)

class BallList:
    def __init__(self, features, labels, partial_target, ball_split_len) -> None:
        self.ball_split_len = ball_split_len
        self.all_sample_size = features.shape[0]
        self.label_size = labels.shape[1]
        # self.labels = labels
        # self.partial_target = partial_target
        self.balls = self.build_ball(features, labels, partial_target, ball_split_len)
        self.ball_size = len(self.balls)
    
        # 分裂球
    def build_ball(self, features, labels, partial_target, ball_split_len):
        sample_indexs = [i for i in range(features.shape[0])]
        balls = self.cluster_predict(features, labels, partial_target, sample_indexs)
        count = 0
        final_balls = []
        while True:
            # print("round iter: {}".format(count))
            ball_len = len(balls)
            update_balls = []
            for ball in balls:
                if ball.sample_size <= 6:
                    final_balls.append(ball)
                    # print('tt: ', ball.sample_size)
                    continue
                else:
                    new_balls = self.cluster_predict(ball.features, ball.labels, ball.partial_target, ball.sample_indexs, ball_split_len)
                    if self.judge_split(new_balls):
                        for new_ball in new_balls:
                            update_balls.append(new_ball)
                    else:
                        final_balls.append(ball)
            new_ball_len = len(update_balls)
            if new_ball_len <= ball_len:
                for ball in update_balls:
                    final_balls.append(ball)
                break
            else:
                balls = update_balls
            count += 1
        return final_balls

    def build_ball_2(self, features, labels, partial_target, ball_split_len):
        # first_ball = Ball(features, labels, partial_target)
        balls = self.cluster_predict(features, labels, partial_target)
        count = 0
        final_balls = []
        while True:
            # print("round iter: {}".format(count))
            ball_len = len(balls)
            update_balls = []
            for ball in balls:
                if ball.sample_size <= 6:
                    final_balls.append(ball)
                    # print('tt: ', ball.sample_size)
                    continue
                else:
                    new_balls = self.cluster_predict(ball.features, ball.labels, ball.partial_target, ball_split_len)
                    for new_ball in new_balls:
                        if new_ball.sample_size > 3:
                            update_balls.append(new_ball)
                        else:
                            final_balls.append(new_ball)
            new_ball_len = len(update_balls)
            if new_ball_len <= ball_len:
                for ball in update_balls:
                    final_balls.append(ball)
                break
            else:
                balls = update_balls
            count += 1
        return final_balls

    # 单纯使用球内个数进行判断
    def judge_split(self, new_balls):
        for new_ball in new_balls:
            if new_ball.sample_size < 3:
                return False
        return True

    # 加上余弦相似度的判断（废弃）
    def judge_split_with_cosine(self, orig_ball, new_balls):
        # print(orig_ball.avg_cosine, new_balls[0].avg_cosine, new_balls[1].avg_cosine)
        for new_ball in new_balls:
            if new_ball.sample_size < 3 or new_ball.avg_cosine > orig_ball.avg_cosine:
                return False
        return True

    # cluster
    def cluster_predict(self, features, labels, partial_target, sample_indexs, n=2):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(features)
        predict = kmeans.predict(features)
        # can_division = judge(features, predict)
        balls = []
        for i in range(n):
            cluster_indexs = np.where(predict == i)
            clusrer_features = features[cluster_indexs]
            clusrer_labels = labels[cluster_indexs]
            clusrer_partial_target = partial_target[cluster_indexs]
            clusrer_sample_indexs = (np.array(sample_indexs)[cluster_indexs]).tolist()
            ball = Ball(clusrer_features, clusrer_labels, clusrer_partial_target, clusrer_sample_indexs)
            balls.append(ball)
        return balls
    
    def get_Yconfidence(self, use_base_Y=False):
        '''
            use_base_Y： 是否使用初始化的Yconfidence
        '''
        balls = self.balls
        result = np.zeros((self.all_sample_size, self.label_size))
        for ball in balls:
            ball_Y = ball.get_Yconfidence(use_base_Y)
            ball_idxs = ball.sample_indexs
            for i in range(len(ball_idxs)):
                idx = ball_idxs[i]
                result[idx, :] = ball_Y[i, :]
        return result

def demo():
    use_base_Y = True
    ball_split_len = 3

    dataset_path = 'demo.mat'
    datas = loadmat(dataset_path)
    features = datas['data']
    labels = datas['target']
    partial_target = datas['partial_target']

    labels, partial_target = process_csc_matrix(labels, partial_target)
    
    balls = BallList(features, labels, partial_target, ball_split_len)
    Yconfidence = balls.get_Yconfidence(use_base_Y)

    result = {
        'data': features,
        'target': labels,
        'partial_target': partial_target,
        'Yconfidence': Yconfidence,
    }
    savemat('demo_ld.mat', result)

    print(result)

if __name__ == "__main__":
    demo()
