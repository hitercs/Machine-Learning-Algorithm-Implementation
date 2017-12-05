#-*- encoding: utf-8 -*-
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import codecs
read_buffer_size = 10000
write_buffer_size = 10000

class KNearest(object):
    def __init__(self, data, k1):
        self.n = data.shape[0]
        self.kn = int(k1 * np.sqrt(self.n))
        self.data = data
        self.build_tree()
    def build_tree(self):
        self.nbrs = NearestNeighbors(n_neighbors=self.kn, algorithm='ball_tree').fit(self.data)
    def get_n_distance(self, x):
        dis, indices = self.nbrs.kneighbors([x])
        return dis[0][-1], indices[0][-1]
    def prob_estimate(self, x):
        return self.kn/(self.n * np.pi * (self.get_n_distance(x)[0])**2)

class KNearestClassify(object):
    def __init__(self, data, labels, k1):
        self.data = data
        self.labels = labels
        self.n = data.shape[0]
        self.kn = int(k1 * np.sqrt(self.n))
        self.build_tree()
    def build_tree(self):
        self.nbrs = NearestNeighbors(n_neighbors=self.kn, algorithm='ball_tree').fit(self.data)
    def get_n_nbrs(self, x):
        dis, indices = self.nbrs.kneighbors([x])
        return indices
    def prob_estimate(self, x):
        indices = self.get_n_nbrs(x)
        k_w1 = 0
        for i in indices[0]:
            if self.labels[i] == 1:
                k_w1 += 1
        return k_w1 / float(self.kn)

class KNearestProbEstimator(object):
    def __init__(self, x_limit, y_limit, x_num, y_num, kNearest):
        self.x_points = np.linspace(x_limit[0], x_limit[1], x_num)
        self.y_points = np.linspace(y_limit[0], y_limit[1], y_num)
        self.kNearest = kNearest
    def estimate(self):
        self.prob = np.zeros((self.x_points.shape[0], self.y_points.shape[0]))
        self.X, self.Y = np.meshgrid(self.y_points, self.x_points)
        for i in range(self.x_points.size):
            for j in range(self.y_points.size):
                self.prob[i][j] = self.kNearest.prob_estimate(np.array([self.X[i][j], self.Y[i][j]]))
    def show(self, n, k):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(self.X, self.Y, self.prob, rstride=1, cstride=1, cmap='rainbow')
        plt.title("k1: {}, sample num: {}".format(k, n))
        # plt.show()
        fig.savefig("./Doc/KNearest/n{}-k{}.png".format(n, k), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file", type=str, default=r"./Data/data1/train_class_1.txt")
    parser.add_argument("-num", "--num", type=int, default=10000)
    args = parser.parse_args()
    data = []
    data_num = [10, 50, 100]
    init_num = [1, 2, 3]
    total = 0
    with codecs.open(args.file, encoding='utf-8', mode='r', buffering=read_buffer_size) as fp:
        for line in fp:
            words = line.strip().split('\t')
            data.append(np.array([float(words[0]), float(words[1])]))
            total += 1
            if total >= args.num:
                break
    for n in data_num:
        for k in init_num:
            kNearest = KNearest(np.array(data)[0:n, :], k)
            estimator = KNearestProbEstimator((-5, 15), (-7.5, 7.5), 100, 100, kNearest)
            estimator.estimate()
            estimator.show(n, k)

