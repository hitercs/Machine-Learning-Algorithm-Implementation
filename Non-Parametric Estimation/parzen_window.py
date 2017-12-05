#-*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import codecs
read_buffer_size = 10000
write_buffer_size = 10000

class GussianKenerl(object):
    def __init__(self, window_len, x_i):
        self.width = window_len
        self.x_i = x_i
    def kenerl_func(self, x):
        # return 1/np.sqrt(2 * np.pi) * np.exp(-np.linalg.norm(x-self.x_i)/(2*self.width**2))
        return 1/(2 * np.pi * (self.width**2)) * np.exp(-np.linalg.norm(x-self.x_i)**2/(2*self.width**2))

class ParzenWindow(object):
    def __init__(self, data, ini_width):
        self.window_functions = []
        self.data = data
        self.ini_width = ini_width
        self.n = len(self.data)
        self.width = self.ini_width / np.sqrt(self.n)
        self.build_funcs()
    def build_funcs(self):
        for x in self.data:
            self.window_functions.append(GussianKenerl(self.width, x))
    def prob_estimate(self, x):
        return sum([f.kenerl_func(x) for f in self.window_functions]) / (self.n)

class ProbEstimator(object):
    def __init__(self, x_limit, y_limit, x_num, y_num, parzenW):
        self.x_points = np.linspace(x_limit[0], x_limit[1], x_num)
        self.y_points = np.linspace(y_limit[0], y_limit[1], y_num)
        self.parzenW = parzenW
    def estimate(self):
        self.prob = np.zeros((self.x_points.shape[0], self.y_points.shape[0]))
        self.X, self.Y = np.meshgrid(self.y_points, self.x_points)
        for i in range(self.x_points.size):
            for j in range(self.y_points.size):
                self.prob[i][j] = self.parzenW.prob_estimate(np.array([self.X[i][j], self.Y[i][j]]))
    def show(self, n, w):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(self.X, self.Y, self.prob, rstride=1, cstride=1, cmap='rainbow')
        plt.title("width: {}, sample num: {}".format(w, n))
        # plt.show()
        fig.savefig("./Doc/ParzenWindow/n{}-w{}.png".format(n, w), bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file", type=str, default=r"./Data/data1/train_class_1.txt")
    parser.add_argument("-num", "--num", type=int, default=100)
    args = parser.parse_args()
    data = []
    total = 0
    with codecs.open(args.file, encoding='utf-8', mode='r', buffering=read_buffer_size) as fp:
        for line in fp:
            words = line.strip().split('\t')
            data.append(np.array([float(words[0]), float(words[1])]))
            total += 1
            if total >= args.num:
                break
    data_num = [5, 10, 50, 100, 1000, 10000]
    init_width = [0.1, 0.5, 1, 5, 10]
    for n in data_num:
        for w in init_width:
            parzenW = ParzenWindow(data[0:n], w)
            estimator = ProbEstimator((-5, 15), (-7.5, 7.5), 100, 100, parzenW)
            estimator.estimate()
            estimator.show(n, w)