#-*- encoding:utf-8 -*-
# This script is designed to generate 2-class 2-dim points with Gaussian Distribution
# Feature of different dimensions is independent
import numpy as np
import matplotlib.pyplot as plt
import codecs
import argparse
read_buffer_size = 10000
write_buffer_size = 10000
class Util:
    @staticmethod
    def draw_data(data1, data2):
        plt.plot(data1[:, 0], data1[:, 1], "x")
        plt.plot(data2[:, 0], data2[:, 1], "*")
        plt.axis('equal')
        plt.show()
class Dataset(object):
    def __init__(self, data_size, cov, mean):
        self.feature_num = mean.size
        self.data_size = data_size
        self.cov_mat = cov
        self.mean = mean
    def generate_data(self):
        self.data = np.random.multivariate_normal(self.mean, self.cov_mat, self.data_size)
    def draw(self):
        plt.plot(self.data[:, 0], self.data[:, 1], "x")
        plt.axis('equal')
        plt.show()
    def print_to_file(self, fn):
        with codecs.open(fn, encoding="utf-8", mode='w', buffering=write_buffer_size) as fp:
            for row in self.data:
                fp.write(u"{}\t{}\n".format(row[0], row[1]))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train-f1", "--train-f1", type=str, default=r"./Data/data1/train_class_1.txt")
    parser.add_argument("-train-f2", "--train-f2", type=str, default=r"./Data/data1/train_class_2.txt")
    parser.add_argument("-test-f1", "--test-f1", type=str, default=r"./Data/data1/test_class_1.txt")
    parser.add_argument("-test-f2", "--test-f2", type=str, default=r"./Data/data1/test_class_2.txt")
    args = parser.parse_args()
    setting1 = {"cov1": np.array([[1, 0], [0, 5]]), "mean1": np.array([0, 0]), "cov2": np.array([[5, 0], [0, 1]]), "mean2": np.array([4, 4])}
    # train class 1
    train_set_class_1 = Dataset(10000, setting1["cov1"], setting1["mean1"])
    train_set_class_1.generate_data()
    # train_set_class_1.draw()
    train_set_class_1.print_to_file(args.train_f1)
    # train class 2
    train_set_class_2 = Dataset(10000, setting1["cov2"], setting1["mean2"])
    train_set_class_2.generate_data()
    # train_set_class_2.draw()
    train_set_class_2.print_to_file(args.train_f2)
    Util.draw_data(train_set_class_1.data, train_set_class_2.data)
    # test class 1
    test_set_class_1 = Dataset(1000, setting1["cov1"], setting1["mean1"])
    test_set_class_1.generate_data()
    test_set_class_1.print_to_file(args.test_f1)
    # test class 2
    test_set_class_2 = Dataset(1000, setting1["cov2"], setting1["mean2"])
    test_set_class_2.generate_data()
    test_set_class_2.print_to_file(args.test_f2)
    Util.draw_data(test_set_class_1.data, test_set_class_2.data)
