#-*- encoding: utf-8 -*-
import numpy as np
import codecs
import settings
import os
import random
import argparse
import datetime
from scipy.stats import multivariate_normal

np.random.seed(settings.seed)
random.seed(settings.seed)
class Util:
    @staticmethod
    def softmax(vec):
        return np.exp(vec)/np.sum(np.exp(vec))
    @staticmethod
    def hsoftmax(mat):
        return np.array([Util.softmax(v) for v in mat])
    @staticmethod
    def sigmoid(scalar):
        return 1/(1+np.exp(-scalar))
    @staticmethod
    def sigmoid_vec(vec):
        return np.array([Util.sigmoid(x) for x in vec])
    @staticmethod
    def sigmoid_mat(mat):
        return np.array([Util.sigmoid_vec(v) for v in mat])
    @staticmethod
    def tanh(mat):
        return (np.exp(mat) - np.exp(-mat))/(np.exp(mat) + np.exp(-mat))
    @staticmethod
    def tanh_derivative(mat):
        return 1 - Util.tanh(mat) ** 2
    @staticmethod
    def softmax_derivative(vec, pos):
        Y = Util.softmax(vec)
        d = []
        for i in range(Y.shape[0]):
            if i == pos:
                d.append(Y[i] * (1 - Y[i]))
            else:
                d.append(- Y[pos] * Y[i])
        return np.array(d)
        # return np.array([y * (1 - y) for y in Util.softmax(vec)])
    @staticmethod
    def softmax_derivative_mat(mat, labels):
        return np.array([Util.softmax_derivative(mat[i, :], labels[i]) for i in range(mat.shape[0])])
    @staticmethod
    def sigmoid_derivative(scalar):
        y = Util.sigmoid(scalar)
        return y * (1 - y)
    @staticmethod
    def sigmoid_derivative_vec(vec):
        Y = Util.sigmoid_vec(vec)
        return Y * (1 - Y)
    @staticmethod
    def sigmoid_derivative_mat(mat):
        Y = Util.sigmoid_mat(mat)
        return Y * (1 - Y)

class MultiClassDataset(object):
    def __init__(self, in_dir):
        self.data = []
        self.labels = []
        self.load_data(os.path.join(in_dir, "train_class_1.txt"), np.array([1, 0, 0, 0]))
        self.load_data(os.path.join(in_dir, "train_class_2.txt"), np.array([0, 1, 0, 0]))
        self.load_data(os.path.join(in_dir, "train_class_3.txt"), np.array([0, 0, 1, 0]))
        self.load_data(os.path.join(in_dir, "train_class_4.txt"), np.array([0, 0, 0, 1]))
        self.shuffle()
        self.test_data = []
        self.test_labels = []
        self.load_test_data(os.path.join(in_dir, "test_class_1.txt"), np.array([1, 0, 0, 0]))
        self.load_test_data(os.path.join(in_dir, "test_class_2.txt"), np.array([0, 1, 0, 0]))
        self.load_test_data(os.path.join(in_dir, "test_class_3.txt"), np.array([0, 0, 1, 0]))
        self.load_test_data(os.path.join(in_dir, "test_class_4.txt"), np.array([0, 0, 0, 1]))

    def load_data(self, in_fn, label):
        with codecs.open(in_fn, encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                words = line.strip().split()
                self.data.append(np.array([float(x) for x in words]))
                self.labels.append(label)

    def load_test_data(self, in_fn, label):
        with codecs.open(in_fn, encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                words = line.strip().split()
                self.test_data.append(np.array([float(x) for x in words]))
                self.test_labels.append(label)

    def shuffle(self):
        data_labels = zip(self.data, self.labels)
        random.shuffle(data_labels)
        self.data, self.labels = list(zip(*data_labels)[0]), list(zip(*data_labels)[1])
class BatchMLP(object):
    def __init__(self, d, nh, c):
        self.d = d
        self.nh = nh
        self.c = c
        self.input_to_hidden_W = np.zeros((d, nh))
        self.hidden_to_output_W = np.zeros((nh, c))
        self.input_to_hidden_b = np.zeros(nh)
        self.hidden_to_output_b = np.zeros(c)
        self.init_params()
    def forward(self, X):
        # X: (batch_size * d)
        # self.Z: (batch_size * c)
        self.Net_j = X.dot(self.input_to_hidden_W) + self.input_to_hidden_b
        self.Y = Util.sigmoid_mat(self.Net_j)
        # self.Y =Util.tanh(self.Net_j)
        self.Net_k = self.Y.dot(self.hidden_to_output_W) + self.hidden_to_output_b
        self.Z = Util.hsoftmax(self.Net_k)
        return self.Z
    def classify(self, X):
        Z = self.forward(X)
        return np.argmax(Z, 1), Z
    def init_params(self):
        self.input_to_hidden_W = np.random.normal(0, settings.init_variance, self.nh * self.d).reshape(self.d, self.nh)
        self.hidden_to_output_W = np.random.normal(0, settings.init_variance, self.c * self.nh).reshape(self.nh, self.c)
        self.input_to_hidden_b = np.random.normal(0, settings.init_variance, self.nh)
        self.hidden_to_output_b = np.random.normal(0, settings.init_variance, self.c)

class Criteria(object):
    def __init__(self, mlp, model_dir):
        self.mlp = mlp
        self.model_dir = os.path.join(model_dir, datetime.datetime.today().strftime('%d-%b-%Y-%H-%M-%S'))
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
    def cross_entropy_loss(self, Z, T):
        # Z : batch_size * c
        # T : batch_size * c
        return -np.sum(T * np.log(Z))
    def cross_entropy_backward(self, X, T):
        # X : batch_size * d
        # T : batch_size * c
        Z = self.mlp.forward(X)
        e_signal_k = - (1.0/np.sum(T * Z, 1)).reshape(T.shape[0], 1) * Util.softmax_derivative_mat(self.mlp.Net_k, np.argmax(T, 1))
        hidden_to_output_W_delta = self.mlp.Y.transpose().dot(e_signal_k)
        hidden_to_output_b_delta = np.sum(e_signal_k, 0)
        e_signal_j = e_signal_k.dot(self.mlp.hidden_to_output_W.transpose()) * Util.sigmoid_derivative_mat(self.mlp.Net_j)
        # e_signal_j = e_signal_k.dot(self.mlp.hidden_to_output_W.transpose()) * Util.tanh_derivative(self.mlp.Net_j)
        input_to_hidden_W_delta = e_signal_j.transpose().dot(X)
        input_to_hidden_b_delta = np.sum(e_signal_j, 0)
        return (hidden_to_output_W_delta, hidden_to_output_b_delta,
                input_to_hidden_W_delta, input_to_hidden_b_delta)
    def cross_entropy_training(self, X_set, T, testX_set, test_T, batch_size, max_epochs, lr):
        with codecs.open(os.path.join(self.model_dir, "log.txt"), encoding='utf-8', mode='a', buffering=settings.write_buffer_size) as log_fp:
            self.save_model(0)
            log_fp.write("batch_size = {}, lr = {}, max_epochs = {}\n".format(batch_size, lr, max_epochs))
            # best_acc = 0.0
            # bad_case = 0
            for i in range(max_epochs):
                print "starting epochs {}".format(i)
                for j in range(len(X_set)/batch_size):
                    x_data = np.array(X_set[j*batch_size:(j+1)*batch_size])
                    t_labels = np.array(T[j*batch_size:(j+1)*batch_size])
                    hidden_to_output_W_delta, hidden_to_output_b_delta, input_to_hidden_W_delta, input_to_hidden_b_delta = self.cross_entropy_backward(x_data, t_labels)
                    batch_loss = self.cross_entropy_loss(self.mlp.Z, t_labels)
                    self.mlp.hidden_to_output_W += -lr * hidden_to_output_W_delta
                    self.mlp.hidden_to_output_b += -lr * hidden_to_output_b_delta
                    self.mlp.input_to_hidden_W += -lr * input_to_hidden_W_delta
                    self.mlp.input_to_hidden_b += -lr * input_to_hidden_b_delta
                    # print "epoch id = {}, batch id = {}, loss = {}".format(i, j, batch_loss)
                self.save_model(i+1)
                acc = self.testing(testX_set, test_T)
                # if acc > best_acc:
                #     best_acc = acc
                # else:
                #     bad_case += 1
                #     if bad_case % 5 == 0:
                #         lr *= 0.50
                print "{} epoch, lr = {}, accuracy = {}".format(i, lr, acc)
                log_fp.write("{} epoch, lr = {}, accuracy = {}\n".format(i, lr, acc))
                if acc > 0.437:
                    break
    def save_model(self, epoch_num):
        # save parameters to file
        # save input_to_hidden_W
        with codecs.open(os.path.join(self.model_dir, "model_epoch_{}.model").format(epoch_num), mode='w', buffering=settings.write_buffer_size) as model_fp:
            model_fp.write("input_to_hidden_W\n")
            model_fp.write("{}\n".format(self.mlp.input_to_hidden_W))
            model_fp.write("input_to_hidden_b\n")
            model_fp.write("{}\n".format(self.mlp.input_to_hidden_b))
            model_fp.write("hidden_to_output_W\n")
            model_fp.write("{}\n".format(self.mlp.hidden_to_output_W))
            model_fp.write("hidden_to_output_b\n")
            model_fp.write("{}\n".format(self.mlp.hidden_to_output_b))
    def testing(self, X_set, T):
        total = 0
        correct_num = 0
        for i in range(len(X_set)):
            total += 1
            x = X_set[i]
            labels = T[i]
            predict_label, prob = self.mlp.classify(x.reshape(1, x.shape[0]))
            # print predict_label
            # print prob
            if labels[predict_label[0]] == 1:
                correct_num += 1
        return correct_num/float(total)
class GoldenClassifier:
    @staticmethod
    def classify(x):
        Py_0 = multivariate_normal.pdf(x, mean=settings.data_gen_parameters[0]["mean"], cov=settings.data_gen_parameters[0]["cov"])
        Py_1 = multivariate_normal.pdf(x, mean=settings.data_gen_parameters[1]["mean"], cov=settings.data_gen_parameters[1]["cov"])
        Py_2 = multivariate_normal.pdf(x, mean=settings.data_gen_parameters[2]["mean"], cov=settings.data_gen_parameters[2]["cov"])
        Py_3 = multivariate_normal.pdf(x, mean=settings.data_gen_parameters[3]["mean"], cov=settings.data_gen_parameters[3]["cov"])
        Px = Py_0 + Py_1 + Py_2 + Py_3
        return np.array([Py_0/Px, Py_1/Px, Py_2/Px, Py_3/Px])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in-dir", "--in-dir", type=str, default=r"../Data/Experiment")
    parser.add_argument("-model-dir", "--model-dir", type=str, default=r"../Model/cross_entropy/Experiment")
    args = parser.parse_args()
    print "start loading dataset..."
    dataset = MultiClassDataset(args.in_dir)
    # lr = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    # for l in lr:
    #     mlp = BatchMLP(3, 3, 4)
    #     criteria = Criteria(mlp, args.model_dir)
    #     criteria.cross_entropy_training(dataset.data, dataset.labels, dataset.test_data,
    #                                     dataset.test_labels, 16, 100, l)
    print "start building MLP model..."
    mlp = BatchMLP(3, 3, 4)
    print "start training..."
    criteria = Criteria(mlp, args.model_dir)
    criteria.cross_entropy_training(dataset.data, dataset.labels, dataset.test_data,
                                    dataset.test_labels, 50, 1000, 0.001)

    # accuracy = criteria.testing(dataset.test_data, dataset.test_labels)
    # print "accuracy = {}".format(accuracy)
    print "estimate posterior probability..."
    print mlp.forward(settings.test_samples)
    print "golden posterior probability..."
    for s in settings.test_samples:
        print GoldenClassifier.classify(s)