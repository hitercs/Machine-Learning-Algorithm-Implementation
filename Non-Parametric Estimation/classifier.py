#-*- encoding:utf-8 -*-
import kn_nearest
import parzen_window
import numpy as np
import codecs
import argparse
import matplotlib.pyplot as plt

read_buffer_size = 10000
write_buffer_size = 10000


class Util:
    @staticmethod
    def covert_to_data(fn, max_lines = 10000):
        data = []
        total = 0
        with codecs.open(fn, encoding='utf-8', mode='r', buffering=read_buffer_size) as fp:
            for line in fp:
                words = line.strip().split('\t')
                data.append(np.array([float(words[0]), float(words[1])]))
                total += 1
                if total >= max_lines:
                    break
        return np.array(data)
    @staticmethod
    def merge_data(data1, data2):
        return np.vstack((data1, data2)), [1]*data1.shape[0]+[0]*data2.shape[0]
    @staticmethod
    def draw_acc_parzen_curve(acc_mat, x_labels, legends):
        i = 0
        for points in acc_mat:
            plt.plot(x_labels, points, label=str(legends[i]))
            i += 1
        plt.ylim([0.50, 1.0])
        plt.xlabel("initial window width")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
    @staticmethod
    def draw_acc_knnearest_curve(acc_mat, x_labels, legends):
        i = 0
        for points in acc_mat:
            plt.plot(x_labels, points, label=str(legends[i]))
            i += 1
        plt.ylim([0.50, 1.0])
        plt.xlabel("data sample#")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()


class Classifier(object):
    def __init__(self, train_data1, train_data2, opt = 0, **kwargs):
        # train_data1: numpy matrix
        # train_data2: numpy matrix
        self.option = opt
        self.train_data_1 = train_data1
        self.train_data_2 = train_data2
        self.args = kwargs
        self.build_classifier()
    def build_classifier(self):
        self.p_w1 = float(self.train_data_1.shape[0])/(self.train_data_1.shape[0]+self.train_data_2.shape[0])
        self.p_w2 = 1 - self.p_w1
        if self.option == 0:
            # Parzen estimation
            self.probDensity = [parzen_window.ParzenWindow(self.train_data_1, self.args["width"]), parzen_window.ParzenWindow(self.train_data_2, self.args["width"])]
        else:
            # K Nearest estimation
            # self.probDensity = [kn_nearest.KNearest(self.train_data_1, self.args["k1"]), kn_nearest.KNearest(self.train_data_2, self.args["k1"])]
            data, labels = Util.merge_data(self.train_data_1, self.train_data_2)
            self.probDensity = kn_nearest.KNearestClassify(data, labels, self.args["k1"])
    def classify(self, x):
        if self.option == 0:
            p1 = self.p_w1 * self.probDensity[0].prob_estimate(x)
            p2 = self.p_w2 * self.probDensity[1].prob_estimate(x)
            return p1/(p1 + p2)
        else:
            return self.probDensity.prob_estimate(x)
    def report_accuracy(self, test_data1, test_data2):
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        for point in test_data1:
            if self.classify(point) > 0.5:
                true_pos += 1
            else:
                false_pos += 1
        for point in test_data2:
            if self.classify(point) < 0.5:
                true_neg += 1
            else:
                false_neg += 1
        return (true_pos + true_neg) / float(true_pos + true_neg + false_pos + false_neg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train-f1", "--train-f1", type=str, default=r"./Data/data1/train_class_1.txt")
    parser.add_argument("-train-f2", "--train-f2", type=str, default=r"./Data/data1/train_class_2.txt")
    parser.add_argument("-test-f1", "--test-f1", type=str, default=r"./Data/data1/test_class_1.txt")
    parser.add_argument("-test-f2", "--test-f2", type=str, default=r"./Data/data1/test_class_2.txt")
    parser.add_argument("-acc-report", "--acc-report", type=str, default=r"./Doc/report.txt")
    args = parser.parse_args()

    parser.add_argument("-num", "--num", type=int, default=10000)
    args = parser.parse_args()
    train_data1 = Util.covert_to_data(args.train_f1, args.num)
    train_data2 = Util.covert_to_data(args.train_f2, args.num)
    test_data1 = Util.covert_to_data(args.test_f1, args.num)
    test_data2 = Util.covert_to_data(args.test_f2, args.num)

    # data_num = [5, 10, 100]
    data_num = np.linspace(5, 500, 20)
    # init_width = np.linspace(0.1, 5, 40)
    init_width = [0.1, 0.5, 1.0, 2.0, 5.0]
    acc_parzen_mat = np.zeros((len(data_num), len(init_width)))
    init_num = [1, 2, 3]
    print "parzen window testing..."
    i = 0
    for n in data_num:
        j = 0
        for w in init_width:
            classifier = Classifier(train_data1[0:n, :], train_data2[0:n, :], 0, width=w)
            accuracy = classifier.report_accuracy(test_data1, test_data2)
            acc_parzen_mat[i][j] = accuracy
            j += 1
            print "n = {}, w = {}, classification accuracy = {}".format(n, w, accuracy)
        i += 1
    Util.draw_acc_parzen_curve(acc_parzen_mat, init_width, data_num)
    print "k nearest testing..."
    acc_kn_mat = np.zeros((len(init_num), len(data_num)))
    i = 0
    for k in init_num:
        j = 0
        for n in data_num:
            classifier = Classifier(train_data1[0:int(n), :], train_data2[0:int(n), :], 1, k1=k)
            accuracy = classifier.report_accuracy(test_data1, test_data2)
            acc_kn_mat[i][j] = accuracy
            j += 1
        i += 1
    Util.draw_acc_knnearest_curve(acc_kn_mat, data_num, init_num)

    #-------------------------------------------------------------------------------------------------
    # with codecs.open(args.acc_report, encoding='utf-8', mode='w', buffering=write_buffer_size) as report:
    #     report.write("parzen accuracy\n")
    #     for n in data_num:
    #         for w in init_width:
    #             classifier = Classifier(train_data1[0:n, :], train_data2[0:n, :], 0, width=w)
    #             accuracy = classifier.report_accuracy(test_data1, test_data2)
    #             print "n = {}, w = {}, classification accuracy = {}".format(n, w, accuracy)
    #             report.write("{}\t".format(accuracy))
    #         report.write("\n")
    #     report.write("k nearest accuracy\n")
    #     for n in data_num:
    #         for k in init_num:
    #             classifier = Classifier(train_data1[0:n, :], train_data2[0:n, :], 1, k1=k)
    #             accuracy = classifier.report_accuracy(test_data1, test_data2)
    #             print "n = {}, k = {}, classification accuracy = {}".format(n, k, accuracy)
    #             report.write("{}\t".format(accuracy))
    #         report.write("\n")



