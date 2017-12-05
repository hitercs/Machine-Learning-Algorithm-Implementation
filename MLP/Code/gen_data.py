#-*- encoding:utf-8 -*-
import numpy as np
import codecs
import argparse
import settings
import os
np.random.seed(settings.seed)

class Dataset(object):
    def __init__(self, data_size, cov, mean):
        self.feature_num = mean.size
        self.data_size = data_size
        self.cov_mat = cov
        self.mean = mean
    def generate_data(self):
        self.data = np.random.multivariate_normal(self.mean, self.cov_mat, self.data_size)
    def print_to_file(self, fn):
        with codecs.open(fn, encoding="utf-8", mode='w', buffering=settings.write_buffer_size) as fp:
            for row in self.data:
                fp.write(u"{}\t{}\t{}\n".format(row[0], row[1], row[2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", "--data-dir", type=str, default=r"../Data/Experiment")
    args = parser.parse_args()
    print "start generating training data..."
    # train dataset
    train_set1 = Dataset(1000, settings.data_gen_parameters[0]["cov"], settings.data_gen_parameters[0]["mean"])
    train_set2 = Dataset(1000, settings.data_gen_parameters[1]["cov"], settings.data_gen_parameters[1]["mean"])
    train_set3 = Dataset(1000, settings.data_gen_parameters[2]["cov"], settings.data_gen_parameters[2]["mean"])
    train_set4 = Dataset(1000, settings.data_gen_parameters[3]["cov"], settings.data_gen_parameters[3]["mean"])

    train_set1.generate_data()
    train_set2.generate_data()
    train_set3.generate_data()
    train_set4.generate_data()

    train_set1.print_to_file(os.path.join(args.data_dir, "train_class_1.txt"))
    train_set2.print_to_file(os.path.join(args.data_dir, "train_class_2.txt"))
    train_set3.print_to_file(os.path.join(args.data_dir, "train_class_3.txt"))
    train_set4.print_to_file(os.path.join(args.data_dir, "train_class_4.txt"))

    print "start generating test data..."
    # test dataset
    test_set1 = Dataset(100, settings.data_gen_parameters[0]["cov"], settings.data_gen_parameters[0]["mean"])
    test_set2 = Dataset(100, settings.data_gen_parameters[1]["cov"], settings.data_gen_parameters[1]["mean"])
    test_set3 = Dataset(100, settings.data_gen_parameters[2]["cov"], settings.data_gen_parameters[2]["mean"])
    test_set4 = Dataset(100, settings.data_gen_parameters[3]["cov"], settings.data_gen_parameters[3]["mean"])

    test_set1.generate_data()
    test_set2.generate_data()
    test_set3.generate_data()
    test_set4.generate_data()

    test_set1.print_to_file(os.path.join(args.data_dir, "test_class_1.txt"))
    test_set2.print_to_file(os.path.join(args.data_dir, "test_class_2.txt"))
    test_set3.print_to_file(os.path.join(args.data_dir, "test_class_3.txt"))
    test_set4.print_to_file(os.path.join(args.data_dir, "test_class_4.txt"))
    print "over..."
