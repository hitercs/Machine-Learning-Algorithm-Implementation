#-*- encoding:utf-8 -*-
import codecs
import settings
from util import Util

class Dataset(object):
    def __init__(self):
        self.trans = []
        self.trans_num = 0
        self.trans_length = 0
        self.distinct_items = set()
        self.item_freq = dict()

    def load_dataset(self, fn):
        with codecs.open(fn, encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                items = line.strip().split()
                # items = [int(item) for item in items]
                transaction = frozenset(items)
                self.trans.append(transaction)
                self.trans_num += 1
                self.trans_length = max(self.trans_length, len(transaction))
                for item in items:
                    self.distinct_items.add(item)
                    Util.add_vocab(self.item_freq, item)

    def load_dataset_from_list(self, trans_list):
        for tran in trans_list:
            self.trans.append(frozenset(tran))
            self.trans_num += 1
            self.trans_length = max(self.trans_length, len(tran))
            for item in tran:
                self.distinct_items.add(item)
                Util.add_vocab(self.item_freq, item)

class FPGrowthDataset(object):
    def __init__(self):
        self.trans = []
        self.trans_num = 0
        self.trans_length = 0
        self.distinct_items = set()
        self.item_freq = dict()

    def load_dataset(self, fn):
        with codecs.open(fn, encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                items = line.strip().split()
                # items = [int(item) for item in items]
                transaction = frozenset(items)
                self.trans.append((transaction, 1))
                self.trans_num += 1
                self.trans_length = max(self.trans_length, len(transaction))
                for item in items:
                    self.distinct_items.add(item)
                    Util.add_vocab(self.item_freq, item)

    def load_dataset_from_list(self, trans_list):
        for tran, count in trans_list:
            self.trans.append((frozenset(tran), count))
            self.trans_num += count
            self.trans_length = max(self.trans_length, len(tran))
            for item in tran:
                self.distinct_items.add(item)
                Util.add_vocab_count(self.item_freq, item, count)
