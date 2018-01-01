#-*- encoding:utf-8 -*-
import argparse
from dataset import Dataset
import time

class Apriori(object):
    def __init__(self, trans, min_sup):
        self.dataset = trans
        self.min_sup = min_sup
        self.min_sup_count = self.min_sup * trans.trans_num
        self.freq_itemset_support_count = dict()

    def contain_infreq_itemsets(self, itemset, freq_k_1_itemsets):
        for item in itemset:
            subset = itemset - {item}
            if not subset in freq_k_1_itemsets:
                return True
        return False

    def prune_k_1_infreq_itemset(self, freq_k_1_itemsets, candidate_k_itemsets):
        new_candidates = []
        for itemset in candidate_k_itemsets:
            if not self.contain_infreq_itemsets(itemset, frozenset(freq_k_1_itemsets)):
                new_candidates.append(itemset)
        return new_candidates

    def gen_freq_itemsets(self):
        self.freq_itemsets = []
        k_1_freq_itemsets = self.get_1_freq_itemsets()
        self.freq_itemsets.append(k_1_freq_itemsets)
        while len(k_1_freq_itemsets) > 0:
            # candidate generation
            k_freq_itemsets = self.gen_k_itemsets(k_1_freq_itemsets)
            # candidate pruning
            k_freq_itemsets_pruned = self.prune_k_1_infreq_itemset(k_1_freq_itemsets, k_freq_itemsets)
            # support counting
            final_k_freq_itemsets_pruned = self.scan_db_pruning(k_freq_itemsets_pruned)
            self.freq_itemsets.append(final_k_freq_itemsets_pruned)
            k_1_freq_itemsets = final_k_freq_itemsets_pruned
        return self.freq_itemsets


    def scan_db_pruning(self, freq_itemsets):
        new_candidates = []
        for item_set in freq_itemsets:
            freq = self.scan_itemset_freq(item_set)
            if freq >= self.min_sup_count:
                new_candidates.append(item_set)
                self.freq_itemset_support_count[item_set] = freq
        return new_candidates


    def scan_itemset_freq(self, item_set):
        freq = 0
        for transcation in self.dataset.trans:
            if item_set.issubset(transcation):
                freq += 1
        return freq


    def scan_item_freq(self):
        freq_item_set = set()
        for item in self.dataset.item_freq:
            if self.dataset.item_freq[item] >= self.min_sup_count:
                freq_item_set.add(item)
                self.freq_itemset_support_count[frozenset({item})] = self.dataset.item_freq[item]
        return frozenset(freq_item_set)

    def get_1_freq_itemsets(self):
        freq_1_item_set = self.scan_item_freq()
        freq_itemsets = []
        for item in freq_1_item_set:
            freq_itemsets.append(frozenset({item}))
        return freq_itemsets

    def gen_k_itemsets(self, freq_k_1_itemsets):
        item_set_size = len(freq_k_1_itemsets)
        freq_k_itemsets = []
        for i in range(item_set_size):
            for j in range(i+1, item_set_size):
                l1 = list(freq_k_1_itemsets[i])
                l2 = list(freq_k_1_itemsets[j])
                l1.sort()
                l2.sort()
                if l1[0:-1] == l2[0:-1]:
                    freq_k_itemsets.append(frozenset(freq_k_1_itemsets[i] | freq_k_1_itemsets[j]))
        return freq_k_itemsets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "--fn", type=str, default=r"./trans.txt")
    parser.add_argument("-min_sup_count", "--min_sup_count", type=int, default=2)
    args = parser.parse_args()
    start_time = time.time()
    dataset = Dataset()
    dataset.load_dataset(args.fn)
    apriori = Apriori(dataset, float(args.min_sup_count)/dataset.trans_num)
    freq_item_sets = apriori.gen_freq_itemsets()
    end_time = time.time()
    print "total time is ", end_time - start_time
    print "frequent item sets are: "
    k = 1
    for k_itemset in freq_item_sets:
        if k_itemset != []:
            print "frequent %d item sets:" % k
            for item in k_itemset:
                print item, "support count: ", apriori.freq_itemset_support_count[item]
        k += 1
