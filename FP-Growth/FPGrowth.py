#-*- encoding: utf-8 -*-
from dataset import FPGrowthDataset
import argparse
import time

class FPTreeNode(object):
    def __init__(self, itemID, count, parent):
        self.itemID = itemID
        self.count = count
        self.parent = parent
        self.childs = dict()
        self.next_item = None

    def count_inc(self, inc):
        self.count += inc

    def print_node(self, dep=1):
        print "     " * dep, "id: ", self.itemID, "count: ",self.count
        for child in self.childs.values():
            child.print_node(dep + 1)

class FPTree(object):
    def __init__(self, trans, min_sup_count):
        self.dataset = trans
        self.header_table = dict()
        self.freq_items = dict()
        self.min_sup_count = min_sup_count
        self.root = FPTreeNode("null", 0, None)

    def get_1_freq_itemset(self):
        for item in self.dataset.item_freq:
            if self.dataset.item_freq[item] >= self.min_sup_count:
                self.freq_items[item] = self.dataset.item_freq[item]

    def remove_infreq_items(self, tran):
        result = set()
        for item in tran:
            if item in self.freq_items:
                result.add(item)
        return frozenset(result)

    def construct_tree(self):
        self.get_1_freq_itemset()
        for tran, count in self.dataset.trans:
            # sort the items
            tran_filter = self.remove_infreq_items(tran)
            sort_items = sorted(list(tran_filter), key=lambda x:self.freq_items[x], reverse=True)
            self.insert_item(self.root, sort_items, count)


    def insert_item(self, root, items, count):
        if len(items) == 0:
            return
        if items[0] in root.childs:
            root.childs[items[0]].count_inc(count)
        else:
            root.childs[items[0]] = FPTreeNode(items[0], count, root)
            # update header table
            if items[0] in self.header_table:
                tail_node = self.header_table[items[0]]
                while tail_node.next_item != None:
                    tail_node = tail_node.next_item
                tail_node.next_item = root.childs[items[0]]
            else:
                self.header_table[items[0]] = root.childs[items[0]]
        self.insert_item(root.childs[items[0]], items[1:], count)

class FPGrowth(object):
    def __init__(self, fp_tree):
        self.fp_tree = fp_tree
        self.freq_itemset = dict()

    def gen_fake_trans(self, postfix, pre_tree):
        if not postfix in pre_tree.header_table:
            return []
        data = []
        cur_node = pre_tree.header_table[postfix]
        while cur_node!=None:
            count = cur_node.count
            pre_fix_path = []
            tmp_node = cur_node
            while tmp_node.parent != None:
                pre_fix_path.append(tmp_node.itemID)
                tmp_node = tmp_node.parent
            data.append((frozenset(pre_fix_path[1:]), count))
            cur_node = cur_node.next_item
        return data


    def gen_cond_fp_tree(self, postfix, pre_tree):
        fake_dataset = FPGrowthDataset()
        fake_dataset.load_dataset_from_list(self.gen_fake_trans(postfix, pre_tree))
        cond_fp_tree = FPTree(fake_dataset, self.fp_tree.min_sup_count)
        cond_fp_tree.construct_tree()
        return cond_fp_tree


    def gen_freq_itemsets(self, postfix, pre_tree):
        items = sorted(pre_tree.freq_items.items(), key=lambda x:x[1])
        for item in items:
            found_freq_itemsets = frozenset(postfix | {item[0]})
            self.freq_itemset[found_freq_itemsets] = pre_tree.freq_items[item[0]]
            cond_fp_tree = self.gen_cond_fp_tree(item[0], pre_tree)
            # print "condition fp-tree for ", found_freq_itemsets
            # cond_fp_tree.root.print_node()
            self.gen_freq_itemsets(found_freq_itemsets, cond_fp_tree)

    def print_freq_item_set(self):
        sorted_freq_item_set = sorted(self.freq_itemset.items(), key=lambda x:len(x[0]))
        print "frequent item sets: "
        for item_set in sorted_freq_item_set:
            print item_set[0], "support count", item_set[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "--fn", type=str, default=r"./trans.txt")
    parser.add_argument("-min_sup_count", "--min_sup_count", type=int, default=2)
    args = parser.parse_args()
    start_time = time.time()
    dataset = FPGrowthDataset()
    dataset.load_dataset(args.fn)
    fptree = FPTree(dataset, args.min_sup_count)
    fptree.construct_tree()
    fptree.root.print_node()
    fpGrowthSol = FPGrowth(fptree)
    fpGrowthSol.gen_freq_itemsets(frozenset([]), fpGrowthSol.fp_tree)
    end_time = time.time()
    print "total time is ", end_time - start_time
    fpGrowthSol.print_freq_item_set()