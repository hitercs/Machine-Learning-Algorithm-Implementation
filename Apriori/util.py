#-*- encoding:utf-8 -*-

class Util:
    @staticmethod
    def add_vocab(vocab, key):
        if not key in vocab:
            vocab[key] = 1
        else:
            vocab[key] += 1
    @staticmethod
    def add_vocab_count(vocab, key, count):
        if not key in vocab:
            vocab[key] = count
        else:
            vocab[key] += count
