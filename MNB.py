import numpy as np
from collections import Counter
import re

class MNB_classifier():
    def __init__ (self, neg_file_path, pos_file_path):
        self.neg_file_path = neg_file_path
        self.pos_file_path = pos_file_path
        self.words_dic = {}
        self.pos_data = []
        self.neg_data = []
        self.pos_MNB_vec = Counter()
        self.neg_MNB_vec = Counter()
        self.index = 0
        self.parameters_k_pos = []
        self.parameters_k_neg = []
        self.being_pos_parameter = None
        self.word_and_pos_parameters = []
        self.word_and_neg_parameters = []

    def build_words_list (self):
        with open(self.pos_file_path, "r") as pos_file:
            self.pos_data = ([re.compile("[a-zA-Z\']*[a-zA-Z]").findall(review) for review in pos_file])

        with open(self.neg_file_path, "r") as neg_file:
            self.neg_data = ([re.compile("[a-zA-Z\']*[a-zA-Z]").findall(review) for review in neg_file])

        for pos_review, neg_review in zip(self.pos_data, self.neg_data):
            for pos_word, neg_word in zip(pos_review, neg_review):
                if not pos_word in self.words_dic:
                    self.words_dic[pos_word] = self.index
                    self.index += 1
                if not neg_word in self.words_dic:
                    self.words_dic[neg_word] = self.index
                    self.index += 1

    def build_samples_feature_vectors (self):
        for pos_review, neg_review in zip(self.pos_data, self.neg_data):
            for word1, word2 in zip(pos_review, neg_review):
                self.pos_MNB_vec.update(word1)
                self.neg_MNB_vec.update(word2)
        for word in self.words_dic:
            self.pos_MNB_vec.update([word])
            self.neg_MNB_vec.update([word])
        print len(self.pos_MNB_vec), len(self.neg_MNB_vec), len(self.words_dic)

    def compute_parameters (self):
        num_of_pos_words = 0
        num_of_neg_words = 0
        for word in self.pos_MNB_vec:
            num_of_pos_words += self.pos_MNB_vec[word]
        print num_of_pos_words
        for word in self.neg_MNB_vec:
            num_of_neg_words += self.neg_MNB_vec[word]
        print num_of_neg_words
        self.being_pos_parameter = len(self.pos_MNB_vec)*1.0 / (len(self.pos_MNB_vec) + len(self.neg_MNB_vec))
        self.word_and_pos_parameters = np.array([self.pos_MNB_vec[word]/1000 for word in self.words_dic])
        self.word_and_neg_parameters = np.array([self.neg_MNB_vec[word]/1000 for word in self.words_dic])
        # TODO: sum of num of words in positive samples instead of 1000

    # TODO: predict function

if __name__ == "__main__":
    MNB_C = MNB_classifier("./data/train-neg.txt", "./data/train-pos.txt")
    MNB_C.build_words_list()
    MNB_C.build_samples_feature_vectors()
    MNB_C.compute_parameters()
