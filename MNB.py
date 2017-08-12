import numpy as np
from collections import Counter
import re
from sklearn.utils import shuffle
from os import listdir

def extract_data (data_file_path, ngram_par, percentage = 100):
    data_samples = []
    with open(data_file_path, "r") as file_:          #making vector of words from the given file
        if ngram_par == 1:
            x = ([re.compile("[a-zA-Z\']*[a-zA-Z]").findall(review) for review in file_])
        elif ngram_par == 2:
            x = ([re.compile("[a-zA-Z\']*[a-zA-Z][ ][a-zA-Z\']*[a-zA-Z]").findall(review) for review in file_])
    with open(data_file_path, "r") as file_:
        y = ([re.compile("^[0-9]").findall(review) for review in file_])
    y = [int(par[0]) for par in y]
    x, y = shuffle(x, y)
    train_data_len = len(x) * percentage / 100
    return x[:train_data_len],y[:train_data_len], x[train_data_len:], y[train_data_len:]

class MNB_classifier():
    def __init__ (self, file_path, num_of_calsses = 2, ngram_par = 1, percentage = 100):
        self.ngram_par = ngram_par
        self.num_of_calsses = num_of_calsses
        (self.samples, self.lables, self.test_samples, self.test_lables) = extract_data(file_path, ngram_par, percentage)
        self.words_dic = {}
        self.data_seperated_by_classes = {i:[] for i in xrange(num_of_calsses)}
        self.words_in_seperated_calsses = {i:Counter() for i in xrange(num_of_calsses)}
        self.being_in_ith_class = {i:None for i in xrange(num_of_calsses)}
        self.words_prob_ith_class = {i:{} for i in xrange(num_of_calsses)}
        self.words_appearance_prob = {}

    def build_words_list (self):
        index = 0
        for i in xrange(len(self.samples)):
            self.data_seperated_by_classes[self.lables[i]].append(self.samples[i])
        for review in self.samples:
            for word in review:
                if not word in self.words_dic:
                    self.words_dic[word] = index
                    index += 1

    def build_samples_feature_vectors (self):
        for class_ in xrange(self.num_of_calsses):
            for review in self.data_seperated_by_classes[class_]:
                for word in review:
                    self.words_in_seperated_calsses[class_].update([word])
        for word in self.words_dic:                #for avoiding the probability "0"
            for class_ in xrange(self.num_of_calsses):
                self.words_in_seperated_calsses[class_].update([word])

    def compute_parameters (self):
        num_of_all_words = 0
        num_of_words_ith_class = {i:0 for i in xrange(self.num_of_calsses)}
        for class_ in xrange(self.num_of_calsses):
            for word in self.words_in_seperated_calsses[class_]:
                x = self.words_in_seperated_calsses[class_][word]
                num_of_words_ith_class[class_] += x
                num_of_all_words += x
        words_appearance_number = Counter()
        for class_ in xrange(self.num_of_calsses):
            words_appearance_number.update(self.words_in_seperated_calsses[class_])
        self.words_appearance_prob = {word:words_appearance_number[word] * 1.0 / num_of_all_words for word in self.words_dic}
        for class_ in xrange(self.num_of_calsses):
            self.words_prob_ith_class[class_] = {word:self.words_in_seperated_calsses[class_][word] * 1.0 / num_of_words_ith_class[class_]
                        for word in self.words_dic}
        for class_ in xrange(self.num_of_calsses):
            self.being_in_ith_class[class_] = len(self.data_seperated_by_classes[class_]) * 1.0 / len(self.samples)

    def print_predicted_result (self, data_file = None):
        counter = 0
        if data_file != None:
            (x, y ,self.test_samples, self.test_lables) = extract_data(data_file, self.ngram_par, 0)
        for i in xrange(len(self.test_samples)):
            try:
                prob = self.predict(self.test_samples[i])
                if self.test_lables[i] == prob:
                    counter += 1
            except (ZeroDivisionError) as e:
                print "Division by zero"
                pass
        print (counter * 1.0 / len(self.test_samples))

    def predict (self, review):
        ith_prob = {i:1.0 for i in xrange(self.num_of_calsses)}
        for word in review:
            if word in self.words_dic:
                for class_ in xrange(self.num_of_calsses):
                    ith_prob[class_] = (ith_prob[class_] * self.words_prob_ith_class[class_][word] * 1.0) / self.words_appearance_prob[word]
        for class_ in xrange(self.num_of_calsses):
            ith_prob[class_] = ith_prob[class_] * self.being_in_ith_class[class_]
        (result, max_prob) = (0,0)
        for class_ in xrange(self.num_of_calsses):
            if ith_prob[class_] > max_prob:
                max_prob = ith_prob[class_]
                result = class_
        return result

if __name__ == "__main__":
    datasets = [f for f in listdir('./data')]
    print 'we use 0.8 for train and 0.2 for test'
    for f in datasets:
        print 'dataset: ' + f
        print 'unigram:'
        MNB_C = MNB_classifier('./data/'+f,2, 1, 80)
        MNB_C.build_words_list()
        MNB_C.build_samples_feature_vectors()
        MNB_C.compute_parameters()
        MNB_C.print_predicted_result()
        print ''
        print 'bigram:'
        MNB_C = MNB_classifier('./data/'+f,2, 2, 80)
        MNB_C.build_words_list()
        MNB_C.build_samples_feature_vectors()
        MNB_C.compute_parameters()
        MNB_C.print_predicted_result()
        print ''
        print ''
