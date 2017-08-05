import numpy as np
from collections import Counter
import re

def extract_data (data_file_path):
    data_samples = []
    with open(data_file_path, "r") as file_:          #making vector of words from the given file
        x = ([re.compile("[a-zA-Z\']*[a-zA-Z]").findall(review) for review in file_])
    with open(data_file_path, "r") as file_:
        y = ([re.compile("^[0-9]").findall(review) for review in file_])
    y = [int(par[0]) for par in y]
    return x,y

class MNB_classifier():
    def __init__ (self, file_path, num_of_calsses = 2):
        self.num_of_calsses = num_of_calsses
        (self.samples, self.lables) = extract_data(file_path)
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

    def print_predicted_result (self, data_file):
        counter = 0
        (data_vec, data_lable) = extract_data(data_file)
        for i in xrange(len(data_vec)):
            try:
                prob = self.predict(data_vec[i])
                if data_lable[i] == prob:
                    counter += 1
            except (ZeroDivisionError) as e:
                print "Division by zero"
                pass
        print (counter * 1.0 / len(data_vec))

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
    MNB_C = MNB_classifier("./data/mpqa/mpqa_t4.dat",2)
    MNB_C.build_words_list()
    MNB_C.build_samples_feature_vectors()
    MNB_C.compute_parameters()
    MNB_C.print_predicted_result("./data/mpqa/mpqa_t4.dat")
