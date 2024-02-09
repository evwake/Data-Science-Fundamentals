import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        self.confusion_mtx = {}
        self.correct = self.predictions == self.actuals
        for label in self.classes_:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, x in enumerate(self.actuals):
                if x == label:
                    if self.predictions[i] == label:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if self.predictions[i] == label:
                        fp += 1
                    else:
                        tn += 1
            self.confusion_mtx[label] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    def compute_class_precision(self, target):
        return self.confusion_mtx[target].get('TP') / (self.confusion_mtx[target].get('TP') + self.confusion_mtx[target].get('FP'))

    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below

        if self.confusion_matrix==None:
            self.confusion()
        if target != None:
            return self.compute_class_precision(target)
        else:
            if average == "macro":
                sum_class_precision = 0
                for label in self.classes_:
                    sum_class_precision += self.compute_class_precision(label)
                return sum_class_precision / len(self.classes_)
            elif average == "micro":
                return sum(self.correct) / len(self.correct)
            elif average == "weighted":
                sum_class_precision = 0
                for label in self.classes_:
                    sum_class_precision += self.compute_class_precision(label) * Counter(self.actuals)[label] / len(self.actuals)
                return sum_class_precision




    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()
        if target != None:
            return self.compute_class_recall(target)
        else:
            if average == "macro":
                sum_class_recall = 0
                for label in self.classes_:
                    sum_class_recall += self.compute_class_recall(label)
                return sum_class_recall / len(self.classes_)
            elif average == "micro":
                return sum(self.correct) / len(self.correct)
            elif average == "weighted":
                sum_class_recall = 0
                for label in self.classes_:
                    sum_class_recall += self.compute_class_recall(label) * Counter(self.actuals)[label] / len(self.actuals)
                return sum_class_recall
    
    def compute_class_recall(self, target):
        return self.confusion_mtx[target].get('TP') / (self.confusion_mtx[target].get('TP') + self.confusion_mtx[target].get('FN'))

    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        precision_ci = self.precision(target, average)
        recall_ci = self.recall(target, average)
        f1_score = 0
        if precision_ci + recall_ci  != 0:
            f1_score = 2 * (precision_ci * recall_ci) / (precision_ci + recall_ci)
        return f1_score
