import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # Calculate P(yj) and P(xi|yj)        
        yj_counter = Counter(y)
        self.P_y = {}
        self.P = {}
        for yj in self.classes_:
            self.P[yj] = {}
            n = yj_counter[yj]
            self.P_y[yj] = n / len(y)
            for Xi in X.columns:
                self.P[yj][Xi] = {}
                all_possible_values = set(X[Xi])
                v = len(all_possible_values)
                nc = Counter(X[Xi][y==yj])
                for xi in all_possible_values:
                    count = 0
                    if xi in nc:
                        count = nc[xi]
                    self.P[yj][Xi][xi] = (count + self.alpha) / (n + v * self.alpha)

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # P(yj|x) = P(x|yj)P(yj)
        # P(x|yj) = P(x1|yj)P(x2|yj)...P(xk|yj) = self.P[yj][X1][x1]*self.P[yj][X2][x2]*...*self.P[yj][Xk][xk]
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs
