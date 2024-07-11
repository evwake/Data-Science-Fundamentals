import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan"}
        # p value only matters when metric = "minkowski"
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions
    
    def calculate_distances(self, x):
        distances = []
        if self.metric == 'minkowski':
            for point in self.X.iterrows():
                idx = point[0]
                sum = 0
                neighbor = point[1]
                for i, value in enumerate(neighbor):
                    sum += (x.iloc[i] - value) ** self.p
                distances.append((self.y[idx], sum ** (1/self.p)))
        elif self.metric == 'euclidean':
            for point in self.X.iterrows():
                idx = point[0]
                sum = 0
                neighbor = point[1]
                for i, value in enumerate(neighbor):
                    sum += (x.iloc[i] - value) ** 2
                distances.append((self.y[idx], sum ** 0.5))
        elif self.metric == 'manhattan':
            for point in self.X.iterrows():
                idx = point[0]
                sum = 0
                neighbor = point[1]
                for i, value in enumerate(neighbor):
                    sum += (x.iloc[i] - value)
                distances.append((self.y[idx], sum))
        return distances
        

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs_list = []
        for row in X.iterrows():
            prob_dict = {}
            for label in self.classes_:
                prob_dict[label] = 0
            x = row[1]
            distances = self.calculate_distances(x)
            nearest_neighbors = sorted(distances, key= lambda x: x[1])[:self.n_neighbors]
            for neighbor in nearest_neighbors:
                prob_dict[neighbor[0]] += (1 / self.n_neighbors)
            probs_list.append(prob_dict)
        probs = pd.DataFrame(probs_list, columns=self.classes_)
        return probs
