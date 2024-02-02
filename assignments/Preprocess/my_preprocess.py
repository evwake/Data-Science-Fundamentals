import numpy as np
from copy import deepcopy
from random import randint

class my_normalizer:
    def __init__(self, norm="Min-Max", axis = 1):
        #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
        #     axis = 0: normalize rows
        #     axis = 1: normalize columns
        self.norm = norm
        self.axis = axis

    def fit(self, X):
        #     X: input matrix
        #     Calculate offsets and scalers which are used in transform()
        X_array  = np.asarray(X)
        # Write your own code below
        offsets = []
        scalers = []
        if self.axis == 0:
            for i in range(X_array.shape[self.axis]):
                offset, scaler = self.calc_norm_variables(X_array[i])
                offsets.append(offset)
                scalers.append(scaler)
        elif self.axis == 1:
            for i in range(X_array.shape[self.axis]):
                offset, scaler = self.calc_norm_variables(X_array[:, i])
                offsets.append(offset)
                scalers.append(scaler)
        self.offsets = np.asarray(offsets)
        self.scalers = np.asarray(scalers)


    def transform(self, X):
        # Transform X into X_norm
        X_norm = deepcopy(np.asarray(X))
        # Write your own code below
        if self.axis == 0:
            for i in range(X_norm.shape[self.axis]):
                X_norm[i] = X_norm[i] - self.offsets[i] / self.scalers[i]
        elif self.axis == 1:
            for i in range(X_norm.shape[self.axis]):
                X_norm[:, i] = X_norm[:, i] - self.offsets[i] / self.scalers[i]
        return X_norm


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def calc_norm_variables(self, X):
        if self.norm == "Min-Max":
            offset = X.min()
            scaler = X.max() - X.min()
        elif self.norm == "Standard_Score":
            offset = X.mean()
            scaler = X.std()
        elif self.norm == "L1":
            offset = 0
            scaler = abs(X).sum()
        elif self.norm == "L2":
            offset = 0
            scaler = ((X ** 2).sum()) ** 0.5
        return offset, scaler
            


def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: len(sample) = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )
    sample = []
    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    stratified_dict = dict()
    for i, x in enumerate(y_array):
        if not stratified_dict.get(x):
            stratified_dict[x] = []
        stratified_dict[x].append(i)
    for key in stratified_dict.keys():
        entries = stratified_dict.get(key)
        num_selections = int(np.ceil(ratio * len(entries)))
        selected = []
        if(replace):
            selected = set()
        while len(selected) < num_selections:
            selection = entries[randint(0, len(entries) - 1)]
            if(type(selected) == "set"):
                selected.add(selection)
            else:
                selected.append(selection)
        for x in selected:
            sample.append(x)


    return np.asarray(sample).astype(int)
