import numpy as np
import pandas as pd

def random_oversample(X, y):
    y = y.reset_index()['fraudulent']
    count_diff = len(y[y==0]) - len(y[y==1])
    selected = np.random.choice(y[y==1].index, size=count_diff, replace=True)
    new_X = pd.concat([X, X.loc[selected]])
    new_y = pd.concat([y, y[selected]])
    return new_X, new_y