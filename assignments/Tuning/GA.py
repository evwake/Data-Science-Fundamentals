import math
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from sklearn.tree import DecisionTreeClassifier
from my_evaluation import my_evaluation

data_train = pd.read_csv("../data/breast_cancer.csv")

# Separate independent variables and dependent variables
independent = ["age","menopause","tumor-size","inv-nodes","node-caps","deg-malig",\
                   "breast","breast-quad","irradiat"]
# use one-hot encoding so discrete features can be used with DecisionTreeClassifier
X_train = pd.get_dummies(data_train[independent], drop_first=True)
y_train = data_train["Class"]
impurity_metrics = ['gini', 'entropy']

def rastrigin(X):
    dim=len(X)         

    OF=0
    for i in range (0,dim):
        OF += (X[i]**2)-10*math.cos(2*math.pi*X[i])+10

    return OF

def f(X):
    metric = impurity_metrics[int(X[0])]
    depth = int(X[1])
    clf = DecisionTreeClassifier(criterion = metric, max_depth = depth)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_train)
    metrics = my_evaluation(predictions, y_train)
    total_f = 0
    for target in clf.classes_:
                total_f += metrics.f1(target)
    return -(total_f / 2)


#rastrigin_bound=np.array([[-5.12,5.12]]*2)
#rastrigin_model=ga(function=rastrigin,dimension=2,variable_type='real',variable_boundaries=rastrigin_bound)
#rastrigin_model.run()

# criterion = 0 for gini, 1 for entropy and max_depth = 1-12 for a DecisionTreeClassifier
varbound = np.array([[0,1], [1,12]])
model=ga(function=f,dimension=2,variable_type='int',variable_boundaries=varbound)
model.run()
