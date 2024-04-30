import time
import sys
sys.path.insert(0,'../..')
from my_evaluation import my_evaluation
from sklearn.neighbors import KNeighborsClassifier
from preprocessor import preprocessor
from random_oversampler import random_oversample
from sklearn.model_selection import GridSearchCV

class my_model():

    def obj_func(self, predictions, actuals, pred_proba=None):
        # One objectives: higher f1 score
        eval = my_evaluation(predictions, actuals, pred_proba)
        return [eval.f1()]

    def fit(self, X, y):
        # do not exceed 29 mins
        self.start_timestamp = time.time()
        self.preprocessor = preprocessor(X, y)
        XX = self.preprocessor.process_X(X)
        self.preprocessing_timestamp = time.time()
        parameters = {'n_neighbors':[1, 2, 3, 4, 5]}
        #parameters = {'loss':('hinge', 'log_loss'),
                        #'penalty':('l2', 'l1', 'elasticnet')
                    #}
        knn = KNeighborsClassifier()
        #sgd = SGDClassifier()
        self.clf = GridSearchCV(knn, parameters, scoring='f1')
        self.tuning_timestamp = time.time()
        oversampled_X, oversampled_y = random_oversample(XX, y)
        self.clf.fit(oversampled_X, oversampled_y)
        self.fitting_timestamp = time.time()
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        XX = self.preprocessor.process_X(X)
        predictions = self.clf.predict(XX)
        self.testing_timestamp = time.time()
        print("Preprocessing Time:", (self.preprocessing_timestamp - self.start_timestamp) / 60)
        print("Tuning Time:", (self.tuning_timestamp - self.preprocessing_timestamp) / 60)
        print("Fitting Time:", (self.fitting_timestamp - self.tuning_timestamp) / 60)
        print("Testing Time:", (self.testing_timestamp - self.fitting_timestamp) / 60)
        return predictions
