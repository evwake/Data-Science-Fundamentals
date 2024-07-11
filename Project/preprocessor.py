import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class preprocessor():
    def __init__(self, X, y):
        self.vectorizer =  TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        self.vectorizer = self.vectorizer.fit(X["description"])
        self.X = X
        self.y = y.reset_index()['fraudulent']
        self.feature_selection()

    def feature_selection(self):
        XX = self.vectorizer.transform(self.X["description"])
        XX = pd.DataFrame(XX.toarray())
        XX_fraud = XX.loc[(self.y[self.y==1]).index]
        fraud_sum = XX_fraud.sum().sort_values()
        XX_not_fraud = XX.loc[(self.y[self.y==0]).index]
        not_fraud_sum = XX_not_fraud.sum().sort_values()
        fraud_sum_gt_1 = fraud_sum[fraud_sum > .5]
        not_fraud_sum_gt_1 = not_fraud_sum[not_fraud_sum < 20]
        feature_set = set(fraud_sum_gt_1.index.tolist() + not_fraud_sum_gt_1.index.tolist())
        self.features = list(feature_set)

    def process_X(self, X):
        XX = self.vectorizer.transform(X["description"])
        XX = pd.DataFrame(XX.toarray())
        XX = XX[self.features]
        return XX
    