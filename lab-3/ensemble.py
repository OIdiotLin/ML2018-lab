import pickle
import numpy as np


class AdaBoostClassifier:
    """A simple AdaBoost Classifier."""

    def __init__(self, weak_classifier, n_weakers_limit, alpha=1e-6):
        """Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        """
        self.weaker_type = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.weakers_weights = []
        self.weakers = []
        self.delta = alpha

    def is_good_enough(self):
        """Optional"""
        pass

    def evaluate(self, X_train, y_train, X_val, y_val):
        """Evaluate the model on train_set and valid_set

        Args:
            X_train: An ndarray indicating the features in train_set.
            y_train: An ndarray indicating the labels in train_set.
            X_val: An ndarray indicating the fetures in valid_set.
            y_val: An ndarray indicating the labels in valid_set.

        Returns:
            (acc_train, acc_valid)
        """
        pred_train = self.predict(X_train)
        pred_val = self.predict(X_val)
        acc_train = np.count_nonzero(pred_train == y_train) / pred_train.shape[0]
        acc_val = np.count_nonzero(pred_val == y_val) / pred_val.shape[0]
        return acc_train, acc_val

    def fit(self, X, y, X_val, y_val):
        """Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        """
        n, m = X.shape
        w = np.ones(n) / n  # sample weight

        accs_train, accs_val = [], []

        for weaker_id in range(self.n_weakers_limit):
            print('training #{}...'.format(weaker_id + 1))
            weaker = self.weaker_type(max_depth=1, random_state=42)
            weaker.fit(X, y, sample_weight=w)
            self.weakers.append(weaker)
            weaker_pred = weaker.predict(X)

            error = (w * ((weaker_pred != y).astype(float))).sum()
            alpha = np.log(1 / (error + self.delta) - 1) / 2
            self.weakers_weights.append(alpha)

            # Update sample weight.
            w = w * np.exp(-y * alpha * weaker_pred)
            w = w / w.sum()

            evaluated = self.evaluate(X, y, X_val, y_val)
            accs_train.append(evaluated[0])
            accs_val.append(evaluated[1])
            print('\tAcc_train: {:.4f}\tAcc_valid: {:.4f}'.format(accs_train[-1], accs_val[-1]))

        return accs_train, accs_val

    def predict_scores(self, X):
        """Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        """
        scores = np.array([weaker.predict(X) for weaker in self.weakers])
        scores = scores * self.weakers_weights[-1]
        return scores

    def predict(self, X, threshold=0):
        """Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        """
        predict = self.predict_scores(X).sum(axis=0)
        predict -= threshold
        predict[predict <= 0] = -1
        predict[predict > 0] = 1
        return predict

    def weak_predict(self, X, classifier_id, threshold=0):
        predict = self.weakers[classifier_id].predict(X)
        predict -= threshold
        predict[predict <= 0] = -1
        predict[predict > 0] = 1
        return predict

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
