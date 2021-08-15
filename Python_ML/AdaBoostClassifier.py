import numpy as np
from WeightedGD import WeightedGradientDescent
from sklearn.tree import DecisionTreeClassifier

# sample_weights with shape: (T, n) => (w)i^t
# stumps with shape: (T, )          => (h)i (x)
# stump_weights with shape (T, )    => (α)t
# errors with shape: (T, )          => (ε)t
# clf.predict(X)                    => (H)i (x)


class AdaBoost:
    """ AdaBoost ensemble classifier from scratch """

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Response variable must be ±1'
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
        """ Fit the model using training data """

        X, y = self._check_X_y(X, y)
        n = X.shape[0]

        # init numpy arrays
        self.sample_weights = np.zeros(shape=(iters, n))
        self.stumps = np.zeros(shape=iters, dtype=object)
        self.stump_weights = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)

        # initialize weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(iters):
            # fit  weak learner
            curr_sample_weights = self.sample_weights[t]

            # # Use weighted gradient descent
            # stump = WeightedGradientDescent()
            # stump = stump.fit(X, y, sample_weights=curr_sample_weights)

            # Use decision tree classifier
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(X, y, sample_weight=curr_sample_weights)

            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err = curr_sample_weights[(stump_pred != y)].sum()  # / n
            stump_weight = np.log((1 - err) / err) / 2

            # update sample weights
            new_sample_weights = (
                    curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
            )

            new_sample_weights /= new_sample_weights.sum()

            # If not final iteration, update sample weights for t+1
            if t + 1 < iters:
                self.sample_weights[t + 1] = new_sample_weights

            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err

        return self

    def predict(self, X):
        """ Make predictions using already fitted model """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_preds))