import numpy as np

class WeightedGradientDescent:
    def __init__(self):
        self.theta = None

    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta,  x):
        # Computes the weighted sum of inputs Similar to Linear Regression
        return np.dot(x, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class

        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self,  x, y, sample_weights):
        # Computes the cost function for all the training samples, and factors in the sample weight

        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(sample_weights * (y * np.log(self.probability(self.theta, x))
                                        + (1 - y) * np.log(1 - self.probability(self.theta, x))))
        return total_cost

    def gradient(self, x, y, sample_weights):
        # Computes the gradient of the cost function at the point theta (derivative of cost function at self.theta)
        m = x.shape[0]
        return 1 / m * np.dot(x.T, sample_weights * (self.probability(self.theta, x) - y))

    def fit(self, x, y, learning_rate = 0.01, max_iters = 100,  min_cost = 0.05, sample_weights = None):

        # initialize the sample weights if none are provided
        if sample_weights is None:
                sample_weights = np.ones(y.shape)

        # prepare data
        Y_pred = np.c_[np.ones((x.shape[0], 1)), x]
        
        # theta = [theta1, theta2, intercept]
        self.theta = np.random.randn(x.shape[1] + 1)

        y = np.array([1 if item == 1 else 0 for item in y])

        prev_cost = self.cost_function(Y_pred, y, sample_weights)

        #minimizing cost function
        for x in range(max_iters):
            self.theta -= learning_rate * self.gradient(Y_pred, y, sample_weights)
            cost = self.cost_function(Y_pred, y, sample_weights)
            if cost < min_cost or cost > prev_cost:
                break
            prev_cost = cost

        return self


    def predict(self, x):
        updated_x = np.c_[np.ones((x.shape[0], 1)), x]

        return np.array([1 if item > 0.5 else -1 for item in self.probability(self.theta, updated_x)])

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        """Computes the accuracy of the classifier
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Training samples
        actual_classes : class labels from the training data set
        probab_threshold: threshold/cutoff to categorize the samples into different classes
        Returns
        -------
        accuracy: accuracy of the model
        """
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100
