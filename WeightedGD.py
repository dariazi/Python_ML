import numpy as np
from scipy.optimize import fmin_tnc


class WeightedGradientDescent:

    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta, x):
        # Computes the weighted sum of inputs Similar to Linear Regression

        return np.dot(x, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class

        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y, sample_weights):
        # Computes the cost function for all the training samples, and factors in the sample weight
        m = x.shape[0]
        total_cost = -(1 / m) * sample_weigts * np.sum(
            y * np.log(self.probability(theta, x)) + (1 - y) * np.log(
                1 - self.probability(theta, x)))

        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta (derivative of cost function at theta)
        m = x.shape[0]
        return 1 / m * np.dot(x.T, sample_weight * (self.probability(self.theta, x) - y))

    def fit(self, x, y, theta = None, learning_rate = 0.01, max_iters = 100,  min_cost = 0.05, sample_weights = None):

        # initialize the sample weights if none are provided
        if sample_weights == None:
                sample_weights = np.ones(y.shape) 

        # initialize a random theta 
        if theta == None:
            theta = 1
        
        # prepare data
        X = np.c_[np.ones((X.shape[0], 1)), X]


        iter = 0
        prev_cost = self.cost_function(self, x, y,sample_weights)

        #minimizing cost function
        while iter < max_iters:
            self.theta -= learning_rate * self.gradient(self, theta, x, y)
            cost = self.cost_function()
            self.theta -= lr * self.gradient(updated_x, y, sample_weight)
            cost = self.cost_function(updated_x, y, sample_weight)
            if cost < min_cost or cost > prev_cost:
                break
            prev_cost = cost
            iters += 1

        return self
        

    def predict(self, x):
        """ Predicts the class labels
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        predicted class labels
        """
        theta = self.w_[:, np.newaxis]
        return self.probability(theta, x)

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