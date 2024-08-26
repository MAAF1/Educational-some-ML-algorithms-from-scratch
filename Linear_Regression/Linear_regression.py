import numpy as np
from data_preparation import prepare_data
from sklearn.datasets import make_regression

class LinearModel:
    def __init__(self, weights, learning_rate = 0.01, precision = 0.0001 , max_iterations = 10000, fit_intercept_option = True) -> None:
        self.weights = weights
        self.learning_rate = learning_rate
        self.precision = precision
        self.max_iter = max_iterations
        self.fit_intercept = fit_intercept_option


    def _learning_process(self, X, y_true, weights, learning_rate, precision, max_iters, interception):
        if interception == True:
            X = np.c_[np.ones(X.shape[0]),X]
        m = X.shape[0]
        initial_weights = weights
        last_weight = initial_weights + 10
        weight_hist, cost_hist, iteration_history = [],  [], []
        cur_weight = initial_weights.copy()
        def cost_func(w):
            y_pdnew = np.dot(X, w)
            cost = 0
            for i in range(m):
                err = ((y_true[i] - y_pdnew[i]) ** 2)
                cost += err

            return float((1/(2*m)) * cost)
        def gradients(weights): ## partial derivatives of the MSE function based on each feature (gradients)
            y_pd = np.dot(X,weights)
            g = 0
            for i in range(m):
                grad = (y_pd[i] - y_true[i]) * X[i]
                g+= grad
            return (1/m )* g
        iter = 0
        while np.linalg.norm(last_weight - cur_weight) > precision and iter < max_iters:
        
            last_weight = cur_weight 
            weight_hist.append(last_weight) ## history of our parameters
            cost_hist.append(cost_func(last_weight)) ## cost history with each iteration
            iteration_history.append(iter) ## to show which iteration we stopped
            cur_weight = cur_weight - (learning_rate *gradients(cur_weight)) ## the update of weights operation
            self.weights = cur_weight
            cost = cost_func(last_weight) ## new update means new cost and error update
            iter += 1
        # you can check for iteration history or any other variables
    def fit(self,X,y_true):
        self._learning_process(X,y_true,self.weights, self.learning_rate, self.precision, self.max_iter, self.fit_intercept)
        return self.weights

    def predict(self,data):
        X = data
        if self.fit_intercept == True:
            X = np.c_[np.ones(X.shape[0]),X]
        self.preds = np.dot(X,self.weights)
        return self.preds
    def cost_func(self, y_true,preds):
        y_pd = preds
        cost = 0
        m = y_true.shape[0]
        for i in range(m):
            err = ((y_true[i] - y_pd[i]) ** 2)
            cost += err
        return float((1/(2*m)) * cost)
        
    
if __name__ == "__main__":
    X, y = make_regression(n_samples=200, n_features=5, noise=1, random_state=42)
    new_X = prepare_data(X)
    weights = np.array([1,1,1,1,1,1]) ## X has 5 features and the weight are 5 values this means interception
    def trial_1():
     ## use same hyper parameters
        model = LinearModel(weights)
        model.fit(new_X,y)
        pred = model.predict(new_X)
        cost = model.cost_func(y,pred)
        print(cost)
    
    def trial_2():
        model = LinearModel(weights,0.01,0.00001,100000)
        model.fit(new_X,y)
        pred = model.predict(new_X)
        cost = model.cost_func(y,pred)
        print(cost)
    #trial_1()
    trial_2()

