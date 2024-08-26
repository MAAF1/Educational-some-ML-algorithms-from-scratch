import numpy as np

def prepare_data(data): # input data (features only)
    ## we can add polynomial features so the prepare_data function can be pipeline that has more than one function
    def scale_data(data): # min max scaler
        X = data
        minimum = np.array([np.min(X[:, i]) for i in range(X.shape[1])])
        maximum = np.array([np.max(X[:, i]) for i in range(X.shape[1])])
        X_scaled = X.copy()

        for i in range(X_scaled.shape[0]):
            X_scaled[i,:] = (X_scaled[i,:] - minimum) / (maximum - minimum)
        return X_scaled
    return scale_data(data)
