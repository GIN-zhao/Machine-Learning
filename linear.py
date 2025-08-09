import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == '__main__':
    # Generate some dummy data
    X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    y_train = np.array([2, 4, 5, 4, 5, 7, 8, 9])

    # Create and train the model
    regressor = LinearRegression(learning_rate=0.01, n_iterations=1000)
    regressor.fit(X_train, y_train)

    # Make a prediction
    test_point = np.array([[9]])
    prediction = regressor.predict(test_point)

    print("Linear Regression Model")
    print("=======================")
    print(f"Learned Weights: {regressor.weights[0]:.4f}")
    print(f"Learned Bias: {regressor.bias:.4f}")
    print(f"Prediction for x=9: {prediction[0]:.4f}")
